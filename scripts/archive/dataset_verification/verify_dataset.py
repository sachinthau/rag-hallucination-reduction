"""
Verifies every in-corpus question in qa_dataset.json against the actual
ingested corpus files. Checks two things per question:

1. Does source_doc actually exist somewhere under data/corpus/?
2. Does source_passage (or a close match of it) actually appear in that file,
   or anywhere in the corpus if the exact file can't be found?

Run from your rag-project root:
    python verify_dataset.py

Requires: pip install rapidfuzz --break-system-packages
(rapidfuzz gives a fuzzy match score so we catch near-matches, not just exact
substring matches, since your passages may have minor whitespace/newline
differences from the raw markdown)
"""

import json
import os
import glob
from difflib import SequenceMatcher

DATASET_PATH = "../data/questions/qa_dataset.json"
CORPUS_ROOT = "../data/corpus"
FUZZY_MATCH_THRESHOLD = 0.75  # below this, flag as likely fabricated/mismatched

def normalize(text):
    return " ".join(text.split()).lower()

def find_file(source_doc):
    """Search recursively under CORPUS_ROOT for a file matching source_doc's basename."""
    basename = os.path.basename(source_doc)
    matches = glob.glob(os.path.join(CORPUS_ROOT, "**", basename), recursive=True)
    return matches

def best_fuzzy_match(passage, full_text):
    """
    Slide the passage length across full_text in chunks and return the best
    similarity ratio found. This catches passages that exist but are chunked
    or reformatted slightly differently than recorded.
    """
    passage_norm = normalize(passage)
    text_norm = normalize(full_text)
    if passage_norm in text_norm:
        return 1.0
    # fallback: whole-file similarity against a sliding window roughly the
    # size of the passage, stepping through the document
    window = max(len(passage_norm), 50)
    step = max(window // 2, 20)
    best = 0.0
    for i in range(0, max(len(text_norm) - window, 1), step):
        chunk = text_norm[i:i+window]
        ratio = SequenceMatcher(None, passage_norm, chunk).ratio()
        if ratio > best:
            best = ratio
    return best

def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    in_corpus_qs = [q for q in questions if q.get("in_corpus")]
    print(f"Checking {len(in_corpus_qs)} in-corpus questions against {CORPUS_ROOT} ...\n")

    issues = []
    ok_count = 0

    for q in in_corpus_qs:
        qid = q.get("id", "UNKNOWN")
        source_doc = q.get("source_doc", "")
        passage = q.get("source_passage", "")

        file_matches = find_file(source_doc)

        if not file_matches:
            issues.append({
                "id": qid,
                "issue": "SOURCE_DOC_NOT_FOUND",
                "detail": f"No file matching '{source_doc}' found under {CORPUS_ROOT}",
                "question": q.get("question", ""),
            })
            continue

        # check passage against the matched file(s); if multiple matches, check all
        best_score = 0.0
        matched_file = None
        for fpath in file_matches:
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as cf:
                    content = cf.read()
            except Exception as e:
                continue
            score = best_fuzzy_match(passage, content)
            if score > best_score:
                best_score = score
                matched_file = fpath

        if best_score < FUZZY_MATCH_THRESHOLD:
            issues.append({
                "id": qid,
                "issue": "PASSAGE_MISMATCH",
                "detail": f"Best fuzzy match score {best_score:.2f} in {matched_file} (threshold {FUZZY_MATCH_THRESHOLD})",
                "question": q.get("question", ""),
                "recorded_passage": passage[:150],
            })
        else:
            ok_count += 1

    print(f"OK: {ok_count} / {len(in_corpus_qs)} in-corpus questions verified cleanly.\n")

    if issues:
        print(f"FLAGGED: {len(issues)} questions need manual review:\n")
        print("=" * 90)
        for iss in issues:
            print(f"ID: {iss['id']}  |  Issue: {iss['issue']}")
            print(f"Question: {iss['question']}")
            print(f"Detail: {iss['detail']}")
            if "recorded_passage" in iss:
                print(f"Recorded passage (truncated): {iss['recorded_passage']}")
            print("-" * 90)
    else:
        print("No issues found. All in-corpus questions verified.")

    # write results to a JSON file for easy reference
    with open("dataset_verification_results.json", "w", encoding="utf-8") as out:
        json.dump({
            "total_checked": len(in_corpus_qs),
            "ok_count": ok_count,
            "flagged_count": len(issues),
            "issues": issues,
        }, out, indent=2)

    print(f"\nFull results also written to dataset_verification_results.json")

if __name__ == "__main__":
    main()