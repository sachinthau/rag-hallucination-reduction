"""
Second-pass verification. The first script (verify_dataset.py) flagged 75
questions using whole-passage fuzzy matching, but many source_passage entries
are paraphrased summaries rather than verbatim quotes, so that method produces
false positives on genuinely accurate content.

This script instead extracts the "factual anchor" from each flagged question,
things like:
  - inline code / setting values in backticks or quotes (e.g. `app.EventHub()`)
  - numbers with units (e.g. "10 minutes", "90 days", "3500")
  - the expected_answer field itself

...and checks whether that anchor literally appears anywhere in the matched
file (or, if no file was found, anywhere in the whole corpus). This is a much
more reliable fabrication test: a paraphrased passage will still contain the
same key numbers/terms as the real source, but a genuinely fabricated one
won't.

Run from scripts/evaluation/ (same folder as verify_dataset.py):
    python verify_dataset_v2.py
"""

import json
import os
import re
import glob

def find_project_root(start_path):
    current = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(current, "data", "corpus")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError("Could not locate project root (data/corpus/ not found).")
        current = parent

PROJECT_ROOT = find_project_root(os.path.dirname(__file__) or ".")
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "questions", "qa_dataset.json")
CORPUS_ROOT = os.path.join(PROJECT_ROOT, "data", "corpus")
PREV_RESULTS = os.path.join(os.path.dirname(__file__) or ".", "dataset_verification_results.json")
OUT_PATH = os.path.join(os.path.dirname(__file__) or ".", "dataset_verification_v2_results.json")

def normalize(text):
    return " ".join(text.split()).lower()

def load_all_corpus_text():
    """Load every corpus file once into a dict {path: content} for fast repeat searching."""
    files = glob.glob(os.path.join(CORPUS_ROOT, "**", "*.md"), recursive=True)
    corpus = {}
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                corpus[f] = fh.read()
        except Exception:
            continue
    return corpus

def extract_anchors(passage, expected_answer):
    """
    Pull out short, specific, checkable tokens from the passage and expected
    answer: backtick code, quoted values, numbers with a unit word nearby,
    and the expected_answer itself if it's short enough to be a literal term.
    """
    anchors = set()

    # backtick code spans, e.g. `app.EventHub()`
    anchors.update(re.findall(r"`([^`]+)`", passage))
    anchors.update(re.findall(r"`([^`]+)`", expected_answer or ""))

    # numbers with a unit word nearby (minutes, seconds, days, MB, ms, etc.)
    for m in re.finditer(r"(\d[\d,]*\s?(?:minutes?|seconds?|days?|hours?|ms|MB|GB|instances?|partitions?|containers?))", passage, re.IGNORECASE):
        anchors.add(m.group(1))

    # standalone numbers (e.g. port numbers like 3500, 50001)
    anchors.update(re.findall(r"\b\d{2,6}\b", passage))

    # expected_answer itself, if short and specific (avoid adding whole sentences)
    if expected_answer and len(expected_answer) < 60:
        anchors.add(expected_answer.strip().rstrip("."))

    # clean up
    anchors = {a.strip() for a in anchors if a and len(a.strip()) > 1}
    return anchors

def anchor_found(anchor, corpus_text_norm):
    return normalize(anchor) in corpus_text_norm

def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        questions = {q["id"]: q for q in json.load(f)}

    with open(PREV_RESULTS, "r", encoding="utf-8") as f:
        prev = json.load(f)

    flagged_ids = [iss["id"] for iss in prev["issues"]]
    print(f"Re-checking {len(flagged_ids)} previously flagged questions using factual anchors...\n")

    corpus = load_all_corpus_text()
    # pre-normalize once for speed
    corpus_norm = {path: normalize(text) for path, text in corpus.items()}
    whole_corpus_norm = " ".join(corpus_norm.values())

    likely_ok = []
    likely_fabricated = []
    no_anchors_found_to_check = []

    for qid in flagged_ids:
        q = questions.get(qid)
        if not q:
            continue
        passage = q.get("source_passage", "")
        expected = q.get("expected_answer", "")
        anchors = extract_anchors(passage, expected)

        if not anchors:
            no_anchors_found_to_check.append({
                "id": qid, "question": q.get("question", ""),
                "reason": "No extractable factual anchor (numbers/code/short answer) to check automatically."
            })
            continue

        found_any = any(anchor_found(a, whole_corpus_norm) for a in anchors)

        record = {
            "id": qid,
            "question": q.get("question", ""),
            "expected_answer": expected,
            "anchors_checked": list(anchors),
            "source_doc": q.get("source_doc", ""),
        }

        if found_any:
            likely_ok.append(record)
        else:
            likely_fabricated.append(record)

    print(f"LIKELY OK (paraphrased but factually present): {len(likely_ok)}")
    print(f"LIKELY FABRICATED (no factual anchor found anywhere in corpus): {len(likely_fabricated)}")
    print(f"NEEDS MANUAL CHECK (no clean anchor to test automatically): {len(no_anchors_found_to_check)}\n")

    print("=" * 90)
    print("LIKELY FABRICATED — review these first, highest priority")
    print("=" * 90)
    for r in likely_fabricated:
        print(f"ID: {r['id']}  |  Question: {r['question']}")
        print(f"  Expected answer: {r['expected_answer']}")
        print(f"  Anchors checked (none found in corpus): {r['anchors_checked']}")
        print(f"  Recorded source_doc: {r['source_doc']}")
        print("-" * 90)

    print("\n" + "=" * 90)
    print("NEEDS MANUAL CHECK — no automatic anchor available")
    print("=" * 90)
    for r in no_anchors_found_to_check:
        print(f"ID: {r['id']}  |  Question: {r['question']}")
        print(f"  Reason: {r['reason']}")
        print("-" * 90)

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        json.dump({
            "likely_ok_count": len(likely_ok),
            "likely_fabricated_count": len(likely_fabricated),
            "needs_manual_check_count": len(no_anchors_found_to_check),
            "likely_ok": likely_ok,
            "likely_fabricated": likely_fabricated,
            "needs_manual_check": no_anchors_found_to_check,
        }, out, indent=2)

    print(f"\nFull results written to {OUT_PATH}")

if __name__ == "__main__":
    main()