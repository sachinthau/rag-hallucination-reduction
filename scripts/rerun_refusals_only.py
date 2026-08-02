"""
scripts/rerun_refusals_only.py

Re-runs ONLY the 54 refusal-type questions from results_config_C_postfix.csv
through Config C again, now that:
  1. retriever_expanded.py has been fixed to use the azure-search-documents
     SDK directly (genuine @search.reranker_score, not LangChain's broken
     score field)
  2. calibrate_abstention_threshold.py has been re-run with the fixed
     scoring, producing a valid tau (2.8514)

All 54 refusal questions are out-of-corpus (Q141-Q200), so this does NOT
need to touch Config A or Config B at all - those are unaffected by the
abstention verification path (which only exists inside config_c.py's call
to grv.validate()).

Run from your project root or from scripts/:
    python scripts/rerun_refusals_only.py
"""

import json
import os
import sys
import csv
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.pipeline.config_c import query as run_config_c

DATASET_PATH = os.path.join(_PROJECT_ROOT, "data", "questions", "qa_dataset.json")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results")

REFUSAL_IDS = [
    "Q141", "Q142", "Q144", "Q145", "Q148", "Q149", "Q150", "Q151", "Q152",
    "Q153", "Q154", "Q155", "Q157", "Q158", "Q159", "Q160", "Q161", "Q162",
    "Q163", "Q164", "Q165", "Q166", "Q167", "Q169", "Q170", "Q171", "Q172",
    "Q173", "Q174", "Q175", "Q176", "Q177", "Q178", "Q179", "Q180", "Q181",
    "Q182", "Q183", "Q184", "Q185", "Q186", "Q187", "Q188", "Q189", "Q190",
    "Q191", "Q192", "Q193", "Q195", "Q196", "Q197", "Q198", "Q199", "Q200",
]

SLEEP_BETWEEN_CALLS = 1.0


def load_target_questions():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
    qmap = {q["id"]: q for q in all_questions}
    selected = [qmap[qid] for qid in REFUSAL_IDS if qid in qmap]
    missing = [qid for qid in REFUSAL_IDS if qid not in qmap]
    if missing:
        print(f"WARNING: IDs not found in dataset: {missing}")
    print(f"Loaded {len(selected)} / {len(REFUSAL_IDS)} target questions.")
    return selected


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    questions = load_target_questions()

    if not questions:
        print("No matching questions found.")
        return

    rows = []
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question_text = q["question"]
        print(f"[{i}/{len(questions)}] {qid}: {question_text[:60]}...")

        try:
            result = run_config_c(question_text)
        except Exception as e:
            print(f"  ERROR on {qid}: {e}")
            result = {"config": "C", "question": question_text, "answer": f"ERROR: {e}"}

        row = {
            "question_id": qid,
            "expected_answer": q.get("expected_answer"),
            "source_doc": q.get("source_doc"),
            "in_corpus": q.get("in_corpus"),
            "category": q.get("category"),
        }
        row.update(result)
        rows.append(row)

        time.sleep(SLEEP_BETWEEN_CALLS)

    out_path = os.path.join(OUTPUT_DIR, "results_config_C_refusals_refixed.csv")
    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")
    print("Next: upload this file so it can be merged into")
    print("results_config_C_postfix.csv, replacing the 54 refusal rows")
    print("that were previously scored with the broken abstention logic.")


if __name__ == "__main__":
    main()