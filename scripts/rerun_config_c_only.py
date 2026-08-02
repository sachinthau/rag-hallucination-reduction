"""
Re-runs Config C ONLY (not A or B) for a small set of specific question IDs
that showed unusually long RAGAS evaluation times in the previous run.

By default this targets just the two genuine outliers:
  - Q115 (took 5m19s)
  - Q118 (took 19m14s with a TimeoutError before completing)

Edit TARGET_IDS below if you also want to include Q089 and Q137 for extra
safety, even though those were within normal range.

Run from your project root or from scripts/:
    python scripts/rerun_config_c_only.py
"""

import json
import os
import sys
import csv
import time

_PROJECT_ROOT_FOR_IMPORT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_FOR_IMPORT)

from src.pipeline.config_c import query as run_config_c

PROJECT_ROOT = _PROJECT_ROOT_FOR_IMPORT
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "questions", "qa_dataset.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

# --- EDIT THIS if you want to include Q089 and Q137 as well ---
# TARGET_IDS = {"Q115", "Q118"}
TARGET_IDS = {"Q089", "Q115", "Q118", "Q137"}  # <- uncomment for all four

SLEEP_BETWEEN_CALLS = 1.0


def load_target_questions():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
    filtered = [q for q in all_questions if q["id"] in TARGET_IDS]
    missing = TARGET_IDS - {q["id"] for q in filtered}
    if missing:
        print(f"WARNING: IDs not found in dataset: {missing}")
    print(f"Loaded {len(filtered)} / {len(TARGET_IDS)} target questions.")
    return filtered


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    questions = load_target_questions()

    if not questions:
        print("No matching questions found. Check TARGET_IDS.")
        return

    rows = []
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question_text = q["question"]
        print(f"\n[{i}/{len(questions)}] Re-running Config C for {qid}: {question_text[:70]}...")

        start = time.time()
        try:
            result = run_config_c(question_text)
        except Exception as e:
            print(f"  ERROR on {qid}: {e}")
            result = {"config": "C", "question": question_text, "answer": f"ERROR: {e}"}
        elapsed = time.time() - start
        print(f"  Completed in {elapsed:.1f}s")

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

    # Write to a small dedicated file, separate from the main rerun batch,
    # so it's clear these are the re-re-run rows for the slow questions.
    out_path = os.path.join(OUTPUT_DIR, "results_config_C_slowfix.csv")
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
    print("Next: send this file back — I'll fold these rows into your")
    print("results_config_C_rerun.csv (replacing the slow-timeout rows),")
    print("then merge everything into the final 200-row result sets.")


if __name__ == "__main__":
    main()