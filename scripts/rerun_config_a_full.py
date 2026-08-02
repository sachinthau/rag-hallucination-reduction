"""
scripts/rerun_config_a_full.py

Regenerates ALL 200 questions through Config A from scratch.

Config A never calls retrieve_chunks(), so it's completely unaffected by
the retriever fix — this is just a plain re-run because the previous
merged 200-row results_config_A.csv appears to have been lost/overwritten
(only a 51-row file remains). Since Config A is just a direct LLM call
with no retrieval or GRV validation, this is fast and cheap to regenerate
cleanly rather than trying to reconstruct the old merge.

Run from your project root or from scripts/:
    python scripts/rerun_config_a_full.py
"""

import json
import os
import sys
import csv
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.pipeline.config_a import query as run_config_a

DATASET_PATH = os.path.join(_PROJECT_ROOT, "data", "questions", "qa_dataset.json")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results")

SLEEP_BETWEEN_CALLS = 1.0


def load_all_questions():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    questions = load_all_questions()
    print(f"Loaded {len(questions)} total questions.")
    print("Regenerating Config A (no retrieval involved, fast/cheap)...\n")

    rows = []
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question_text = q["question"]
        print(f"[{i}/{len(questions)}] {qid}: {question_text[:60]}...")

        try:
            result = run_config_a(question_text)
        except Exception as e:
            print(f"  ERROR on {qid}: {e}")
            result = {"config": "A", "question": question_text, "answer": f"ERROR: {e}"}

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

    out_path = os.path.join(OUTPUT_DIR, "results_config_A.csv")
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
    print("Config A regeneration complete.")


if __name__ == "__main__":
    main()