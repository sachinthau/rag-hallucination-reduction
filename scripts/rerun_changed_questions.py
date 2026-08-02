"""
Re-runs only the 53 changed/replaced question IDs through Config A, B, and C,
saving results to separate "_rerun" CSVs so your original 200-row results
stay untouched.

Run from your project root:
    python scripts/rerun_changed_questions.py
"""

import json
import os
import sys
import csv
import time

# Add project root to sys.path so `import src....` works regardless of
# whether this script is run from the project root or from scripts/.
_PROJECT_ROOT_FOR_IMPORT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_FOR_IMPORT)

# =========================================================================
# Pipeline imports - matches src/pipeline/config_a.py, config_b.py, config_c.py
# Each module exposes a query(question: str) -> dict function that already
# returns all the fields your original results CSVs use (config, question,
# answer, retrieved_chunks, latency_ms, grv_score, grv_label,
# grv_layer_scores, flagged, ragas_faithfulness, cross_encoder_score,
# reranker_score).
# =========================================================================
from src.pipeline.config_a import query as run_config_a
from src.pipeline.config_b import query as run_config_b
from src.pipeline.config_c import query as run_config_c

# =========================================================================
# Configuration
# =========================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "questions", "qa_dataset.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

CHANGED_IDS = {
    "Q082", "Q085", "Q087", "Q088", "Q089", "Q090", "Q092", "Q093", "Q094", "Q095",
    "Q097", "Q098", "Q099", "Q101", "Q102", "Q103", "Q104", "Q105", "Q106", "Q107",
    "Q108", "Q109", "Q110", "Q111", "Q112", "Q113", "Q115", "Q117", "Q118", "Q119",
    "Q120", "Q121", "Q122", "Q123", "Q124", "Q125", "Q126", "Q127", "Q128", "Q129",
    "Q130", "Q131", "Q132", "Q133", "Q134", "Q135", "Q136", "Q137", "Q138", "Q139", "Q140"
}

# Delay between calls to avoid rate limits - adjust as needed for your Azure quota
SLEEP_BETWEEN_CALLS = 1.0


def load_filtered_questions():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
    filtered = [q for q in all_questions if q["id"] in CHANGED_IDS]
    missing = CHANGED_IDS - {q["id"] for q in filtered}
    if missing:
        print(f"WARNING: {len(missing)} expected IDs not found in dataset: {missing}")
    print(f"Loaded {len(filtered)} / {len(CHANGED_IDS)} target questions from dataset.")
    return filtered


def run_config(config_name, run_fn, questions):
    print(f"\n{'='*70}")
    print(f"Running Config {config_name} on {len(questions)} questions...")
    print(f"{'='*70}")

    rows = []
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question_text = q["question"]
        print(f"[{i}/{len(questions)}] {qid}: {question_text[:60]}...")

        try:
            # run_fn(question) returns a dict already containing: config,
            # question, answer, retrieved_chunks, latency_ms, grv_score,
            # grv_label, grv_layer_scores, flagged, ragas_faithfulness,
            # cross_encoder_score, reranker_score (see src/pipeline/config_*.py)
            result = run_fn(question_text)
        except Exception as e:
            print(f"  ERROR on {qid}: {e}")
            result = {"config": config_name, "question": question_text, "answer": f"ERROR: {e}"}

        # Prepend dataset metadata (id + expected answer + source info),
        # then merge in everything the pipeline function returned.
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

    return rows


def write_csv(rows, path):
    if not rows:
        print(f"No rows to write for {path}")
        return
    # Union of all keys across rows, in case Config C rows have extra fields
    # (grv_score, retrieved_chunks, etc.) that Config A/B rows don't.
    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    questions = load_filtered_questions()

    if not questions:
        print("No questions to run. Check CHANGED_IDS against your dataset IDs.")
        return

    # --- Config A ---
    rows_a = run_config("A", run_config_a, questions)
    write_csv(rows_a, os.path.join(OUTPUT_DIR, "results_config_A_rerun.csv"))

    # --- Config B ---
    rows_b = run_config("B", run_config_b, questions)
    write_csv(rows_b, os.path.join(OUTPUT_DIR, "results_config_B_rerun.csv"))

    # --- Config C ---
    rows_c = run_config("C", run_config_c, questions)
    write_csv(rows_c, os.path.join(OUTPUT_DIR, "results_config_C_rerun.csv"))

    print("\nDone. Three rerun CSVs written to the results/ directory:")
    print("  - results_config_A_rerun.csv")
    print("  - results_config_B_rerun.csv")
    print("  - results_config_C_rerun.csv")
    print("\nYour original 200-row results files were NOT touched.")
    print("Next: send these 3 files back for merging into the final result sets.")


if __name__ == "__main__":
    main()
