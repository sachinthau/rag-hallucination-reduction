"""
scripts/rerun_full_after_retriever_fix.py

Re-runs ALL 200 questions through Config C, after the retriever.py fix
(similarity_search -> semantic_hybrid_search_with_score), and derives the
Config B rows from the SAME calls rather than calling Config B separately.

Why: config_c.py's query() internally calls config_b.query() first (see
src/pipeline/config_c.py: "result = rag_query(question)") before adding GRV
scoring. Calling both run_config_b() and run_config_c() separately for every
question would duplicate the retrieval + LLM generation call for no reason,
doubling cost/time, and could even produce slightly different answers for
"the same" underlying generation if temperature isn't locked to exactly 0.

Instead, this script calls config_c.query() once per question, and builds
the Config B row by stripping the GRV-specific fields from that same result
and relabeling config back to "B". This is equivalent to what a separate
config_b.query() call would have produced, without the duplicate work.

Config A is deliberately NOT re-run: it never calls retrieve_chunks(), so
the retriever fix has zero effect on its outputs.

Run from your project root or from scripts/:
    python scripts/rerun_full_after_retriever_fix.py
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

SLEEP_BETWEEN_CALLS = 1.0

# Fields that only exist because of GRV scoring (Config C specific).
# Everything else in the result dict came from the underlying Config B call.
GRV_ONLY_FIELDS = {
    "grv_score", "grv_label", "grv_layer_scores", "flagged",
    "ragas_faithfulness", "cross_encoder_score", "reranker_score",
    "abstention_flag",
}


def load_all_questions():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def derive_config_b_row(config_c_result: dict) -> dict:
    """
    Builds the Config B equivalent row from a Config C result, by removing
    GRV-only fields and relabeling config back to "B". The underlying
    answer/retrieved_chunks/latency_ms are identical to what a direct
    config_b.query() call would have produced, since config_c.py calls
    config_b.query() internally before adding GRV fields.
    """
    b_row = {k: v for k, v in config_c_result.items() if k not in GRV_ONLY_FIELDS}
    b_row["config"] = "B"
    # Set GRV fields to None explicitly, matching config_b.py's own output shape
    for field in GRV_ONLY_FIELDS:
        if field in config_c_result:
            b_row[field] = None
    return b_row


def run_all(questions):
    print(f"\n{'='*70}")
    print(f"Running Config C on {len(questions)} questions (post retriever-fix)...")
    print(f"Config B rows will be derived from these same calls (see docstring).")
    print(f"{'='*70}")

    rows_c = []
    rows_b = []

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question_text = q["question"]
        print(f"[{i}/{len(questions)}] {qid}: {question_text[:60]}...")

        try:
            result_c = run_config_c(question_text)
        except Exception as e:
            print(f"  ERROR on {qid}: {e}")
            result_c = {"config": "C", "question": question_text, "answer": f"ERROR: {e}"}

        meta = {
            "question_id": qid,
            "expected_answer": q.get("expected_answer"),
            "source_doc": q.get("source_doc"),
            "in_corpus": q.get("in_corpus"),
            "category": q.get("category"),
        }

        row_c = dict(meta)
        row_c.update(result_c)
        rows_c.append(row_c)

        row_b = dict(meta)
        row_b.update(derive_config_b_row(result_c))
        rows_b.append(row_b)

        time.sleep(SLEEP_BETWEEN_CALLS)

    return rows_b, rows_c


def write_csv(rows, path):
    if not rows:
        print(f"No rows to write for {path}")
        return
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
    questions = load_all_questions()
    print(f"Loaded {len(questions)} total questions.")
    print("NOTE: Config A is skipped intentionally (unaffected by retriever fix).")
    print("NOTE: Config B rows are derived from Config C calls (no duplicate LLM calls).")

    rows_b, rows_c = run_all(questions)

    write_csv(rows_b, os.path.join(OUTPUT_DIR, "results_config_B_postfix.csv"))
    write_csv(rows_c, os.path.join(OUTPUT_DIR, "results_config_C_postfix.csv"))

    print("\nDone. Two files written to results/:")
    print("  - results_config_B_postfix.csv")
    print("  - results_config_C_postfix.csv")
    print("\nConfig A results remain unchanged from your original run.")
    print("Next: upload these two files so metrics can be recomputed against")
    print("the fixed retriever, alongside your existing Config A results.")


if __name__ == "__main__":
    main()

