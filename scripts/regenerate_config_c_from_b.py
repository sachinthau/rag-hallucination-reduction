"""
scripts/regenerate_config_c_from_b.py

Rebuilds Config C results by running GRV validation directly on top of the
EXISTING, already-generated results_config_B.csv, rather than re-calling
config_c.query() (which would re-invoke config_b.query() internally and
generate a fresh, potentially different answer due to LLM non-determinism).

Why this is better than a fresh Config C re-run:
  - No duplicate/wasted LLM generation calls (config_b's answers/chunks are
    reused as-is)
  - Guarantees Config B and Config C are scored against the EXACT SAME
    underlying answer, with zero risk of drift between two separate
    generation calls
  - Lets us capture precise, honest grv_validation_latency_ms and
    total_latency_ms for ALL 200 questions in one clean pass, not just
    the 54 refusal questions from before

Requires: results/results_config_B.csv already contains the final,
retriever-fixed 200 rows (generation + retrieval only, no GRV columns
needed - those get computed fresh here).

Run from your project root or from scripts/:
    python scripts/regenerate_config_c_from_b.py
"""

import ast
import os
import sys
import csv
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
from src.validator.grv import validate

RESULTS_B_PATH = os.path.join(_PROJECT_ROOT, "results", "results_config_B.csv")
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "results", "results_config_C.csv")


def parse_chunks(raw_value):
    """
    retrieved_chunks is stored in the CSV as a stringified Python list
    (e.g. "['chunk one text...', 'chunk two text...']"). Parse it back
    into a real list of strings for passing to validate().
    """
    if pd.isna(raw_value):
        return []
    if isinstance(raw_value, list):
        return raw_value
    try:
        parsed = ast.literal_eval(raw_value)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return []


def main():
    if not os.path.exists(RESULTS_B_PATH):
        print(f"ERROR: {RESULTS_B_PATH} not found.")
        return

    df_b = pd.read_csv(RESULTS_B_PATH)
    print(f"Loaded {len(df_b)} rows from results_config_B.csv")
    print("Running GRV validation on top of existing answers (no new LLM calls)...\n")

    rows = []
    for i, row in df_b.iterrows():
        qid = row.get("question_id", f"row{i}")
        question = row["question"]
        answer = row["answer"]
        chunks = parse_chunks(row.get("retrieved_chunks"))

        print(f"[{i+1}/{len(df_b)}] {qid}: {str(question)[:60]}...")

        grv_start = time.time()
        try:
            grv_output = validate(question=question, answer=answer, chunks=chunks)
        except Exception as e:
            print(f"  ERROR running validate() on {qid}: {e}")
            grv_output = {
                "score": None, "label": "ungrounded", "scoring_path": "error",
                "layer_scores": {"cross_encoder": None, "ragas_faithfulness": None, "reranker": None}
            }
        grv_latency_ms = int((time.time() - grv_start) * 1000)

        new_row = row.to_dict()
        new_row["config"] = "C"
        new_row["grv_score"] = grv_output["score"]
        new_row["grv_label"] = grv_output["label"]
        new_row["grv_layer_scores"] = grv_output["layer_scores"]
        new_row["flagged"] = (grv_output["score"] is not None) and (grv_output["score"] < 0.6)
        new_row["scoring_path"] = grv_output.get("scoring_path")
        if "abstention_flag" in grv_output:
            new_row["abstention_flag"] = grv_output["abstention_flag"]

        layer_scores = grv_output["layer_scores"]
        new_row["ragas_faithfulness"] = layer_scores.get("ragas_faithfulness")
        new_row["cross_encoder_score"] = layer_scores.get("cross_encoder")
        new_row["reranker_score"] = layer_scores.get("reranker")

        # Precise latency breakdown.
        # latency_ms: generation only (the time the user actually waits for
        # the RAG answer). grv_latency_ms: time spent inside validate() —
        # in a real deployment this runs AFTER the answer is already
        # returned to the user (async grounding-flag update), so it does
        # NOT add to user-perceived response time. total_latency_ms is the
        # full pipeline cost for internal tracking purposes only.
        generation_latency_ms = row.get("latency_ms", 0)
        new_row["latency_ms"] = generation_latency_ms
        new_row["grv_latency_ms"] = grv_latency_ms
        new_row["total_latency_ms"] = (generation_latency_ms or 0) + grv_latency_ms

        rows.append(new_row)

    # Preserve a sensible column order, with new columns appended at the end
    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_PATH}")
    print("This is now your complete, final Config C results file,")
    print("with precise total_latency_ms for every question (including")
    print("the abstention verification path's extra network call time).")


if __name__ == "__main__":
    main()