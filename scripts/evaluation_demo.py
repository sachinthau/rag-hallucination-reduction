"""
scripts/evaluation_demo.py

Quick live demo: runs a handful of real questions from qa_dataset.json
through Config A, B, and C, and prints a side-by-side comparison table.
Useful for showing a supervisor or examiner a live run without waiting
for the full 200-question evaluation.

Run from your project root or from scripts/:
    python scripts/evaluation_demo.py

To change which questions are demoed, edit DEMO_QUESTION_IDS below.
"""

import os
import sys
import time
import json

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.pipeline.config_a import query as query_a
from src.pipeline.config_b import query as query_b
from src.pipeline.config_c import query as query_c

QA_DATASET_PATH = os.path.join(_PROJECT_ROOT, "data", "questions", "qa_dataset.json")

# Pick a small, illustrative mix: two in-corpus questions (one that should
# ground cleanly, one with a specific fact worth checking) and one
# out-of-corpus question (to show a correct refusal). Edit freely.
DEMO_QUESTION_IDS = ["Q088", "Q003", "Q186"]


def sep(char="-", width=110):
    print(char * width)


def header(title):
    sep("=")
    print(f"  {title}")
    sep("=")


def load_demo_questions():
    with open(QA_DATASET_PATH, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
    qmap = {q["id"]: q for q in all_questions}
    selected = [qmap[qid] for qid in DEMO_QUESTION_IDS if qid in qmap]
    missing = [qid for qid in DEMO_QUESTION_IDS if qid not in qmap]
    if missing:
        print(f"  WARNING: question IDs not found in dataset: {missing}")
    return selected


def wrap_print(label, text, width=100, indent="    "):
    print(f"{indent}{label}")
    words = str(text).split()
    line = indent + "  "
    for w in words:
        if len(line) + len(w) > width:
            print(line)
            line = indent + "  " + w + " "
        else:
            line += w + " "
    if line.strip():
        print(line)


def run_demo():
    header("RAG Hallucination Reduction - Live Demo")
    print("  Generation model : GPT-4o via Azure AI Foundry")
    print("  Embedding model  : text-embedding-3-large")
    print("  Retrieval        : Azure AI Search (semantic hybrid search)")
    print("  GRV Layer 1      : cross-encoder/nli-deberta-v3-base (NLI entailment, 30%)")
    print("  GRV Layer 2      : RAGAS faithfulness (claim-level precision, 30%)")
    print("  GRV Layer 3      : cross-encoder/ms-marco-MiniLM-L6-v2 (relevance ranking, 40%)")
    print("  Corpus           : Azure Functions + Container Apps docs")
    sep()

    questions = load_demo_questions()
    if not questions:
        print("  No demo questions found. Check DEMO_QUESTION_IDS against your dataset.")
        return

    all_results = []

    for q in questions:
        qid = q["id"]
        question_text = q["question"]
        header(f"{qid}: {question_text}")
        print(f"  In corpus: {q.get('in_corpus')}  |  Category: {q.get('category')}")
        if q.get("expected_answer"):
            print(f"  Expected answer: {q['expected_answer']}")
        sep()

        row = {"id": qid, "question": question_text, "configs": {}}

        for cfg_name, query_fn in [("A", query_a), ("B", query_b), ("C", query_c)]:
            print(f"\n  Running Config {cfg_name}...", end=" ", flush=True)
            try:
                start = time.time()
                result = query_fn(question_text)
                elapsed = int((time.time() - start) * 1000)
                print(f"done ({elapsed}ms)")
                row["configs"][cfg_name] = result
            except Exception as e:
                print(f"ERROR: {e}")
                row["configs"][cfg_name] = {"error": str(e)}

        all_results.append(row)

        print()
        sep()
        print("\n  ANSWERS\n")
        for cfg in ("A", "B", "C"):
            d = row["configs"].get(cfg, {})
            if "error" in d:
                print(f"  Config {cfg}: ERROR - {d['error']}")
                continue
            grv_score = d.get("grv_score")
            grv_label = d.get("grv_label")
            label_str = f"  [GRV: {grv_score:.4f} / {grv_label}]" if grv_score is not None else ""
            wrap_print(f"Config {cfg}{label_str}:", d.get("answer", ""))
            print()

        sep()
        print(f"\n  METRICS - {qid}\n")
        col_w = 20
        print(f"  {'Metric':<28} {'Config A':>{col_w}} {'Config B':>{col_w}} {'Config C':>{col_w}}")
        sep("-", 96)

        def metric_row(label, key, fmt="{}"):
            vals = []
            for cfg in ("A", "B", "C"):
                d = row["configs"].get(cfg, {})
                v = d.get(key)
                if "error" in d:
                    vals.append("ERROR")
                elif v is None:
                    vals.append("N/A")
                else:
                    try:
                        vals.append(fmt.format(v))
                    except Exception:
                        vals.append(str(v))
            print(f"  {label:<28} {vals[0]:>{col_w}} {vals[1]:>{col_w}} {vals[2]:>{col_w}}")

        metric_row("Latency (ms)", "latency_ms", "{:,}")
        metric_row("Chunks retrieved", "retrieved_chunks", "{}")  # will show list length via fallback below
        metric_row("GRV hybrid score", "grv_score", "{:.4f}")
        metric_row("GRV label", "grv_label", "{}")

        c_layers = row["configs"].get("C", {}).get("grv_layer_scores") or {}
        if c_layers:
            print()
            print(f"  {'GRV Layer 1 (NLI entailment)':<28} {'-':>{col_w}} {'-':>{col_w}} {c_layers.get('cross_encoder', 'N/A'):>{col_w}}")
            print(f"  {'GRV Layer 2 (RAGAS faith.)':<28} {'-':>{col_w}} {'-':>{col_w}} {c_layers.get('ragas_faithfulness', 'N/A'):>{col_w}}")
            print(f"  {'GRV Layer 3 (reranker)':<28} {'-':>{col_w}} {'-':>{col_w}} {c_layers.get('reranker', 'N/A'):>{col_w}}")
        print()

    # -- Aggregate summary --------------------------------------------------
    header(f"AGGREGATE SUMMARY - {len(all_results)} Demo Questions")
    col_w = 20
    print(f"\n  {'Metric':<30} {'Config A':>{col_w}} {'Config B':>{col_w}} {'Config C':>{col_w}}")
    sep("-", 100)

    for cfg in ("A", "B", "C"):
        latencies = [row["configs"][cfg].get("latency_ms", 0) for row in all_results
                     if "error" not in row["configs"].get(cfg, {})]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        print(f"  Config {cfg} avg latency: {avg_latency:,.0f} ms")

    grv_scores = [row["configs"]["C"].get("grv_score") for row in all_results
                  if row["configs"].get("C", {}).get("grv_score") is not None]
    if grv_scores:
        print(f"\n  Config C avg GRV score: {sum(grv_scores)/len(grv_scores):.4f}")
        grv_labels = [row["configs"]["C"].get("grv_label") for row in all_results]
        grounded = grv_labels.count("grounded")
        print(f"  Config C grounded: {grounded}/{len(grv_labels)}")

    sep("=")
    print("  Demo complete.")
    sep("=")


if __name__ == "__main__":
    run_demo()