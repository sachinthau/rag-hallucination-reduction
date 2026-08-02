"""
calculate_metrics.py
====================
Run this script after:
  1. Completing the full evaluation run
  2. Manually annotating 100-150 Config C responses

Usage:
    python calculate_metrics.py

Outputs:
    - Cohen's Kappa between GRV labels and human annotations
    - Precision, Recall, F1 for hallucination detection
    - Hallucination rate per configuration
    - Full summary table printed to terminal
    - Results saved to logs/metrics_summary.json

UPDATED: Config C responses now go through one of two scoring paths:
  - "standard_three_layer": 0-1 weighted hybrid score (cross-encoder,
    RAGAS, reranker)
  - "abstention_verification": raw Azure AI Search semantic reranker score
    (~1.0-4.0+), used only for refusal-type answers
These two scales are NOT comparable, so avg_grv_score is now reported
separately per scoring path instead of as one blended (and previously
meaningless) average across both. Older results files without a
scoring_path column are treated as entirely "standard_three_layer" for
backward compatibility.
"""

import os
import sys
import json
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -- File paths ----------------------------------------------------------------
RESULTS_A   = "../results/results_config_A.csv"
RESULTS_B   = "../results/results_config_B.csv"
RESULTS_C   = "../results/results_config_C.csv"
ANNOTATIONS = "../data/annotation_template.csv"
QA_DATASET  = "../data/questions/qa_dataset.json"
OUTPUT_JSON = "../logs/metrics_summary.json"

# -- Label mappings --------------------------------------------------------------
def to_binary(label: str) -> int:
    if label in ("ungrounded", "partially_grounded"):
        return 1
    return 0

def sep(char="-", width=70):
    print(char * width)

def header(title):
    sep("=")
    print(f"  {title}")
    sep("=")

# -- Hallucination rate ----------------------------------------------------------
def hallucination_rate(results_path: str, config: str) -> dict:
    if not os.path.exists(results_path):
        print(f"  WARNING: {results_path} not found. Run the evaluation first.")
        return {}

    df = pd.read_csv(results_path)
    total = len(df)

    if config == "C" and "grv_label" in df.columns:
        hallucinated = df[df["grv_label"].isin(["ungrounded", "partially_grounded"])]
        flagged = len(hallucinated)
        rate = round(flagged / total, 4) if total > 0 else 0
        grounded = len(df[df["grv_label"] == "grounded"])
        partial  = len(df[df["grv_label"] == "partially_grounded"])
        unground = len(df[df["grv_label"] == "ungrounded"])

        result = {
            "total": total,
            "grounded": grounded,
            "partially_grounded": partial,
            "ungrounded": unground,
            "hallucination_rate": rate,
        }

        # Split avg score by scoring path if the column exists
        if "scoring_path" in df.columns:
            std = df[df["scoring_path"] == "standard_three_layer"]
            abst = df[df["scoring_path"] == "abstention_verification"]
            if len(std) > 0:
                result["avg_grv_score_standard_path"] = round(
                    std["grv_score"].dropna().astype(float).mean(), 4
                )
                result["n_standard_path"] = len(std)
            if len(abst) > 0:
                result["avg_relevance_score_abstention_path"] = round(
                    abst["grv_score"].dropna().astype(float).mean(), 4
                )
                result["n_abstention_path"] = len(abst)
            # abstention flag breakdown, if present
            if "abstention_flag" in df.columns:
                flags = df["abstention_flag"].dropna()
                if len(flags) > 0:
                    result["correct_abstentions"] = int((flags == "correct_abstention").sum())
                    result["suspected_retrieval_misses"] = int((flags == "retrieval_miss_suspected").sum())
        else:
            # Backward compatibility: no scoring_path column means this is
            # an older results file where every row used the standard path.
            result["avg_grv_score"] = round(df["grv_score"].dropna().astype(float).mean(), 4)

        return result
    else:
        return {
            "total": total,
            "note": "No GRV labels for this config. Hallucination rate from human annotation only."
        }

# -- GRV performance against human annotations ------------------------------------
def grv_performance(results_path: str, annotations_path: str) -> dict:
    if not os.path.exists(results_path):
        print(f"  ERROR: {results_path} not found.")
        return {}
    if not os.path.exists(annotations_path):
        print(f"  ERROR: {annotations_path} not found.")
        return {}

    results     = pd.read_csv(results_path)

    # annotation_template.csv is often re-saved through Excel/Numbers, which
    # can write it as Windows-1252/Latin-1 rather than UTF-8, especially if
    # it contains smart quotes, em-dashes, or other special characters typed
    # or pasted during manual annotation. Try UTF-8 first, fall back cleanly.
    try:
        annotations = pd.read_csv(annotations_path, encoding="utf-8")
    except UnicodeDecodeError:
        print("  NOTE: annotation file is not valid UTF-8, retrying with cp1252 encoding...")
        annotations = pd.read_csv(annotations_path, encoding="cp1252")

    if "question_id" not in results.columns or "question_id" not in annotations.columns:
        print("  ERROR: Both files need a question_id column to merge on.")
        return {}

    merged = results.merge(annotations, on="question_id", how="inner")
    n = len(merged)

    if n == 0:
        print("  ERROR: No matching question IDs between results and annotations.")
        return {}

    print(f"  Matched {n} annotated responses for analysis")

    grv_labels   = merged["grv_label"].fillna("ungrounded").tolist()
    human_labels = merged["human_label"].fillna("ungrounded").tolist()

    grv_binary   = [to_binary(l) for l in grv_labels]
    human_binary = [to_binary(l) for l in human_labels]

    kappa     = round(cohen_kappa_score(human_binary, grv_binary), 4)
    precision = round(precision_score(human_binary, grv_binary, zero_division=0), 4)
    recall    = round(recall_score(human_binary, grv_binary, zero_division=0), 4)
    f1        = round(f1_score(human_binary, grv_binary, zero_division=0), 4)
    cm        = confusion_matrix(human_binary, grv_binary).tolist()

    if kappa >= 0.8:
        kappa_label = "Almost perfect agreement"
    elif kappa >= 0.6:
        kappa_label = "Substantial agreement (meets threshold)"
    elif kappa >= 0.4:
        kappa_label = "Moderate agreement"
    elif kappa >= 0.2:
        kappa_label = "Fair agreement"
    else:
        kappa_label = "Slight agreement"

    result = {
        "n_samples":        n,
        "cohens_kappa":     kappa,
        "kappa_label":      kappa_label,
        "precision":        precision,
        "recall":           recall,
        "f1_score":         f1,
        "confusion_matrix": cm,
        "note": "Binary classification: 1=hallucinated (ungrounded/partial), 0=grounded"
    }

    # Optional: break down agreement by scoring path too, if available
    if "scoring_path" in merged.columns:
        for path_name in ("standard_three_layer", "abstention_verification"):
            subset = merged[merged["scoring_path"] == path_name]
            if len(subset) == 0:
                continue
            sub_grv = [to_binary(l) for l in subset["grv_label"].fillna("ungrounded")]
            sub_human = [to_binary(l) for l in subset["human_label"].fillna("ungrounded")]
            try:
                sub_kappa = round(cohen_kappa_score(sub_human, sub_grv), 4)
            except Exception:
                sub_kappa = None
            result[f"kappa_{path_name}"] = sub_kappa
            result[f"n_{path_name}"] = len(subset)

    return result

# -- Per-category hallucination analysis -------------------------------------------
def category_analysis(results_c_path: str, qa_dataset_path: str) -> dict:
    if not os.path.exists(results_c_path):
        return {}
    if not os.path.exists(qa_dataset_path):
        return {}

    results = pd.read_csv(results_c_path)
    with open(qa_dataset_path) as f:
        questions = {q["id"]: q for q in json.load(f)}

    if "question_id" not in results.columns:
        return {}

    category_stats = {}
    for _, row in results.iterrows():
        qid = row.get("question_id", "")
        q = questions.get(qid, {})
        cat = q.get("category", "unknown")
        in_corpus = q.get("in_corpus", True)

        if cat not in category_stats:
            category_stats[cat] = {
                "total": 0,
                "grounded": 0,
                "hallucinated": 0,
                "in_corpus": in_corpus
            }
        category_stats[cat]["total"] += 1
        label = row.get("grv_label", "")
        if label == "grounded":
            category_stats[cat]["grounded"] += 1
        else:
            category_stats[cat]["hallucinated"] += 1

    for cat in category_stats:
        t = category_stats[cat]["total"]
        h = category_stats[cat]["hallucinated"]
        category_stats[cat]["hallucination_rate"] = round(h / t, 4) if t > 0 else 0

    return category_stats

# -- RAGAS scores summary -----------------------------------------------------------
def ragas_summary(results_path: str) -> dict:
    """
    NOTE: grv_score is intentionally EXCLUDED from this generic summary,
    since it can now come from two different, non-comparable scales.
    See hallucination_rate() for the correctly-separated grv_score
    reporting by scoring_path.
    """
    if not os.path.exists(results_path):
        return {}
    df = pd.read_csv(results_path)
    summary = {}
    for col in ["ragas_faithfulness", "ragas_answer_relevance"]:
        if col in df.columns:
            vals = df[col].dropna().astype(float)
            if len(vals) == 0:
                continue
            summary[col] = {
                "mean":   round(vals.mean(), 4),
                "median": round(vals.median(), 4),
                "min":    round(vals.min(), 4),
                "max":    round(vals.max(), 4),
                "std":    round(vals.std(), 4),
                "n":      len(vals),
            }
    return summary

# -- MAIN ----------------------------------------------------------------------------
def main():
    header("RAG Hallucination Reduction - Metrics Calculator")
    print(f"  Dissertation: K.G. Sachintha Udara | MSc Advanced Software Engineering")
    sep()

    all_metrics = {}

    header("1. Hallucination Rate Per Configuration")
    for cfg, path in [("A", RESULTS_A), ("B", RESULTS_B), ("C", RESULTS_C)]:
        print(f"\n  Config {cfg}:")
        stats = hallucination_rate(path, cfg)
        all_metrics[f"config_{cfg}_stats"] = stats
        if stats:
            for k, v in stats.items():
                print(f"    {k:<40} {v}")

    header("2. GRV Validator Performance (Cohen's Kappa)")
    perf = grv_performance(RESULTS_C, ANNOTATIONS)
    all_metrics["grv_performance"] = perf
    if perf:
        sep("-", 50)
        print(f"  {'Metric':<30} {'Value':>15}")
        sep("-", 50)
        print(f"  {'Samples annotated':<30} {perf.get('n_samples', 'N/A'):>15}")
        print(f"  {'Cohen Kappa':<30} {perf.get('cohens_kappa', 'N/A'):>15}")
        print(f"  {'Kappa interpretation':<30} {perf.get('kappa_label', 'N/A'):>15}")
        print(f"  {'Precision':<30} {perf.get('precision', 'N/A'):>15}")
        print(f"  {'Recall':<30} {perf.get('recall', 'N/A'):>15}")
        print(f"  {'F1 Score':<30} {perf.get('f1_score', 'N/A'):>15}")
        sep("-", 50)
        print(f"\n  Confusion Matrix (rows=human, cols=GRV):")
        cm = perf.get("confusion_matrix", [])
        if cm:
            print(f"                    GRV:Grounded  GRV:Hallucinated")
            print(f"  Human:Grounded    {cm[0][0]:<14} {cm[0][1]:<14}")
            print(f"  Human:Hallucinated {cm[1][0]:<13} {cm[1][1]:<14}")

        if "kappa_standard_three_layer" in perf or "kappa_abstention_verification" in perf:
            print(f"\n  Agreement by scoring path:")
            if "kappa_standard_three_layer" in perf:
                print(f"    Standard three-layer (n={perf.get('n_standard_three_layer')}): Kappa = {perf['kappa_standard_three_layer']}")
            if "kappa_abstention_verification" in perf:
                print(f"    Abstention verification (n={perf.get('n_abstention_verification')}): Kappa = {perf['kappa_abstention_verification']}")

        kappa = perf.get("cohens_kappa", 0)
        print()
        if kappa >= 0.6:
            print(f"  RESULT: Cohen Kappa {kappa} >= 0.6 threshold. GRV reliability CONFIRMED.")
        else:
            print(f"  RESULT: Cohen Kappa {kappa} < 0.6 threshold. GRV reliability needs review.")

    header("3. Hallucination Rate by Category (Config C)")
    cats = category_analysis(RESULTS_C, QA_DATASET)
    all_metrics["category_analysis"] = cats
    if cats:
        print(f"\n  {'Category':<40} {'Total':>7} {'Halluci.':>10} {'Rate':>8} {'Corpus'}")
        sep("-", 75)
        for cat, stats in sorted(cats.items()):
            corpus = "In" if stats.get("in_corpus") else "Out"
            print(f"  {cat:<40} {stats['total']:>7} {stats['hallucinated']:>10} {stats['hallucination_rate']:>8.2%} {corpus:>6}")
    else:
        print("\n  WARNING: No category data found.")

    header("4. RAGAS Score Summary (Config C)")
    ragas = ragas_summary(RESULTS_C)
    all_metrics["ragas_summary"] = ragas
    if ragas:
        for metric, stats in ragas.items():
            print(f"\n  {metric}:")
            for k, v in stats.items():
                print(f"    {k:<10} {v}")
    print("\n  NOTE: grv_score is reported separately by scoring path in")
    print("  Section 1 above (standard_three_layer vs abstention_verification),")
    print("  since these use different, non-comparable scales.")

    header("5. Trade-off Summary (Answers Sub-RQ3)")
    print(f"""
  Sub-RQ3: How does adding the post-generation validation step affect
  response latency, and is the trade-off acceptable?
""")

    os.makedirs("../logs", exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"  Full metrics saved to {OUTPUT_JSON}")
    sep("=")

if __name__ == "__main__":
    main()