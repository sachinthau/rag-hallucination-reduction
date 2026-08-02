"""
scripts/compute_final_stats.py

Computes the two statistics still needed for the dissertation's Chapter 5
tables, which calculate_metrics.py doesn't currently produce:

  1. Latency aggregation (mean/median/std/min/max) across Config A, B, C,
     including the new grv_latency_ms / total_latency_ms breakdown for C
  2. Per-layer GRV score stats (cross_encoder_score, reranker_score,
     ragas_faithfulness) for Config C, computed ONLY over the
     standard_three_layer scoring path rows (abstention-path rows have
     None for these fields and are correctly excluded)

Run from your project root or from scripts/:
    python scripts/compute_final_stats.py
"""

import os
import sys
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

RESULTS_A = os.path.join(_PROJECT_ROOT, "results", "results_config_A.csv")
RESULTS_B = os.path.join(_PROJECT_ROOT, "results", "results_config_B.csv")
RESULTS_C = os.path.join(_PROJECT_ROOT, "results", "results_config_C.csv")


def sep(w=70):
    print("-" * w)


def describe(series, label):
    series = series.dropna().astype(float)
    if len(series) == 0:
        print(f"  {label}: no data")
        return
    print(f"  {label}:")
    print(f"    n      = {len(series)}")
    print(f"    mean   = {series.mean():.2f}")
    print(f"    median = {series.median():.2f}")
    print(f"    std    = {series.std():.4f}")
    print(f"    min    = {series.min():.2f}")
    print(f"    max    = {series.max():.2f}")


def main():
    print("="*70)
    print("  1. LATENCY AGGREGATION")
    print("="*70)

    for cfg_name, path in [("A", RESULTS_A), ("B", RESULTS_B), ("C", RESULTS_C)]:
        if not os.path.exists(path):
            print(f"\n  Config {cfg_name}: FILE NOT FOUND at {path}")
            continue
        df = pd.read_csv(path)
        print(f"\n  Config {cfg_name} (n={len(df)}):")
        if "latency_ms" in df.columns:
            describe(df["latency_ms"], "latency_ms (generation only)")
        if cfg_name == "C":
            if "grv_latency_ms" in df.columns:
                describe(df["grv_latency_ms"], "grv_latency_ms (validation only)")
            if "total_latency_ms" in df.columns:
                describe(df["total_latency_ms"], "total_latency_ms (end-to-end)")
            # Also split grv_latency_ms by scoring path if available
            if "scoring_path" in df.columns and "grv_latency_ms" in df.columns:
                sep(50)
                print("  grv_latency_ms BY SCORING PATH:")
                for path_name in ("standard_three_layer", "abstention_verification"):
                    subset = df[df["scoring_path"] == path_name]
                    if len(subset) > 0:
                        describe(subset["grv_latency_ms"], f"  {path_name} (n={len(subset)})")

    print("\n" + "="*70)
    print("  2. PER-LAYER GRV SCORE STATS (Config C, standard_three_layer path only)")
    print("="*70)

    if os.path.exists(RESULTS_C):
        df_c = pd.read_csv(RESULTS_C)
        if "scoring_path" in df_c.columns:
            std_path = df_c[df_c["scoring_path"] == "standard_three_layer"]
        else:
            std_path = df_c  # backward compatibility if column doesn't exist

        print(f"\n  n = {len(std_path)} (standard three-layer path only)\n")
        for col, label in [
            ("cross_encoder_score", "Layer 1 (cross-encoder NLI entailment)"),
            ("ragas_faithfulness", "Layer 2 (RAGAS faithfulness)"),
            ("reranker_score", "Layer 3 (reranker relevance)"),
        ]:
            if col in std_path.columns:
                describe(std_path[col], label)
                print()
            else:
                print(f"  {label}: column '{col}' not found\n")

        print("  grv_score (weighted hybrid, standard path only):")
        if "grv_score" in std_path.columns:
            describe(std_path["grv_score"], "  grv_score")

    print("\n" + "="*70)
    print("Done. Copy the values above into the dissertation's Chapter 5 tables.")
    print("="*70)


if __name__ == "__main__":
    main()
