import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score


def calculate_grv_performance(results_csv: str, annotation_csv: str) -> dict:
    """
    Compares GRV labels against human annotations.
    Returns precision, recall, F1, and Cohen's Kappa.
    """
    results = pd.read_csv(results_csv)
    annotations = pd.read_csv(annotation_csv)
    merged = results.merge(annotations, on="question_id")

    grv_binary = (merged["grv_label"] == "ungrounded").astype(int)
    human_binary = (merged["human_label"] == "ungrounded").astype(int)

    return {
        "precision": round(precision_score(human_binary, grv_binary, zero_division=0), 4),
        "recall": round(recall_score(human_binary, grv_binary, zero_division=0), 4),
        "f1": round(f1_score(human_binary, grv_binary, zero_division=0), 4),
        "cohens_kappa": round(cohen_kappa_score(human_binary, grv_binary), 4),
        "n_samples": len(merged)
    }


def calculate_hallucination_rate(results_csv: str) -> float:
    """Returns proportion of responses flagged as ungrounded or partially grounded."""
    df = pd.read_csv(results_csv)
    if "grv_label" not in df.columns:
        return None
    flagged = df[df["grv_label"].isin(["ungrounded", "partially_grounded"])]
    return round(len(flagged) / len(df), 4)
