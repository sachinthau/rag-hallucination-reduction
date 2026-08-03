import json
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
THRESHOLD_PATH = os.path.join(_PROJECT_ROOT, "data", "abstention_threshold.json")

_DEFAULT_TAU = 0.5

def _load_tau() -> float:
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("tau", _DEFAULT_TAU)
    return _DEFAULT_TAU

def verify_abstention(question: str) -> dict:
    from src.pipeline.retriever_expanded import max_relevance_score

    tau = _load_tau()
    max_score = max_relevance_score(question, k=20)

    if max_score >= tau:
        label = "ungrounded"
        flag = "retrieval_miss_suspected"
    else:
        label = "grounded"
        flag = "correct_abstention"

    score = round(max_score, 4)

    return {
        "score": score,
        "label": label,
        "layer_scores": {
            "cross_encoder": None,
            "ragas_faithfulness": None,
            "reranker": None,
            "abstention_max_relevance": score,
            "abstention_tau_used": tau,
        },
        "abstention_flag": flag,
    }
