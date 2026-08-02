"""
src/validator/abstention_verify.py

Implements the abstention verification step referenced in the dissertation
discussion of GRV's reranker-ambiguity limitation (see Q185 vs Q187
comparison): when Config B/C refuses to answer, the standard three-layer
GRV formula cannot reliably distinguish "correct abstention" from "retrieval
miss", because RAGAS faithfulness and cross-encoder entailment both default
to vacuously high scores (no claims to check), while the reranker score
alone is ambiguous between the two cases.

This module resolves that ambiguity by re-checking retrieval coverage with
a genuine Azure AI Search semantic reranker score (via retriever_expanded.py,
which calls the azure-search-documents SDK directly rather than trusting
LangChain's wrapper, which was confirmed to return a broken, non-relevance
score during manual diagnostic testing).

NOTE ON SCALE: the "score" returned here is a RAW semantic reranker score
(typically ~1.0-4.0+), NOT the 0-1 weighted hybrid score used by the
standard three-layer path in grv.py. Do not average these two together;
grv.py tags every result with a "scoring_path" field specifically so
downstream reporting can keep them separate.
"""

import json
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
THRESHOLD_PATH = os.path.join(_PROJECT_ROOT, "data", "abstention_threshold.json")

_DEFAULT_TAU = 0.5  # fallback if calibration hasn't been run yet


def _load_tau() -> float:
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("tau", _DEFAULT_TAU)
    return _DEFAULT_TAU


def verify_abstention(question: str) -> dict:
    """
    Given a question that Config B/C refused to answer, checks whether a
    broader, genuinely semantically-ranked search finds a chunk that looks
    relevant (suggesting the refusal was a retrieval miss) or not
    (suggesting the refusal was correct).

    Returns a dict matching the shape expected by grv.py's validate()
    function. grv.py adds the "scoring_path" field on top of this.
    """
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
