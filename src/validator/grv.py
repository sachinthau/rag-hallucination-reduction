# src/validator/grv.py
import concurrent.futures
from src.validator import layer_crossencoder, layer_ragas, layer_reranker
from src.config.settings import settings

# Layer 1: cross-encoder/nli-deberta-v3-base    (30%) - logical entailment
# Layer 2: RAGAS faithfulness                    (30%) - claim-level precision
# Layer 3: cross-encoder/ms-marco-MiniLM-L6-v2  (40%) - relevance ranking


def validate(question: str, answer: str, chunks: list) -> dict:
    if not chunks:
        return {
            "score": 0.0,
            "label": "ungrounded",
            "layer_scores": {
                "cross_encoder": 0.0,
                "ragas_faithfulness": 0.0,
                "reranker": 0.0
            }
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        fut_ce = executor.submit(layer_crossencoder.score, answer, chunks)
        fut_ragas = executor.submit(layer_ragas.score, question, answer, chunks)
        fut_reranker = executor.submit(layer_reranker.score, answer, chunks)

        ce_score = fut_ce.result()
        ragas_score = fut_ragas.result()
        reranker_score = fut_reranker.result()

    hybrid_score = (
        ce_score * settings.GRV_WEIGHT_CROSSENCODER +
        ragas_score * settings.GRV_WEIGHT_RAGAS +
        reranker_score * settings.GRV_WEIGHT_PHI4
    )

    if hybrid_score >= settings.GRV_THRESHOLD:
        label = "grounded"
    elif hybrid_score >= 0.35:
        label = "partially_grounded"
    else:
        label = "ungrounded"

    return {
        "score": round(hybrid_score, 4),
        "label": label,
        "layer_scores": {
            "cross_encoder": round(ce_score, 4),
            "ragas_faithfulness": round(ragas_score, 4),
            "reranker": round(reranker_score, 4)
        }
    }