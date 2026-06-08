import concurrent.futures
from src.validator import layer_crossencoder, layer_ragas, layer_phi4
from src.config.settings import settings


def validate(question: str, answer: str, chunks: list) -> dict:
    """
    Runs all three validation layers in parallel and returns a combined grounding score.

    Layers:
    - Cross-encoder (30%): token-level semantic similarity
    - RAGAS faithfulness (30%): claim-level factual grounding
    - Phi-4 (40%): logical consistency using an independent model

    Returns a dict with: score, label, layer_scores
    """
    if not chunks:
        return {
            "score": 0.0,
            "label": "ungrounded",
            "layer_scores": {"cross_encoder": 0.0, "ragas_faithfulness": 0.0, "phi4_consistency": 0.0}
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        fut_ce = executor.submit(layer_crossencoder.score, answer, chunks)
        fut_ragas = executor.submit(layer_ragas.score, question, answer, chunks)
        fut_phi4 = executor.submit(layer_phi4.score, answer, chunks)
        ce_score = fut_ce.result()
        ragas_score = fut_ragas.result()
        phi4_score = fut_phi4.result()

    hybrid_score = (
        ce_score * settings.GRV_WEIGHT_CROSSENCODER +
        ragas_score * settings.GRV_WEIGHT_RAGAS +
        phi4_score * settings.GRV_WEIGHT_PHI4
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
            "phi4_consistency": round(phi4_score, 4)
        }
    }
