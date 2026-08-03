
import concurrent.futures
from src.validator import layer_crossencoder, layer_ragas, layer_reranker
from src.validator.abstention import is_refusal
from src.validator.abstention_verify import verify_abstention
from src.config.settings import settings

def validate(question: str, answer: str, chunks: list) -> dict:
    if not chunks:
        return {
            "score": 0.0,
            "label": "ungrounded",
            "scoring_path": "no_chunks_retrieved",
            "layer_scores": {
                "cross_encoder": 0.0,
                "ragas_faithfulness": 0.0,
                "reranker": 0.0
            }
        }

    if is_refusal(answer):
        result = verify_abstention(question)
        result["scoring_path"] = "abstention_verification"
        return result

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
        "scoring_path": "standard_three_layer",
        "layer_scores": {
            "cross_encoder": round(ce_score, 4),
            "ragas_faithfulness": round(ragas_score, 4),
            "reranker": round(reranker_score, 4)
        }
    }
