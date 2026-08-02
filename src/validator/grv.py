# src/validator/grv.py
import concurrent.futures
from src.validator import layer_crossencoder, layer_ragas, layer_reranker
from src.validator.abstention import is_refusal
from src.validator.abstention_verify import verify_abstention
from src.config.settings import settings

# Layer 1: cross-encoder/nli-deberta-v3-base    (30%) - logical entailment
# Layer 2: RAGAS faithfulness                    (30%) - claim-level precision
# Layer 3: cross-encoder/ms-marco-MiniLM-L6-v2  (40%) - relevance ranking
#
# NOTE: for refusal-type answers ("I could not find relevant information..."),
# the above three-layer formula is unreliable. RAGAS faithfulness and the
# cross-encoder both default to vacuously high scores when there are no
# factual claims to check, while the reranker score alone is ambiguous
# between "correct abstention" (genuinely out-of-corpus) and "retrieval
# miss" (question is answerable but retrieval failed to find the chunk).
# See dissertation Chapter 6 discussion (Q185 vs Q187 comparison) for the
# evidence behind this design decision. Refusals are therefore routed
# through verify_abstention() instead of the standard weighted average.
#
# IMPORTANT: the two scoring paths use DIFFERENT, NON-COMPARABLE SCALES.
#   - "standard_three_layer": hybrid_score is a 0-1 weighted average
#   - "abstention_verification": score is a raw Azure AI Search semantic
#     reranker score (typically ranges ~1.0-4.0+)
# Every result dict includes a "scoring_path" field so downstream analysis
# (calculate_metrics.py) can report these separately rather than averaging
# them together, which would be statistically meaningless.


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

    # -- Abstention branch --------------------------------------------------
    if is_refusal(answer):
        result = verify_abstention(question)
        result["scoring_path"] = "abstention_verification"
        return result

    # -- Standard three-layer branch (unchanged) -----------------------------
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
