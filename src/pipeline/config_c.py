"""
src/pipeline/config_c.py

UPDATED: previously, latency_ms was fixed inside config_b.py's rag_query()
call and only covered retrieval + generation. GRV validation time (the
three-layer scoring, OR the abstention verification path's extra live
Azure Search API call) was never measured or added back in, meaning the
reported latency understated the true end-to-end cost of Config C,
especially for refusal-type answers routed through verify_abstention(),
which makes its own separate, non-parallelized network call.

This version keeps "latency_ms" meaning exactly what it always meant
(generation-only, for backward compatibility with already-analyzed rows),
and adds two new fields:
  - grv_validation_latency_ms: time spent inside validate() specifically
  - total_latency_ms: latency_ms + grv_validation_latency_ms (the true
    end-to-end time a user would actually experience)
"""

import time
from src.pipeline.config_b import query as rag_query
from src.validator.grv import validate
from src.utils.logger import log_result


def query(question: str) -> dict:
    result = rag_query(question)
    result["config"] = "C"

    grv_start = time.time()
    grv_output = validate(
        question=question,
        answer=result["answer"],
        chunks=result["retrieved_chunks"]
    )
    grv_latency_ms = int((time.time() - grv_start) * 1000)

    result["grv_score"] = grv_output["score"]
    result["grv_label"] = grv_output["label"]
    result["grv_layer_scores"] = grv_output["layer_scores"]
    result["flagged"] = grv_output["score"] < 0.6
    result["scoring_path"] = grv_output.get("scoring_path")
    if "abstention_flag" in grv_output:
        result["abstention_flag"] = grv_output["abstention_flag"]

    # Save individual layer scores as separate columns for analysis
    layer_scores = grv_output["layer_scores"]
    result["ragas_faithfulness"] = layer_scores.get("ragas_faithfulness")
    result["cross_encoder_score"] = layer_scores.get("cross_encoder")
    result["reranker_score"] = layer_scores.get("reranker")

    # -- Latency breakdown (new) ---------------------------------------------
    # latency_ms: unchanged meaning — generation only (retrieval + LLM call).
    # This is the time the user actually waits for the RAG answer itself.
    # grv_latency_ms: time spent specifically inside validate(), including
    # the abstention path's extra live search call when triggered. In a
    # real deployment, GRV runs asynchronously AFTER the answer is already
    # returned to the user — the answer is shown immediately, and the
    # grounding/hallucination flag arrives shortly after as an update, not
    # as something the user waits for. So grv_latency_ms does NOT add to
    # the user-perceived response time.
    # total_latency_ms: latency_ms + grv_latency_ms — useful for internal
    # tracking of full pipeline cost, but NOT the same as user-facing
    # response time, since GRV validation happens post-hoc/async.
    result["grv_latency_ms"] = grv_latency_ms
    result["total_latency_ms"] = result.get("latency_ms", 0) + grv_latency_ms

    log_result(result)
    return result