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

    result["grv_latency_ms"] = grv_latency_ms
    result["total_latency_ms"] = result.get("latency_ms", 0) + grv_latency_ms

    log_result(result)
    return result
