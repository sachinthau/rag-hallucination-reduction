from src.pipeline.config_b import query as rag_query
from src.validator.grv import validate
from src.utils.logger import log_result


def query(question: str) -> dict:
    result = rag_query(question)
    result["config"] = "C"
    grv_output = validate(
        question=question,
        answer=result["answer"],
        chunks=result["retrieved_chunks"]
    )
    result["grv_score"] = grv_output["score"]
    result["grv_label"] = grv_output["label"]
    result["grv_layer_scores"] = grv_output["layer_scores"]
    result["flagged"] = grv_output["score"] < 0.6
    log_result(result)
    return result
