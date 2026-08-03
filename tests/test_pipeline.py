import pytest


def test_config_a_returns_answer():
    from src.pipeline.config_a import query
    result = query("What is Azure Blob Storage?")
    assert "answer" in result
    assert len(result["answer"]) > 10
    assert result["config"] == "A"
    assert result["latency_ms"] > 0


def test_config_b_returns_chunks():
    from src.pipeline.config_b import query
    result = query("What is Azure Blob Storage?")
    assert "answer" in result
    assert "retrieved_chunks" in result
    assert result["config"] == "B"


def test_config_c_returns_grv_score():
    from src.pipeline.config_c import query
    result = query("What is Azure Blob Storage?")
    assert "grv_score" in result
    assert result["grv_score"] is not None
    assert 0.0 <= result["grv_score"] <= 1.0
    assert result["grv_label"] in ["grounded", "partially_grounded", "ungrounded"]
    assert result["config"] == "C"


def test_grv_score_high_for_grounded_answer():
    from src.validator.grv import validate
    result = validate(
        question="How many replicas does Azure AI Search Basic tier support?",
        answer="The Basic pricing tier supports up to 3 replicas.",
        chunks=["The Basic pricing tier supports up to 3 replicas and 1 partition."]
    )
    assert result["score"] >= 0.6, f"Expected grounded score, got {result['score']}"
    assert result["label"] == "grounded"


def test_grv_score_low_for_hallucinated_answer():
    from src.validator.grv import validate
    result = validate(
        question="How many replicas does Azure AI Search Basic tier support?",
        answer="The Basic pricing tier supports up to 50 replicas and unlimited partitions.",
        chunks=["The Basic pricing tier supports up to 3 replicas and 1 partition."]
    )
    assert result["score"] < 0.6, f"Expected ungrounded score, got {result['score']}"
