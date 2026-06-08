from sentence_transformers import CrossEncoder
from src.config.settings import settings

_model = None


def get_model() -> CrossEncoder:
    global _model
    if _model is None:
        print(f"Loading cross-encoder model: {settings.CROSSENCODER_MODEL}")
        _model = CrossEncoder(settings.CROSSENCODER_MODEL)
    return _model


def score(answer: str, chunks: list) -> float:
    """
    Returns a 0-1 score for how well the answer is grounded in the given chunks.
    Uses NLI entailment probability as the grounding signal.
    """
    if not chunks:
        return 0.0
    model = get_model()
    pairs = [(answer, chunk) for chunk in chunks]
    scores = model.predict(pairs)
    # NLI model returns [contradiction, neutral, entailment] per pair
    entailment_scores = [float(s[2]) for s in scores]
    return max(entailment_scores)
