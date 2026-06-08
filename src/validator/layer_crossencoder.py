# src/validator/layer_crossencoder.py
import numpy as np
from sentence_transformers import CrossEncoder
from src.config.settings import settings

_model = None


def get_model() -> CrossEncoder:
    global _model
    if _model is None:
        print(f"Loading cross-encoder model: {settings.CROSSENCODER_MODEL}")
        _model = CrossEncoder(settings.CROSSENCODER_MODEL)
    return _model


def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


def score(answer: str, chunks: list) -> float:
    """
    Returns a 0-1 score for how well the answer is grounded in the chunks.
    Uses NLI entailment probability after softmax normalisation.
    """
    if not chunks:
        return 0.0
    model = get_model()
    pairs = [(answer, chunk) for chunk in chunks]
    raw_scores = model.predict(pairs)

    # Apply softmax to convert raw logits to probabilities
    # NLI labels: [contradiction, neutral, entailment]
    entailment_probs = [softmax(s)[2] for s in raw_scores]
    return float(max(entailment_probs))