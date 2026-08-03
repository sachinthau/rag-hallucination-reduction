

import math
from sentence_transformers import CrossEncoder

_model = None
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

def get_model():
    global _model
    if _model is None:
        print(f"Loading Layer 3 reranker model: {MODEL_NAME}")
        _model = CrossEncoder(MODEL_NAME, max_length=512)
    return _model

def sigmoid(x: float) -> float:
    """Normalises raw ms-marco scores to 0-1 range."""
    return 1 / (1 + math.exp(-x))

def score(answer: str, chunks: list) -> float:
    """
    Layer 3: Relevance ranking using ms-marco-MiniLM-L6-v2.
    Measures how relevant the answer is to the retrieved context.

    Note: pair order is (chunk, answer) not (answer, chunk)
    because ms-marco was trained with document first, query second.

    Returns average relevance score across all chunks, normalised to 0-1.
    """
    if not chunks:
        return 0.0

    model = get_model()

    pairs = [(chunk, answer) for chunk in chunks]
    raw_scores = model.predict(pairs)

    normalised = [sigmoid(float(s)) for s in raw_scores]
    return float(sum(normalised) / len(normalised))