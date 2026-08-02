"""
src/validator/abstention.py

Detects whether a generated answer is a refusal/abstention rather than a
substantive, claim-bearing answer. This is the trigger condition for routing
a response through the abstention verification path in grv.py instead of
the standard three-layer weighted validator.

The patterns below are based on the exact RAG_SYSTEM_PROMPT wording in
src/pipeline/config_b.py, which explicitly instructs the model to say:
"I could not find relevant information in the available documents."
Additional loose variants are included in case the model paraphrases.
"""

import re

REFUSAL_PATTERNS = [
    r"i could not find relevant information",
    r"i couldn't find relevant information",
    r"the context does not contain",
    r"the context doesn't contain",
    r"does not contain (enough|sufficient) information",
    r"cannot find (this|that|the) information",
    r"i don't have (enough|sufficient) information",
    r"not (mentioned|specified|provided|found) in the (available )?documents",
    r"no relevant information (is|was) (available|found)",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]


def is_refusal(answer: str) -> bool:
    """
    Returns True if the answer matches a known refusal/abstention pattern,
    i.e. the model declined to answer rather than making a factual claim.
    """
    if not answer or not isinstance(answer, str):
        return False
    return any(pattern.search(answer) for pattern in _COMPILED_PATTERNS)
