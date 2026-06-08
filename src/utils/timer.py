import time
from functools import wraps


def timed(func):
    """Decorator that adds latency_ms to the returned dict."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_ms = int((time.time() - start) * 1000)
        if isinstance(result, dict):
            result["latency_ms"] = elapsed_ms
        return result
    return wrapper
