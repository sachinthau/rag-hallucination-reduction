# src/validator/layer_gpt_consistency.py
import json
import re
from openai import AzureOpenAI
from src.config.settings import settings

GPT_CONSISTENCY_PROMPT = """You are a factual consistency checker. Your only job is to check if an answer is supported by the given context.

CONTEXT:
{context}

ANSWER:
{answer}

Rate how well the ANSWER is supported by the CONTEXT on a scale from 0 to 1.
- 1.0 means every claim in the answer is directly supported by the context
- 0.5 means some claims are supported but others are not
- 0.0 means the answer contradicts or ignores the context completely

Respond with only a JSON object like this:
{{"score": 0.85, "reasoning": "brief explanation"}}"""


def extract_score(response_text: str) -> float:
    try:
        data = json.loads(response_text.strip())
        return float(data.get("score", 0.5))
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*?"score".*?\}', response_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return float(data.get("score", 0.5))
        except json.JSONDecodeError:
            pass
    match = re.search(r'"score"\s*:\s*([0-9.]+)', response_text)
    if match:
        return float(match.group(1))
    return 0.5


def score(answer: str, chunks: list) -> float:
    """
    Uses gpt-4.1-mini as a temporary Layer 3 replacement for Phi-4.
    Note: This is for testing only. Phi-4 is the intended Layer 3 model
    for architectural independence. Documented in TODO.md.
    """
    if not chunks:
        return 0.0

    client = AzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION
    )

    context = "\n\n".join(chunks[:3])
    prompt = GPT_CONSISTENCY_PROMPT.format(context=context, answer=answer)

    try:
        response = client.chat.completions.create(
            model=settings.GPT4O_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": "You are a factual consistency checker. Always respond with only a JSON object."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=150
        )
        response_text = response.choices[0].message.content.strip()
        return extract_score(response_text)
    except Exception as e:
        print(f"GPT consistency check failed: {e}")
        return 0.5