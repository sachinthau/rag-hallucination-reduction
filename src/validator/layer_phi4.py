# src/validator/layer_phi4.py
import json
import re
from openai import OpenAI
from src.config.settings import settings

PHI4_PROMPT = """You are a helpful assistant that checks if an answer is supported by a given context.

Read the CONTEXT and the ANSWER below. Then rate how well the ANSWER is supported by the CONTEXT on a scale from 0 to 1, where 0 means not supported at all and 1 means fully supported.

CONTEXT:
{context}

ANSWER:
{answer}

Please respond with a JSON object containing your score and a brief reason. Example format:
{{"score": 0.85, "reasoning": "The answer is mostly supported by the context"}}

Your response:"""


def extract_score(response_text: str) -> float:
    # Try direct JSON parse first
    try:
        data = json.loads(response_text.strip())
        return float(data.get("score", 0.5))
    except json.JSONDecodeError:
        pass
    # Try extracting JSON from within the response
    match = re.search(r'\{.*?"score".*?\}', response_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return float(data.get("score", 0.5))
        except json.JSONDecodeError:
            pass
    # Try extracting just the number after score key
    match = re.search(r'"score"\s*:\s*([0-9.]+)', response_text)
    if match:
        return float(match.group(1))
    print(f"Phi-4 could not extract score from: {response_text[:100]}")
    return 0.5


def score(answer: str, chunks: list) -> float:
    """
    Uses Microsoft Phi-4 deployed on Azure to check logical consistency.
    Phi-4 is architecturally separate from GPT-4.1-mini, avoiding
    the circular dependency problem.
    """
    if not chunks:
        return 0.0

    client = OpenAI(
        base_url=f"{settings.PHI4_ENDPOINT}models",
        api_key=settings.PHI4_API_KEY,
    )

    context = "\n\n".join(chunks[:3])
    prompt = PHI4_PROMPT.format(context=context, answer=answer)

    try:
        response = client.chat.completions.create(
            model=settings.PHI4_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        response_text = response.choices[0].message.content.strip()
        print(f"Phi-4 raw response: {response_text[:150]}")
        return extract_score(response_text)
    except Exception as e:
        print(f"Phi-4 Azure call failed: {e}")
        return 0.5