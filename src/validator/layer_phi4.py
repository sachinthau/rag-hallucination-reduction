import json
from openai import AzureOpenAI
from src.config.settings import settings

PHI4_PROMPT = """You are a factual consistency checker.

Given the SOURCE CONTEXT and the ANSWER below, assess whether every claim
in the ANSWER is logically supported by the SOURCE CONTEXT.

Respond with ONLY a JSON object in this exact format:
{{"score": 0.85, "reasoning": "brief explanation"}}

The score must be a number between 0 (completely unsupported) and 1 (fully supported).

SOURCE CONTEXT:
{context}

ANSWER:
{answer}

JSON response:"""


def score(answer: str, chunks: list) -> float:
    """
    Uses Microsoft Phi-4 deployed on Azure to check logical consistency.
    Phi-4 is architecturally separate from GPT-4o, avoiding the circular dependency problem.
    """
    if not chunks:
        return 0.0

    client = AzureOpenAI(
        azure_endpoint=settings.PHI4_ENDPOINT,
        api_key=settings.PHI4_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION
    )

    context = "\n\n".join(chunks[:3])
    prompt = PHI4_PROMPT.format(context=context, answer=answer)

    try:
        response = client.chat.completions.create(
            model=settings.PHI4_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        response_text = response.choices[0].message.content.strip()
        data = json.loads(response_text)
        raw_score = float(data.get("score", 0.5))
        return max(0.0, min(1.0, raw_score))
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Phi-4 parse warning: {e}")
        return 0.5
    except Exception as e:
        print(f"Phi-4 Azure call failed: {e}")
        return 0.5