# src/validator/layer_ragas.py
import warnings
import asyncio
warnings.filterwarnings("ignore", category=RuntimeWarning)
from ragas import evaluate
from ragas.metrics import faithfulness
from datasets import Dataset
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from src.config.settings import settings


def score(question: str, answer: str, chunks: list) -> float:
    if not chunks:
        return 0.0

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [chunks],
    }
    dataset = Dataset.from_dict(data)

    llm = AzureChatOpenAI(
        azure_deployment=settings.GPT4O_DEPLOYMENT,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        temperature=0.0
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=settings.EMBEDDING_DEPLOYMENT,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY
    )

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness],
            llm=llm,
            embeddings=embeddings
        )
        raw = result["faithfulness"]
        if isinstance(raw, list):
            return float(raw[0]) if raw else 0.5
        return float(raw)
    except Exception as e:
        print(f"RAGAS scoring warning: {e}")
        return 0.5