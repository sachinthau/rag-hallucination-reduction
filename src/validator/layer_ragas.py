from ragas import evaluate
from ragas.metrics import faithfulness
from datasets import Dataset
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from src.config.settings import settings


def score(question: str, answer: str, chunks: list) -> float:
    """
    Returns RAGAS faithfulness score between 0 and 1.
    Measures what proportion of claims in the answer are supported by the retrieved context.
    """
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
    result = evaluate(dataset, metrics=[faithfulness], llm=llm, embeddings=embeddings)
    return float(result["faithfulness"])
