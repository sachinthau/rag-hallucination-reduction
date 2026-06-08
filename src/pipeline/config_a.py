import time
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config.settings import settings
from src.utils.logger import log_result

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question factually and concisely. "
    "If you do not know the answer, say so clearly."
)


def query(question: str) -> dict:
    llm = AzureChatOpenAI(
        azure_deployment=settings.GPT4O_DEPLOYMENT,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        temperature=settings.LLM_TEMPERATURE
    )
    start = time.time()
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
    response = llm.invoke(messages)
    latency_ms = int((time.time() - start) * 1000)
    result = {
        "config": "A",
        "question": question,
        "answer": response.content,
        "retrieved_chunks": [],
        "latency_ms": latency_ms,
        "grv_score": None,
        "grv_label": None
    }
    log_result(result)
    return result
