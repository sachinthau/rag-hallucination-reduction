import time
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.pipeline.retriever import retrieve_chunks
from src.config.settings import settings
from src.utils.logger import log_result

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the context provided below. "
    "If the context does not contain enough information to answer, say: "
    "'I could not find relevant information in the available documents.' "
    "Do not use any knowledge from your training."
)


def build_prompt(question: str, chunks: list) -> str:
    context = "\n\n---\n\n".join([c.page_content for c in chunks])
    return f"Context:\n{context}\n\nQuestion: {question}"


def query(question: str) -> dict:
    llm = AzureChatOpenAI(
        azure_deployment=settings.GPT4O_DEPLOYMENT,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        temperature=settings.LLM_TEMPERATURE
    )
    start = time.time()
    chunks = retrieve_chunks(question)
    full_prompt = build_prompt(question, chunks)
    messages = [SystemMessage(content=RAG_SYSTEM_PROMPT), HumanMessage(content=full_prompt)]
    response = llm.invoke(messages)
    latency_ms = int((time.time() - start) * 1000)
    result = {
        "config": "B",
        "question": question,
        "answer": response.content,
        "retrieved_chunks": [c.page_content for c in chunks],
        "latency_ms": latency_ms,
        "grv_score": None,
        "grv_label": None,
        "grv_layer_scores": None,
        "flagged": None,
        "ragas_faithfulness": None,
        "cross_encoder_score": None,
        "reranker_score": None,
    }
    log_result(result)
    return result