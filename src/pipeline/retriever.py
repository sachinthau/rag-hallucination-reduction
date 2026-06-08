from langchain_community.vectorstores.azuresearch import AzureSearch
from src.ingestion.indexer import get_embeddings
from src.config.settings import settings


def retrieve_chunks(question: str) -> list:
    embeddings = get_embeddings()
    vector_store = AzureSearch(
        azure_search_endpoint=settings.SEARCH_ENDPOINT,
        azure_search_key=settings.SEARCH_API_KEY,
        index_name=settings.SEARCH_INDEX_NAME,
        embedding_function=embeddings.embed_query,
        semantic_configuration_name="default"
    )
    retriever = vector_store.as_retriever(
        search_type="hybrid",
        search_kwargs={"k": settings.TOP_K_CHUNKS}
    )
    chunks = retriever.invoke(question)
    return chunks
