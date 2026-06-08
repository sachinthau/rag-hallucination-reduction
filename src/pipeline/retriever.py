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
    # Use similarity search instead of hybrid to avoid the k conflict bug
    chunks = vector_store.similarity_search(
        query=question,
        k=settings.TOP_K_CHUNKS
    )
    return chunks