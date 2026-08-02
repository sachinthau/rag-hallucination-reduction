# scripts/check_retriever_metadata.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.vectorstores.azuresearch import AzureSearch
from src.ingestion.indexer import get_embeddings
from src.config.settings import settings

embeddings = get_embeddings()
vector_store = AzureSearch(
    azure_search_endpoint=settings.SEARCH_ENDPOINT,
    azure_search_key=settings.SEARCH_API_KEY,
    index_name=settings.SEARCH_INDEX_NAME,
    embedding_function=embeddings.embed_query,
)

# hybrid_search = vector + keyword, no semantic re-ranker, no billing needed
results = vector_store.hybrid_search(
    query="How do I create an Azure Function trigger?",
    k=3
)

for doc in results:
    print(f"metadata={doc.metadata}")
    print("---")