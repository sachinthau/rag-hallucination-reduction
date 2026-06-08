# src/ingestion/indexer.py
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from src.config.settings import settings
import time

def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_deployment=settings.EMBEDDING_DEPLOYMENT,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY
    )

def build_index(chunks: list):
    embeddings = get_embeddings()
    vector_store = AzureSearch(
        azure_search_endpoint=settings.SEARCH_ENDPOINT,
        azure_search_key=settings.SEARCH_API_KEY,
        index_name=settings.SEARCH_INDEX_NAME,
        embedding_function=embeddings.embed_query,
        semantic_configuration_name="default"
    )

    # Process in small batches to avoid timeouts
    batch_size = 100
    total = len(chunks)
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        vector_store.add_documents(batch)
        print(f"  Indexed batch {i // batch_size + 1} / {(total // batch_size) + 1}  ({i + len(batch)}/{total} chunks)")
        time.sleep(1)  # Small pause to avoid rate limiting

    print(f"Indexed {total} chunks into Azure AI Search")
    return vector_store