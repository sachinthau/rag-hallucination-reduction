"""
src/pipeline/retriever.py

FIXED VERSION: previously used vector_store.similarity_search(), which
bypasses Azure AI Search's semantic reranking entirely, despite
semantic_configuration_name="default" being set (it was configured but
unused). This was confirmed as the likely root cause of several
retrieval-miss cases identified during manual investigation (Q019, Q085,
Q122 — see dissertation Chapter 6).

This version uses semantic_hybrid_search_with_score(), which actually
invokes vector search + keyword search + semantic reranking together.
The function's return type is kept as a plain list of Document objects
(same as before) so no changes are needed in config_b.py or config_c.py,
which both expect retrieve_chunks() to return a list of chunks with
.page_content.
"""

from langchain_community.vectorstores.azuresearch import AzureSearch
from src.ingestion.indexer import get_embeddings
from src.config.settings import settings
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta, timezone

def retrieve_chunks(question: str) -> list:
    embeddings = get_embeddings()
    vector_store = AzureSearch(
        azure_search_endpoint=settings.SEARCH_ENDPOINT,
        azure_search_key=settings.SEARCH_API_KEY,
        index_name=settings.SEARCH_INDEX_NAME,
        embedding_function=embeddings.embed_query,
        semantic_configuration_name="default"
    )

    # semantic_hybrid_search_with_score actually invokes the semantic
    # reranker (unlike similarity_search, which only does vector search).
    # Returns list of (Document, relevance_score) tuples.
    results = vector_store.semantic_hybrid_search_with_score(
        query=question,
        k=settings.TOP_K_CHUNKS
    )

    # Strip the scores to preserve the existing return interface
    # (list of Document objects) so config_b.py / config_c.py don't
    # need any changes.
    chunks = [doc for doc, score in results]
    return chunks

def get_chunk_sources(chunks: list) -> list:
    """
    Look up blob_url for each retrieved chunk by its id.
    chunks is the plain list of Document objects returned by retrieve_chunks().
    """
    client = SearchClient(
        endpoint=settings.SEARCH_ENDPOINT,
        index_name=settings.SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(settings.SEARCH_API_KEY)
    )

    sources = []
    for chunk in chunks:
        doc_id = chunk.metadata.get("id")
        try:
            doc = client.get_document(key=doc_id, selected_fields=["id", "blob_url"])
            sources.append({"id": doc_id, "blob_url": add_sas_token(doc.get("blob_url", "unknown"))})
        except Exception:
            sources.append({"id": doc_id, "blob_url": "unknown"})
    return sources

def add_sas_token(blob_url: str) -> str:
    """Append a short-lived SAS token to a blob URL so it's viewable without auth."""
    account_name = settings.STORAGE_CONNECTION_STRING.split("AccountName=")[1].split(";")[0]
    account_key = settings.STORAGE_CONNECTION_STRING.split("AccountKey=")[1].split(";")[0]

    # blob_url looks like: https://account.blob.core.windows.net/container/path/file.md
    path_part = blob_url.split(f"{settings.BLOB_CONTAINER}/")[1]

    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=settings.BLOB_CONTAINER,
        blob_name=path_part,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(hours=6)   # covers your demo window
    )
    return f"{blob_url}?{sas_token}"
