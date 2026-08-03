from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from src.config.settings import settings

CONTENT_FIELD = "content"  # confirmed field name from the index schema

def retrieve_chunks_expanded_semantic(question: str, k: int = 20):
    client = SearchClient(
        endpoint=settings.SEARCH_ENDPOINT,
        index_name=settings.SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(settings.SEARCH_API_KEY),
    )

    results = client.search(
        search_text=question,
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name="default",
        top=k,
        select=[CONTENT_FIELD],
    )

    output = []
    for r in results:
        content = r.get(CONTENT_FIELD, "")
        reranker_score = r.get("@search.reranker_score")
        if reranker_score is None:
            # Some result rows may lack a reranker score if semantic
            # ranking didn't apply to them; treat as zero relevance.
            reranker_score = 0.0
        output.append((content, reranker_score))

    output.sort(key=lambda pair: pair[1], reverse=True)
    return output

def max_relevance_score(question: str, k: int = 20) -> float:
    results = retrieve_chunks_expanded_semantic(question, k=k)
    if not results:
        return 0.0
    return max(score for _, score in results)
