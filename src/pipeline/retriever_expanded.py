"""
src/pipeline/retriever_expanded.py

FIXED VERSION: the previous implementation used LangChain's
AzureSearch.semantic_hybrid_search_with_score(), which returns a "score"
field that does NOT correspond to Azure AI Search's genuine semantic
reranker score. Diagnostic testing confirmed this directly: for the
question "What port does the Dapr sidecar expose its API on by default?",
the chunk containing the exact correct answer ("port 3500... port 50001")
was returned FIRST (i.e. Azure's own reranking was correct), but was
assigned the LOWEST langchain "score" (0.0159) of the 5 results, while an
unrelated screenshot-caption chunk scored highest (0.0328). The ORDER of
results from LangChain's call is trustworthy (which is why retriever.py's
retrieve_chunks() still works correctly, since it just returns chunks in
the order given). But the numeric score itself cannot be trusted and must
not be used for threshold comparisons.

This version bypasses LangChain's wrapper for this specific purpose and
calls the azure-search-documents SDK directly, extracting the genuine
"@search.reranker_score" field. This is the same field confirmed earlier
(via direct Azure portal / REST queries) to range meaningfully from
roughly 1.3 to 2.6+ for real relevant matches, unlike the ~0.01-0.03
compressed, non-discriminative values LangChain was returning.
"""

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from src.config.settings import settings

CONTENT_FIELD = "content"  # confirmed field name from the index schema


def retrieve_chunks_expanded_semantic(question: str, k: int = 20):
    """
    Runs a genuine semantic search (query_type="semantic") against the same
    Azure AI Search index used by the primary pipeline, using the SDK
    directly rather than LangChain's wrapper, with a wider top-k than
    normal generation uses.

    Returns a list of (content_text, reranker_score) tuples, sorted by
    Azure's genuine @search.reranker_score descending (as returned natively
    by the search API, matching what you'd see querying the same index
    directly via the Azure portal or REST API).
    """
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

    # Sort descending by genuine reranker score (should already be the
    # order returned, but sort explicitly to be safe/explicit).
    output.sort(key=lambda pair: pair[1], reverse=True)
    return output


def max_relevance_score(question: str, k: int = 20) -> float:
    """
    Convenience wrapper: returns just the single highest GENUINE reranker
    score found across the expanded search, for threshold comparison in
    the abstention verification step.
    """
    results = retrieve_chunks_expanded_semantic(question, k=k)
    if not results:
        return 0.0
    return max(score for _, score in results)