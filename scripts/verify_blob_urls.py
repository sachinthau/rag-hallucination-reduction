
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from src.config.settings import settings

client = SearchClient(
    endpoint=settings.SEARCH_ENDPOINT,
    index_name=settings.SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(settings.SEARCH_API_KEY)
)

patched_ids = [
    "NThlM2VjOTktNThkNC00OWFhLTg2OGUtODQ0NjFiNzNhNDY4",
    "MzEwNjljMDEtOTI0ZC00MDU4LWE3ZmItNjkwZGUzZmY0ZGI2",
    "MjMzMTQwNzItOTIzMy00NjFlLTlmNWQtN2Y2ZTZjMTIzYzIx",
]

for doc_id in patched_ids:
    doc = client.get_document(key=doc_id)
    print(f"id={doc_id[:12]}...  blob_url={doc.get('blob_url')}")