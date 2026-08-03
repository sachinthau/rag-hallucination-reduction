
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from src.config.settings import settings

client = SearchClient(
    endpoint=settings.SEARCH_ENDPOINT,
    index_name=settings.SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(settings.SEARCH_API_KEY)
)

account_name = settings.STORAGE_CONNECTION_STRING.split("AccountName=")[1].split(";")[0]

def get_blob_url(topic, filename):
    return f"https://{account_name}.blob.core.windows.net/{settings.BLOB_CONTAINER}/{topic}/{filename}"

batch_size = 1000
skip = 0
total_patched = 0
total_failed = 0

while True:
    results = list(client.search(search_text="*", select=["id", "metadata"], top=batch_size, skip=skip))
    if not results:
        break

    updates = []
    for doc in results:
        meta = json.loads(doc["metadata"])
        source_path = meta["source"]
        parts = source_path.split("/")
        topic = parts[-2]
        filename = parts[-1]
        url = get_blob_url(topic, filename)
        updates.append({"id": doc["id"], "blob_url": url})

    result = client.merge_documents(documents=updates)
    succeeded = sum(1 for r in result if r.succeeded)
    failed = len(result) - succeeded
    total_patched += succeeded
    total_failed += failed
    print(f"Batch skip={skip}: {succeeded} succeeded, {failed} failed")

    skip += batch_size

print(f"\nDone. Total patched: {total_patched}, failed: {total_failed}")