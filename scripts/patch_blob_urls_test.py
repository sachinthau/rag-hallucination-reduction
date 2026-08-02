# scripts/patch_blob_urls_test.py
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

results = client.search(search_text="*", top=3)

updates = []
for doc in results:
    meta = json.loads(doc["metadata"])
    source_path = meta["source"]  # e.g. data/corpus/azure-functions/flex-consumption-plan.md
    parts = source_path.split("/")
    topic = parts[-2]
    filename = parts[-1]
    url = get_blob_url(topic, filename)
    updates.append({"id": doc["id"], "blob_url": url})
    print(f"id={doc['id'][:12]}...  topic={topic}  filename={filename}  url={url}")

result = client.merge_documents(documents=updates)
for r in result:
    print(f"  merged={r.succeeded}  key={r.key[:12]}...")