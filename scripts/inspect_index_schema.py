# scripts/inspect_index_schema.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from src.config.settings import settings

# --- Part 1: Schema fields ---
index_client = SearchIndexClient(
    endpoint=settings.SEARCH_ENDPOINT,
    credential=AzureKeyCredential(settings.SEARCH_API_KEY)
)

index = index_client.get_index(settings.SEARCH_INDEX_NAME)

print("=" * 60)
print(f"INDEX: {settings.SEARCH_INDEX_NAME}")
print("=" * 60)
print("\nFIELDS:")
for f in index.fields:
    print(f"  name={f.name!r:30} type={str(f.type):25} key={f.key}")

# --- Part 2: Sample documents to see actual stored values ---
search_client = SearchClient(
    endpoint=settings.SEARCH_ENDPOINT,
    index_name=settings.SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(settings.SEARCH_API_KEY)
)

print("\n" + "=" * 60)
print("SAMPLE DOCUMENTS (first 3)")
print("=" * 60)

results = search_client.search(search_text="*", top=3)
for i, doc in enumerate(results):
    print(f"\n--- Document {i+1} ---")
    for k, v in doc.items():
        val_str = str(v)
        if len(val_str) > 150:
            val_str = val_str[:150] + "...[truncated]"
        print(f"  {k}: {val_str}")