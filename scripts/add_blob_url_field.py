# scripts/add_blob_url_field.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SimpleField, SearchFieldDataType
from azure.core.credentials import AzureKeyCredential
from src.config.settings import settings

client = SearchIndexClient(
    endpoint=settings.SEARCH_ENDPOINT,
    credential=AzureKeyCredential(settings.SEARCH_API_KEY)
)

index = client.get_index(settings.SEARCH_INDEX_NAME)
existing_names = {f.name for f in index.fields}

if "blob_url" not in existing_names:
    index.fields.append(SimpleField(name="blob_url", type=SearchFieldDataType.String, filterable=True, retrievable=True))
    client.create_or_update_index(index)
    print("Added blob_url field.")
else:
    print("blob_url field already exists, nothing changed.")