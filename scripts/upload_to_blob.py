# scripts/upload_to_blob.py
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from azure.storage.blob import BlobServiceClient
from src.config.settings import settings

client = BlobServiceClient.from_connection_string(settings.STORAGE_CONNECTION_STRING)
container = client.get_container_client(settings.BLOB_CONTAINER)

try:
    container.create_container()
    print("Container created")
except Exception:
    print("Container already exists (fine, continuing)")

corpus_folders = [
    "data/corpus/azure-functions",
    "data/corpus/container-apps",
]

total = 0
for folder in corpus_folders:
    full_folder = os.path.join(PROJECT_ROOT, folder)
    topic = folder.split("/")[-1]
    if not os.path.isdir(full_folder):
        print(f"Warning: {full_folder} not found, skipping")
        continue
    for filename in os.listdir(full_folder):
        if filename.endswith(".md"):
            filepath = os.path.join(full_folder, filename)
            blob_name = f"{topic}/{filename}"
            with open(filepath, "rb") as f:
                container.upload_blob(name=blob_name, data=f, overwrite=True)
            total += 1
            print(f"Uploaded: {blob_name}")

print(f"\nDone. {total} files uploaded to container '{settings.BLOB_CONTAINER}'.")