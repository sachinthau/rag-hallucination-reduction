
import time
import json
import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from src.config.settings import settings

PROGRESS_FILE = "logs/ingestion_progress.json"

def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_deployment=settings.EMBEDDING_DEPLOYMENT,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY
    )

def save_progress(batch_number: int, total_batches: int, chunks_done: int):
    os.makedirs("logs", exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "last_completed_batch": batch_number,
            "total_batches": total_batches,
            "chunks_done": chunks_done
        }, f)

def load_progress() -> int:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
            last = data.get("last_completed_batch", 0)
            chunks_done = data.get("chunks_done", 0)
            print(f"  Resuming from batch {last + 1} ({chunks_done} chunks already indexed)")
            return last
    return 0

def clear_progress():
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

def build_index(chunks: list):
    embeddings = get_embeddings()
    vector_store = AzureSearch(
        azure_search_endpoint=settings.SEARCH_ENDPOINT,
        azure_search_key=settings.SEARCH_API_KEY,
        index_name=settings.SEARCH_INDEX_NAME,
        embedding_function=embeddings.embed_query,
        semantic_configuration_name="default"
    )

    batch_size = 100
    total = len(chunks)
    total_batches = (total // batch_size) + 1

    start_batch = load_progress()

    if start_batch > 0:
        print(f"  Skipping first {start_batch * batch_size} already-indexed chunks")

    for i in range(start_batch * batch_size, total, batch_size):
        batch_number = (i // batch_size) + 1
        batch = chunks[i:i + batch_size]

        try:
            vector_store.add_documents(batch)
            save_progress(batch_number, total_batches, i + len(batch))
            print(f"  Batch {batch_number}/{total_batches}  ({i + len(batch)}/{total} chunks indexed)")
            time.sleep(1)
        except Exception as e:
            print(f"  ERROR on batch {batch_number}: {e}")
            print(f"  Progress saved. Re-run the script to resume from this batch.")
            raise

    clear_progress()
    print(f"Indexed {total} chunks into Azure AI Search")
    return vector_store