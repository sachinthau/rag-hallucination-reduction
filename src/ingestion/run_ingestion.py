
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from src.ingestion.chunker import chunk_documents
from src.ingestion.indexer import build_index

CORPUS_FOLDERS = [
    "data/corpus/azure-functions",
    "data/corpus/container-apps"
]

def run():
    all_chunks = []
    for folder in CORPUS_FOLDERS:
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from {folder}")
        if not documents:
            print(f"  Warning: no .md files found in {folder}")
            continue
        chunks = chunk_documents(documents)
        all_chunks.extend(chunks)

    print(f"\nTotal chunks to index: {len(all_chunks)}")
    build_index(all_chunks)
    print("Ingestion complete.")

if __name__ == "__main__":
    run()