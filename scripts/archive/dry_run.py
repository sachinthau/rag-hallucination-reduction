# scripts/dry_run.py
import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

CORPUS_FOLDERS = [
    "../data/corpus/azure-functions",
    "../data/corpus/container-apps",
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def dry_run():
    print("=" * 60)
    print("DRY RUN - No API calls will be made")
    print("=" * 60)

    # Step 1: Check folders exist
    print("\n[1] Checking corpus folders...")
    for folder in CORPUS_FOLDERS:
        path = Path(folder)
        if path.exists():
            count = len(list(path.glob("*.md")))
            print(f"  OK  {folder} ({count} files)")
        else:
            print(f"  MISSING  {folder} - create this folder first")

    # Step 2: Check .env values are filled in
    print("\n[2] Checking .env configuration...")
    from dotenv import load_dotenv
    load_dotenv()

    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_GPT4O_DEPLOYMENT",
        "AZURE_EMBEDDING_DEPLOYMENT",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
        "AZURE_STORAGE_CONNECTION_STRING",
        "PHI4_ENDPOINT",
        "PHI4_DEPLOYMENT",
        "PHI4_API_KEY",
    ]

    all_good = True
    for var in required_vars:
        value = os.getenv(var, "")
        if not value or "your_" in value or value.endswith("..."):
            print(f"  MISSING  {var}")
            all_good = False
        else:
            print(f"  OK  {var} = {value[:40]}...")

    # Step 3: Load and chunk documents (no API calls)
    print("\n[3] Loading and chunking documents (no API calls)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    total_docs = 0
    total_chunks = 0

    for folder in CORPUS_FOLDERS:
        path = Path(folder)
        if not path.exists():
            continue
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        try:
            documents = loader.load()
            chunks = splitter.split_documents(documents)
            total_docs += len(documents)
            total_chunks += len(chunks)
            avg_chunk = sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 0
            print(f"  {folder}")
            print(f"    Documents : {len(documents)}")
            print(f"    Chunks    : {len(chunks)}")
            print(f"    Avg chunk : {avg_chunk} characters")
        except Exception as e:
            print(f"  ERROR loading {folder}: {e}")

    print(f"\n  Total documents : {total_docs}")
    print(f"  Total chunks    : {total_chunks}")
    print(f"  Batches needed  : {(total_chunks // 100) + 1} (at 100 chunks per batch)")

    # Step 4: Estimate cost
    print("\n[4] Cost estimate...")
    tokens_per_chunk = CHUNK_SIZE
    total_tokens = total_chunks * tokens_per_chunk
    embedding_cost = (total_tokens / 1_000_000) * 0.13
    print(f"  Estimated tokens    : {total_tokens:,}")
    print(f"  Embedding cost      : ~${embedding_cost:.2f} (text-embedding-3-large)")
    print(f"  Evaluation cost     : ~$10 to $15 (150 questions x 3 configs)")
    print(f"  Total project cost  : ~${embedding_cost + 12:.2f} to ${embedding_cost + 15:.2f}")

    # Step 5: Sample chunk preview
    print("\n[5] Sample chunk preview (first chunk from first folder)...")
    first_folder = CORPUS_FOLDERS[0]
    if Path(first_folder).exists():
        loader = DirectoryLoader(
            first_folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        docs = loader.load()
        if docs:
            chunks = splitter.split_documents(docs[:1])
            if chunks:
                print(f"\n  Source: {chunks[0].metadata.get('source', 'unknown')}")
                print(f"  Length: {len(chunks[0].page_content)} characters")
                print(f"  Preview:\n")
                print("  " + chunks[0].page_content[:300].replace("\n", "\n  "))

    # Final summary
    print("\n" + "=" * 60)
    if all_good and total_chunks > 0:
        print("DRY RUN PASSED - Ready to run ingestion")
        print("Run: python -m src.ingestion.run_ingestion")
    else:
        print("DRY RUN FAILED - Fix the issues above before ingesting")
    print("=" * 60)

if __name__ == "__main__":
    dry_run()