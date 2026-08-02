# scripts/demo_blob_attribution.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.retriever import retrieve_chunks, get_chunk_sources

question = "How do I create an Azure Function trigger?"

chunks = retrieve_chunks(question)
sources = get_chunk_sources(chunks)

print(f"Q: {question}\n")
for i, (chunk, src) in enumerate(zip(chunks, sources), 1):
    preview = chunk.page_content[:100].replace("\n", " ")
    print(f"[{i}] {preview}...")
    print(f"    Source: {src['blob_url']}\n")