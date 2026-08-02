# scripts/test_semantic_after_billing.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.retriever import retrieve_chunks

chunks = retrieve_chunks("How do I create an Azure Function trigger?")
print(f"Got {len(chunks)} chunks back")
for c in chunks:
    print(f"  metadata={c.metadata}")