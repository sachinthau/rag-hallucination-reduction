import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from src.pipeline import config_a, config_b, config_c

def compare(question: str):
    print("=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    print("\n[Config A - Baseline LLM]")
    result_a = config_a.query(question)
    print(f"Answer: {result_a['answer']}")
    print(f"Latency: {result_a['latency_ms']}ms")

    print("\n[Config B - RAG Pipeline]")
    result_b = config_b.query(question)
    print(f"Answer: {result_b['answer']}")
    print(f"Chunks retrieved: {len(result_b['retrieved_chunks'])}")
    print(f"Latency: {result_b['latency_ms']}ms")

    print("\n[Config C - RAG + GRV]")
    result_c = config_c.query(question)
    print(f"Answer: {result_c['answer']}")
    print(f"GRV Score: {result_c['grv_score']}")
    print(f"GRV Label: {result_c['grv_label']}")
    print(f"Layer scores: {result_c['grv_layer_scores']}")
    print(f"Latency: {result_c['latency_ms']}ms")

    print("\n" + "=" * 70)
    print("COPY THIS TO GROK FOR EXTERNAL VALIDATION:")
    print("=" * 70)
    print(f"""
Question: {question}

Answer A (Baseline LLM, no retrieval):
{result_a['answer']}

Answer B (RAG pipeline, retrieved from Azure Functions and Container Apps docs):
{result_b['answer']}

Answer C (RAG + GRV validator, grounding score: {result_c['grv_score']}):
{result_c['answer']}

Please evaluate:
1. Which answer is most factually accurate?
2. Which answer is most grounded in real documentation?
3. Does Answer A contain any hallucinations or outdated information?
4. Rate each answer 1-10 for factual accuracy.
""")

    # Save to file for easy copying
    with open("logs/comparison_output.txt", "a", encoding="utf-8") as f:
        f.write(f"\nQUESTION: {question}\n")
        f.write(f"A: {result_a['answer']}\n")
        f.write(f"B: {result_b['answer']}\n")
        f.write(f"C: {result_c['answer']} (GRV: {result_c['grv_score']})\n")
        f.write("-" * 70 + "\n")

if __name__ == "__main__":
    # Test questions - change these to whatever you want to test
    questions = [
        "What is Azure Functions?",
        "How does Azure Container Apps handle scaling?",
        "What triggers are available in Azure Functions?",
    ]

    for q in questions:
        compare(q)
        print("\n")