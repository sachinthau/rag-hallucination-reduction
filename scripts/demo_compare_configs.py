
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import config_a, config_b, config_c
from src.pipeline.retriever import retrieve_chunks, get_chunk_sources

DEMO_QUESTIONS = [
    "How do I create an Azure Function trigger?",
    "What is the default scaling limit for Container Apps?",
    "What is the capital of France?",
]

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

def header(text, color=C.CYAN):
    line = "=" * 78
    print(f"\n{color}{C.BOLD}{line}\n{text}\n{line}{C.RESET}")

def subheader(text, color=C.BLUE):
    print(f"\n{color}{C.BOLD}--- {text} ---{C.RESET}")

def label_color(label):
    if label == "grounded":
        return C.GREEN
    if label == "partially_grounded":
        return C.YELLOW
    if label == "ungrounded":
        return C.RED
    return C.WHITE

def demo_question(question: str, index: int, total: int):
    header(f"QUESTION {index}/{total}: {question}", color=C.MAGENTA)

    result_a = config_a.query(question)
    subheader("CONFIG A — Baseline LLM (no retrieval)", color=C.WHITE)
    print(f"{C.DIM}Answer:{C.RESET} {result_a['answer']}")
    print(f"{C.DIM}Latency:{C.RESET} {result_a['latency_ms']}ms")

    result_b = config_b.query(question)
    subheader("CONFIG B — RAG with hybrid retrieval", color=C.BLUE)
    print(f"{C.DIM}Answer:{C.RESET} {result_b['answer']}")
    print(f"{C.DIM}Latency:{C.RESET} {result_b['latency_ms']}ms")

    chunks = retrieve_chunks(question)
    sources = get_chunk_sources(chunks)
    print(f"{C.DIM}Retrieved sources:{C.RESET}")
    for i, src in enumerate(sources, 1):
        print(f"  {C.CYAN}[{i}]{C.RESET} {src['blob_url']}")

    result_c = config_c.query(question)
    subheader("CONFIG C — RAG + Grounded Response Validator", color=C.GREEN)
    print(f"{C.DIM}Answer:{C.RESET} {result_c['answer']}")

    lbl = result_c['grv_label']
    lbl_color = label_color(lbl)
    flagged = result_c['flagged']
    flag_color = C.RED if flagged else C.GREEN
    flag_text = "YES — hallucination risk" if flagged else "No"

    print(f"{C.DIM}GRV score:{C.RESET} {result_c['grv_score']:.4f}   "
          f"{C.DIM}Label:{C.RESET} {lbl_color}{C.BOLD}{lbl}{C.RESET}")
    print(f"{C.DIM}Flagged:{C.RESET} {flag_color}{C.BOLD}{flag_text}{C.RESET}")
    print(f"{C.DIM}Scoring path:{C.RESET} {result_c.get('scoring_path')}")
    print(f"{C.DIM}Layer scores:{C.RESET} "
          f"ragas={result_c['ragas_faithfulness']}, "
          f"cross_encoder={result_c['cross_encoder_score']}, "
          f"reranker={result_c['reranker_score']}")
    print(f"{C.DIM}Generation latency:{C.RESET} {result_c['latency_ms']}ms   "
          f"{C.DIM}GRV latency:{C.RESET} {result_c['grv_latency_ms']}ms   "
          f"{C.DIM}Total:{C.RESET} {result_c['total_latency_ms']}ms")

if __name__ == "__main__":
    total = len(DEMO_QUESTIONS)
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        demo_question(q, i, total)
    print(f"\n{C.BOLD}{C.CYAN}{'=' * 78}\nDEMO COMPLETE\n{'=' * 78}{C.RESET}\n")