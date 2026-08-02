"""
Cross-references qa_dataset.json against the list of confirmed "hollow" files
(pages that rely on a missing includes/ file for their actual content).

Run from scripts/evaluation/:
    python check_hollow_files.py
"""

import json
import os

def find_project_root(start_path):
    current = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(current, "data", "corpus")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError("Could not locate project root.")
        current = parent

PROJECT_ROOT = find_project_root(os.path.dirname(__file__) or ".")
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "questions", "qa_dataset.json")

# Confirmed hollow files (rely on missing includes/ folder for actual content)
HOLLOW_FILES = {
    "azure-functions/functions-bindings-azure-mysql-trigger.md",
    "azure-functions/functions-bindings-azure-sql-trigger.md",
    "azure-functions/functions-bindings-cosmosdb-v2-trigger.md",
    "azure-functions/functions-bindings-dapr-trigger.md",
    "azure-functions/functions-bindings-documentdb-trigger.md",
    "azure-functions/functions-bindings-event-grid-trigger.md",
    "azure-functions/functions-bindings-event-hubs-trigger.md",
    "azure-functions/functions-bindings-event-iot-trigger.md",
    "azure-functions/functions-bindings-azure-data-explorer-output.md",
    "azure-functions/functions-bindings-azure-mysql-output.md",
    "azure-functions/functions-bindings-azure-sql-output.md",
    "azure-functions/functions-bindings-cache-output.md",
    "azure-functions/functions-bindings-cosmosdb-v2-output.md",
    "azure-functions/functions-bindings-dapr-output.md",
    "azure-functions/functions-bindings-documentdb-output.md",
    "azure-functions/functions-bindings-event-grid-output.md",
    "azure-functions/functions-bindings-event-hubs-output.md",
    "azure-functions/functions-bindings-azure-data-explorer-input.md",
    "azure-functions/functions-bindings-azure-mysql-input.md",
    "azure-functions/functions-bindings-azure-sql-input.md",
    "azure-functions/functions-bindings-cache-input.md",
    "azure-functions/functions-bindings-cosmosdb-v2-input.md",
    "azure-functions/functions-bindings-documentdb-input.md",
    # already-confirmed hollow non-binding-pattern files
    "azure-functions/functions-bindings-event-hubs.md",
    "azure-functions/functions-bindings-event-grid.md",
}

def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    in_corpus = [q for q in questions if q.get("in_corpus")]
    hits = []
    for q in in_corpus:
        doc = q.get("source_doc", "")
        if doc in HOLLOW_FILES:
            hits.append(q)

    print(f"Checked {len(in_corpus)} in-corpus questions against {len(HOLLOW_FILES)} known hollow files.\n")
    if not hits:
        print("No matches found. No additional questions reference these hollow files.")
        return

    print(f"FOUND {len(hits)} questions still referencing hollow files:\n")
    print("=" * 90)
    for q in hits:
        print(f"ID: {q['id']}  |  source_doc: {q['source_doc']}")
        print(f"  Question: {q['question']}")
        print(f"  Expected answer: {q['expected_answer']}")
        print("-" * 90)

if __name__ == "__main__":
    main()