import json
import pandas as pd
from src.pipeline import config_a, config_b, config_c


def run_full_evaluation(dataset_path: str = "data/questions/qa_dataset.json", output_dir: str = "logs"):
    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    handlers = {"A": config_a.query, "B": config_b.query, "C": config_c.query}

    for cfg, handler in handlers.items():
        print(f"\nRunning Configuration {cfg} on {len(questions)} questions...")
        rows = []
        for i, item in enumerate(questions):
            print(f"  [{i+1}/{len(questions)}] {item['id']}")
            try:
                result = handler(item["question"])
                result["expected_answer"] = item.get("expected_answer", "")
                result["in_corpus"] = item.get("in_corpus", True)
                result["question_id"] = item["id"]
                result["category"] = item.get("category", "")
                rows.append(result)
            except Exception as e:
                print(f"  ERROR on {item['id']}: {e}")
                rows.append({
                    "config": cfg,
                    "question_id": item["id"],
                    "question": item["question"],
                    "answer": f"ERROR: {e}",
                    "error": True
                })

        df = pd.DataFrame(rows)
        out_path = f"{output_dir}/results_config_{cfg}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    print("\nFull evaluation complete.")


if __name__ == "__main__":
    run_full_evaluation()
