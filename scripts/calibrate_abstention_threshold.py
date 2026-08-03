import os
import sys
import json
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.pipeline.retriever_expanded import max_relevance_score

QA_DATASET_PATH = os.path.join(_PROJECT_ROOT, "data", "questions", "qa_dataset.json")
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "data", "abstention_threshold.json")

PERCENTILE = 10

SLEEP_BETWEEN_CALLS = 0.5

def percentile(sorted_values, pct):
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1

def main():
    with open(QA_DATASET_PATH, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    in_corpus_questions = [q for q in all_questions if q.get("in_corpus")]
    print(f"Calibrating against {len(in_corpus_questions)} confirmed in-corpus questions...\n")

    scores = []
    for i, q in enumerate(in_corpus_questions, 1):
        qid = q["id"]
        question_text = q["question"]
        try:
            score = max_relevance_score(question_text, k=20)
        except Exception as e:
            print(f"  [{i}/{len(in_corpus_questions)}] {qid}: ERROR - {e}")
            continue
        scores.append(score)
        print(f"  [{i}/{len(in_corpus_questions)}] {qid}: max_relevance = {score:.4f}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    if not scores:
        print("ERROR: no scores collected. Cannot calibrate.")
        return

    scores_sorted = sorted(scores)
    tau = percentile(scores_sorted, PERCENTILE)

    print(f"\n{'='*60}")
    print(f"Score distribution across {len(scores)} in-corpus questions:")
    print(f"  min:    {scores_sorted[0]:.4f}")
    print(f"  p10:    {percentile(scores_sorted, 10):.4f}")
    print(f"  p25:    {percentile(scores_sorted, 25):.4f}")
    print(f"  median: {percentile(scores_sorted, 50):.4f}")
    print(f"  p75:    {percentile(scores_sorted, 75):.4f}")
    print(f"  max:    {scores_sorted[-1]:.4f}")
    print(f"\nCalibrated tau (p{PERCENTILE}): {tau:.4f}")
    print(f"{'='*60}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "tau": round(tau, 4),
            "percentile_used": PERCENTILE,
            "n_calibration_questions": len(scores),
            "score_distribution": {
                "min": round(scores_sorted[0], 4),
                "p10": round(percentile(scores_sorted, 10), 4),
                "p25": round(percentile(scores_sorted, 25), 4),
                "median": round(percentile(scores_sorted, 50), 4),
                "p75": round(percentile(scores_sorted, 75), 4),
                "max": round(scores_sorted[-1], 4),
            }
        }, f, indent=2)

    print(f"\nSaved tau to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
