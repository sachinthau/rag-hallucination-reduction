"""
scripts/calibrate_abstention_threshold.py

One-time calibration script for the abstention verification threshold (tau).

Rationale:
When Config B/C refuses to answer ("I could not find relevant information..."),
we need to decide whether that refusal was CORRECT (question is genuinely
out-of-corpus, or the correct chunk truly isn't retrievable) or INCORRECT
(a retrieval miss on a question that IS answerable from the corpus, as
confirmed for Q019, Q085, Q122, Q136 during manual investigation).

We can't just pick an arbitrary relevance-score cutoff out of thin air, so
instead we calibrate tau empirically: we take every question we KNOW is
in-corpus (in_corpus=true in qa_dataset.json) and run the expanded semantic
search (retrieve_chunks_expanded_semantic, k=20) against each one, recording
the single highest relevance score found for each. This gives us a real
distribution of "what relevance score looks like when the answer genuinely
is in the corpus."

tau is then set to a low percentile (default: 10th) of that distribution:
"the weakest relevance score we've ever seen for a question we know has a
real answer in the corpus." If a refusal's max relevance score during
verification meets or exceeds tau, we treat it as a likely retrieval miss
(ungrounded) rather than a correct abstention (grounded).

Usage:
    python scripts/calibrate_abstention_threshold.py

Outputs:
    - Prints the full score distribution and the calibrated tau value
    - Saves tau to data/abstention_threshold.json for use by
      src/validator/abstention_verify.py
"""

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

PERCENTILE = 10  # use the 10th percentile as tau (conservative: only the
                 # weakest-scoring in-corpus questions set the bar)

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
