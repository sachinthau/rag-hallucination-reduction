import os
import sys
import argparse
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

RESULTS_C = os.path.join(_PROJECT_ROOT, "results", "results_config_C.csv")
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "data", "annotation_template.csv")

COLUMNS = ["question_id", "question", "expected_answer", "answer", "grv_label", "human_label", "notes"]


def main():
    parser = argparse.ArgumentParser(
        description="Build a blank human-annotation CSV from results/results_config_C.csv."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite data/annotation_template.csv even if it already exists."
    )
    args = parser.parse_args()

    if not os.path.exists(RESULTS_C):
        print(f"ERROR: {RESULTS_C} not found. Run Config C over the QA dataset first.")
        return

    if os.path.exists(OUTPUT_PATH) and not args.force:
        print(f"ERROR: {OUTPUT_PATH} already exists. Pass --force to overwrite it.")
        print("       (the existing file may already contain filled-in human labels)")
        return

    df = pd.read_csv(RESULTS_C)
    template = df.reindex(columns=COLUMNS)
    template["human_label"] = ""
    template["notes"] = ""

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    template.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(template)} rows to {OUTPUT_PATH}")
    print("Fill in 'human_label' per row with one of: grounded, partially_grounded, ungrounded")


if __name__ == "__main__":
    main()
