"""Export a final assignment CSV only after the student has completed review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from takemeter.workflow import VALID_LABELS, finalization_summary, load_working


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-file", type=Path, default=PROJECT_ROOT / "data" / "annotation_working.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "labeled_posts.csv")
    parser.add_argument("--confirm-review-complete", action="store_true", help="Required to create the final CSV.")
    args = parser.parse_args()
    if not args.confirm_review_complete:
        parser.error("Refusing to finalize until you pass --confirm-review-complete after personal review.")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing final dataset: {args.output}")
    frame = load_working(args.working_file)
    summary = finalization_summary(frame)
    if summary["accepted_examples"] == 0:
        parser.error("No reviewed, accepted rows are eligible for finalization.")
    eligible = frame.loc[
        frame["human_reviewed"] & ~frame["excluded"] & frame["final_label"].isin(VALID_LABELS), ["text", "final_label"]
    ].rename(columns={"final_label": "label"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(args.output, index=False)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {len(eligible)} reviewed examples to {args.output}")


if __name__ == "__main__":
    main()
