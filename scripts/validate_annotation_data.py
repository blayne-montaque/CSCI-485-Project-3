"""Check candidate provenance and review state without mutating data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from takemeter.workflow import RAW_COLUMNS, VALID_LABELS, load_csv, load_working, validate_candidate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=PROJECT_ROOT / "data" / "raw_candidates.csv")
    parser.add_argument("--working", type=Path, default=PROJECT_ROOT / "data" / "annotation_working.csv")
    args = parser.parse_args()
    raw = load_csv(args.raw, RAW_COLUMNS)
    working = load_working(args.working)
    validate_candidate_rows(raw)
    if raw["comment_id"].tolist() != working["comment_id"].tolist():
        raise ValueError("Working file does not preserve raw-candidate order and IDs.")
    invalid_proposals = working.loc[
        working["proposed_label"].ne("") & ~working["proposed_label"].isin(VALID_LABELS), "comment_id"
    ].tolist()
    invalid_final = working.loc[
        working["final_label"].ne("") & ~working["final_label"].isin(VALID_LABELS), "comment_id"
    ].tolist()
    if invalid_proposals or invalid_final:
        raise ValueError("Invalid proposed or final labels found.")
    report = {
        "raw_candidates": len(raw),
        "unique_threads": int(raw["thread_id"].nunique()),
        "valid_reddit_permalinks": int(raw["source_url"].str.startswith("https://www.reddit.com/r/3Dprinting/").sum()),
        "proposed_labels": int(working["proposed_label"].ne("").sum()),
        "human_reviewed": int(working["human_reviewed"].sum()),
        "accepted_final_labels": int((working["human_reviewed"] & ~working["excluded"] & working["final_label"].ne("")).sum()),
        "excluded": int(working["excluded"].sum()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
