"""Create and validate TakeMeter's one locked, stratified experiment split.

The assignment mapping is deliberately append-only in practice: this script will
validate an existing mapping but refuses to overwrite or regenerate it.  Later
training and evaluation work must load ``data/split_assignments.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_LABELS = (
    "reasoned_explanation",
    "actionable_advice",
    "bare_claim",
    "reaction",
)
SPLIT_NAMES = ("train", "validation", "test")
SEED = 42
TEST_SIZE = 43
VALIDATION_SIZE = 42


def normalized_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_finalized_rows(working_path: Path, labeled_path: Path) -> list[dict[str, str]]:
    """Load identifier-bearing reviewed rows and prove they match the final CSV."""
    working_rows = read_csv(working_path)
    final_rows = [
        row
        for row in working_rows
        if row["human_reviewed"].strip().lower() == "true"
        and row["excluded"].strip().lower() == "false"
    ]
    if len(final_rows) != 282:
        raise ValueError(f"Expected 282 accepted rows, found {len(final_rows)}.")
    if any(row["final_label"] not in VALID_LABELS for row in final_rows):
        raise ValueError("Accepted working rows contain an invalid final label.")
    if len({row["comment_id"] for row in final_rows}) != len(final_rows):
        raise ValueError("Accepted working rows contain duplicate comment IDs.")
    if len({normalized_text(row["text"]) for row in final_rows}) != len(final_rows):
        raise ValueError("Accepted working rows contain duplicate normalized text.")
    if any(not row["text"].strip() or not row["source_url"].strip() for row in final_rows):
        raise ValueError("Accepted working rows are missing text or provenance.")

    labeled_rows = read_csv(labeled_path)
    expected_projection = [
        {"text": row["text"], "label": row["final_label"]} for row in final_rows
    ]
    if labeled_rows != expected_projection:
        raise ValueError("labeled_posts.csv does not exactly match reviewed working rows.")
    return final_rows


def create_assignments(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return a deterministic 197/42/43 stratified train/validation/test split."""
    labels = [row["final_label"] for row in rows]
    all_indices = list(range(len(rows)))
    first = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    remaining_indices, test_indices = next(first.split(all_indices, labels))
    remaining_labels = [labels[index] for index in remaining_indices]
    second = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_SIZE, random_state=SEED)
    train_relative, validation_relative = next(
        second.split(remaining_indices, remaining_labels)
    )
    split_by_index = {
        **{index: "test" for index in test_indices},
        **{remaining_indices[index]: "train" for index in train_relative},
        **{remaining_indices[index]: "validation" for index in validation_relative},
    }
    if set(split_by_index) != set(all_indices):
        raise AssertionError("Split did not assign every finalized row exactly once.")
    return [
        {"comment_id": row["comment_id"], "split": split_by_index[index]}
        for index, row in enumerate(rows)
    ]


def validate_assignments(
    rows: list[dict[str, str]], assignments: list[dict[str, str]]
) -> dict[str, object]:
    expected_ids = {row["comment_id"] for row in rows}
    assignment_ids = [row.get("comment_id", "") for row in assignments]
    if any(row.get("split") not in SPLIT_NAMES for row in assignments):
        raise ValueError("Assignments contain an invalid split name.")
    if len(assignments) != len(rows) or set(assignment_ids) != expected_ids:
        raise ValueError("Assignments do not cover every finalized comment ID exactly once.")
    if len(set(assignment_ids)) != len(assignment_ids):
        raise ValueError("Assignments contain a duplicate comment ID.")

    row_by_id = {row["comment_id"]: row for row in rows}
    split_counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for assignment in assignments:
        split = assignment["split"]
        label = row_by_id[assignment["comment_id"]]["final_label"]
        split_counts[split] += 1
        class_counts[split][label] += 1
    expected_sizes = {"train": 197, "validation": 42, "test": 43}
    if dict(split_counts) != expected_sizes:
        raise ValueError(f"Unexpected locked split sizes: {dict(split_counts)}")
    return {
        "total_counts": dict(sorted(split_counts.items())),
        "per_class_counts": {
            split: {label: class_counts[split][label] for label in VALID_LABELS}
            for split in SPLIT_NAMES
        },
    }


def current_checkpoint() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def write_new_lock(rows: list[dict[str, str]], assignments_path: Path, summary_path: Path) -> None:
    if assignments_path.exists() or summary_path.exists():
        raise FileExistsError(
            "A split artifact already exists. Refusing to overwrite the locked split."
        )
    assignments = create_assignments(rows)
    summary = validate_assignments(rows, assignments)
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with assignments_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comment_id", "split"])
        writer.writeheader()
        writer.writerows(assignments)
    checksum = hashlib.sha256(assignments_path.read_bytes()).hexdigest()
    payload = {
        "dataset_checkpoint": current_checkpoint(),
        "dataset_file": "data/labeled_posts.csv",
        "assignment_file": "data/split_assignments.csv",
        "seed": SEED,
        "method": "two-stage sklearn StratifiedShuffleSplit; 43 test then 42 validation from the remainder",
        "locked_test_set": True,
        "total_examples": len(rows),
        "split_counts": summary["total_counts"],
        "per_class_counts": summary["per_class_counts"],
        "assignment_sha256": checksum,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-file", type=Path, default=PROJECT_ROOT / "data" / "annotation_working.csv")
    parser.add_argument("--labeled-file", type=Path, default=PROJECT_ROOT / "data" / "labeled_posts.csv")
    parser.add_argument("--assignments", type=Path, default=PROJECT_ROOT / "data" / "split_assignments.csv")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "artifacts" / "split_summary.json")
    args = parser.parse_args()
    rows = load_finalized_rows(args.working_file, args.labeled_file)
    if args.assignments.exists():
        assignments = read_csv(args.assignments)
        summary = validate_assignments(rows, assignments)
        print(json.dumps({"status": "existing locked split validated", **summary}, indent=2))
        return
    if args.summary.exists():
        raise FileExistsError("Split summary exists without assignments; refusing to replace it.")
    write_new_lock(rows, args.assignments, args.summary)
    print(json.dumps({"status": "locked split created", **validate_assignments(rows, read_csv(args.assignments))}, indent=2))


if __name__ == "__main__":
    main()
