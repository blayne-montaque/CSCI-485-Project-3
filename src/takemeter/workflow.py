"""CSV validation and explicit human-review state transitions."""

from __future__ import annotations

import os
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

VALID_LABELS = (
    "reasoned_explanation",
    "actionable_advice",
    "bare_claim",
    "reaction",
)
TARGET_PER_LABEL = 60

RAW_COLUMNS = (
    "comment_id", "text", "source_url", "thread_id", "thread_url", "thread_title",
    "created_utc", "score", "retrieved_at_utc", "source_topic",
)
WORKING_COLUMNS = RAW_COLUMNS + (
    "proposed_label", "ai_ambiguous", "ai_note", "final_label", "human_reviewed",
    "ambiguous", "excluded", "review_notes",
)
BOOL_COLUMNS = ("ai_ambiguous", "human_reviewed", "ambiguous", "excluded")


def normalize_text(value: object) -> str:
    """Return a stable, case-insensitive representation for duplicate checks."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def _as_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _atomic_csv_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", suffix=".csv", delete=False, dir=path.parent
    ) as temporary:
        temp_path = Path(temporary.name)
        frame.to_csv(temporary, index=False)
    os.replace(temp_path, path)


def load_csv(path: str | Path, columns: tuple[str, ...]) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    return frame.loc[:, list(columns)].copy()


def load_working(path: str | Path) -> pd.DataFrame:
    frame = load_csv(path, WORKING_COLUMNS)
    for column in BOOL_COLUMNS:
        frame[column] = frame[column].map(_as_bool)
    return frame


def initialize_working_file(raw_path: str | Path, working_path: str | Path) -> pd.DataFrame:
    raw = load_csv(raw_path, RAW_COLUMNS)
    working = raw.copy()
    working["proposed_label"] = ""
    working["ai_ambiguous"] = False
    working["ai_note"] = ""
    working["final_label"] = ""
    working["human_reviewed"] = False
    working["ambiguous"] = False
    working["excluded"] = False
    working["review_notes"] = ""
    _atomic_csv_write(working.loc[:, list(WORKING_COLUMNS)], Path(working_path))
    return working


def validate_candidate_rows(frame: pd.DataFrame) -> None:
    if frame["comment_id"].eq("").any():
        raise ValueError("Candidate rows must have a Reddit comment ID.")
    if frame["text"].map(normalize_text).eq("").any():
        raise ValueError("Candidate rows must have non-empty text.")
    if frame["source_url"].str.startswith("https://www.reddit.com/r/3Dprinting/").eq(False).any():
        raise ValueError("Each source URL must be an r/3Dprinting Reddit permalink.")
    if frame["comment_id"].duplicated(keep=False).any():
        raise ValueError("Duplicate comment IDs remain in the candidate pool.")
    if frame["text"].map(normalize_text).duplicated(keep=False).any():
        raise ValueError("Normalized-text duplicates remain in the candidate pool.")


def apply_review_action(
    working_path: str | Path,
    comment_id: str,
    *,
    action: str,
    final_label: str = "",
    ambiguous: bool = False,
    review_notes: str = "",
) -> None:
    """Persist one deliberate review action; displaying a row never calls this function."""
    if action not in {"save", "exclude"}:
        raise ValueError("Review action must be 'save' or 'exclude'.")
    if action == "save" and final_label not in VALID_LABELS:
        raise ValueError("A saved accepted review requires one valid final label.")
    path = Path(working_path)
    frame = load_working(path)
    matches = frame.index[frame["comment_id"] == comment_id].tolist()
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for comment ID {comment_id!r}.")
    row = matches[0]
    frame.loc[row, "human_reviewed"] = True
    frame.loc[row, "ambiguous"] = bool(ambiguous)
    frame.loc[row, "review_notes"] = review_notes.strip()
    if action == "exclude":
        frame.loc[row, "excluded"] = True
        frame.loc[row, "final_label"] = ""
    else:
        frame.loc[row, "excluded"] = False
        frame.loc[row, "final_label"] = final_label
    _atomic_csv_write(frame.loc[:, list(WORKING_COLUMNS)], path)


def accepted_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        frame["human_reviewed"] & ~frame["excluded"] & frame["final_label"].isin(VALID_LABELS)
    ].copy()


def finalization_summary(frame: pd.DataFrame) -> dict[str, object]:
    invalid_final = frame.loc[
        frame["human_reviewed"] & ~frame["excluded"] & ~frame["final_label"].isin(VALID_LABELS),
        "comment_id",
    ].tolist()
    if invalid_final:
        raise ValueError("Reviewed accepted rows contain blank or invalid final labels.")
    accepted = accepted_rows(frame)
    validate_candidate_rows(accepted.loc[:, list(RAW_COLUMNS)])
    counts = Counter(accepted["final_label"])
    return {
        "accepted_examples": len(accepted),
        "class_distribution": {label: counts.get(label, 0) for label in VALID_LABELS},
        "below_target": [label for label in VALID_LABELS if counts.get(label, 0) < TARGET_PER_LABEL],
        "dominating_classes": [
            label for label in VALID_LABELS
            if len(accepted) and counts.get(label, 0) / len(accepted) > 0.50
        ],
    }
