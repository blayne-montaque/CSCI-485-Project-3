from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from takemeter.workflow import (
    RAW_COLUMNS,
    apply_review_action,
    finalization_summary,
    initialize_working_file,
    load_working,
)


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.raw_path = base / "raw.csv"
        self.working_path = base / "working.csv"
        raw = pd.DataFrame(
            [
                {
                    "comment_id": "test_a", "text": "Lower your Z offset.",
                    "source_url": "https://www.reddit.com/r/3Dprinting/comments/thread/test_a/",
                    "thread_id": "thread", "thread_url": "https://www.reddit.com/r/3Dprinting/comments/thread/",
                    "thread_title": "Temporary test thread", "created_utc": "0", "score": "0",
                    "retrieved_at_utc": "0", "source_topic": "test",
                },
                {
                    "comment_id": "test_b", "text": "This is the way.",
                    "source_url": "https://www.reddit.com/r/3Dprinting/comments/thread_two/test_b/",
                    "thread_id": "thread_two", "thread_url": "https://www.reddit.com/r/3Dprinting/comments/thread_two/",
                    "thread_title": "Temporary test thread two", "created_utc": "0", "score": "0",
                    "retrieved_at_utc": "0", "source_topic": "test",
                },
            ], columns=RAW_COLUMNS,
        )
        raw.to_csv(self.raw_path, index=False)
        initialize_working_file(self.raw_path, self.working_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_initialization_never_marks_human_review(self) -> None:
        frame = load_working(self.working_path)
        self.assertFalse(frame["human_reviewed"].any())
        self.assertTrue(frame["final_label"].eq("").all())

    def test_explicit_save_persists_reviewed_label(self) -> None:
        apply_review_action(self.working_path, "test_a", action="save", final_label="actionable_advice")
        row = load_working(self.working_path).query("comment_id == 'test_a'").iloc[0]
        self.assertTrue(row["human_reviewed"])
        self.assertFalse(row["excluded"])
        self.assertEqual(row["final_label"], "actionable_advice")

    def test_excluded_and_unreviewed_rows_are_not_finalizable(self) -> None:
        apply_review_action(self.working_path, "test_a", action="exclude")
        self.assertEqual(finalization_summary(load_working(self.working_path))["accepted_examples"], 0)

    def test_invalid_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_review_action(self.working_path, "test_a", action="save", final_label="not_a_label")

    def test_manually_corrupted_invalid_label_cannot_finalize(self) -> None:
        frame = load_working(self.working_path)
        frame.loc[0, ["human_reviewed", "excluded", "final_label"]] = [True, False, "not_a_label"]
        with self.assertRaises(ValueError):
            finalization_summary(frame)


if __name__ == "__main__":
    unittest.main()
