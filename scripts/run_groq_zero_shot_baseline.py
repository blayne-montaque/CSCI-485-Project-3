"""Run the prescribed zero-shot Groq baseline on TakeMeter's locked test set.

The script loads the persistent split map and never regenerates it. It attempts
the assignment-prescribed model first. If that model cannot be used, it records
a sanitized failure artifact and stops without selecting a substitute model or
creating evaluation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lock_dataset_split import VALID_LABELS, load_finalized_rows, read_csv, validate_assignments


PRESCRIBED_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
PROMPT_TEMPLATE = """Classify the following single public r/3Dprinting comment into exactly one label.

Valid labels:
reasoned_explanation
actionable_advice
bare_claim
reaction

Definitions and decision rules:
- reasoned_explanation: a comment that meaningfully explains a conclusion, diagnosis, recommendation, or viewpoint using causal reasoning, observable evidence, a technical principle, an explicit tradeoff, or relevant experience that explains why the conclusion follows. Merely naming a problem or adding a very short descriptive phrase is not enough; the comment must provide meaningful support.
- actionable_advice: a comment that gives a concrete recommendation, procedure, setting change, resource, or next step a reader could perform, but does not provide substantial reasoning for why that action should work.
- bare_claim: a comment that states an answer, diagnosis, opinion, preference, or conclusion without substantial supporting reasoning and without sufficiently concrete actionable guidance.
- reaction: a comment whose primary function is social or emotional rather than a technical contribution, including humor, agreement, appreciation, encouragement, surprise, dismissal, and similar responses that do not materially advance the technical discussion.

If advice has little or no meaningful rationale, use actionable_advice. If a comment gives a concrete action the reader can perform, use actionable_advice; if it only supplies an answer, diagnosis, conclusion, preference, or opinion without a concrete next step, use bare_claim. Personal experience qualifies as reasoned_explanation only when it explains why a conclusion or recommendation follows.

Output only one exact label name, with no punctuation, explanation, JSON, or markdown.

Comment:
{comment_text}
"""


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sanitize(value: object) -> str:
    text = str(value)
    text = re.sub(r"gsk_[A-Za-z0-9_-]+", "[REDACTED]", text)
    return text[:1000]


def parse_label(content: str) -> str | None:
    candidate = content.strip()
    return candidate if candidate in VALID_LABELS else None


def request_label(client: Groq, model: str, text: str) -> tuple[str | None, str, int]:
    """Classify once, then perform one format-only retry if parsing fails."""
    prompt = PROMPT_TEMPLATE.format(comment_text=text)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content or ""
    label = parse_label(content)
    if label:
        return label, content, 1
    retry_prompt = prompt + "\nFormat correction: output only one exact valid label name."
    retry = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": retry_prompt}],
    )
    retry_content = retry.choices[0].message.content or ""
    return parse_label(retry_content), retry_content, 2


def save_failure(path: Path, error: Exception) -> None:
    status = getattr(error, "status_code", None)
    payload = {
        "attempted_at_utc": now_utc(),
        "requested_model": PRESCRIBED_MODEL,
        "error_type": type(error).__name__,
        "http_status": status,
        "sanitized_error_message": sanitize(error),
        "substitute_model_used": False,
        "baseline_metrics_created": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=PRESCRIBED_MODEL, choices=[PRESCRIBED_MODEL])
    parser.add_argument("--working-file", type=Path, default=PROJECT_ROOT / "data" / "annotation_working.csv")
    parser.add_argument("--labeled-file", type=Path, default=PROJECT_ROOT / "data" / "labeled_posts.csv")
    parser.add_argument("--assignments", type=Path, default=PROJECT_ROOT / "data" / "split_assignments.csv")
    parser.add_argument("--prompt-output", type=Path, default=PROJECT_ROOT / "artifacts" / "baseline_prompt_template.txt")
    parser.add_argument("--predictions-output", type=Path, default=PROJECT_ROOT / "artifacts" / "baseline_predictions.csv")
    parser.add_argument("--results-output", type=Path, default=PROJECT_ROOT / "artifacts" / "baseline_results.json")
    parser.add_argument("--failure-output", type=Path, default=PROJECT_ROOT / "artifacts" / "prescribed_baseline_failure.json")
    args = parser.parse_args()

    if not args.assignments.exists():
        parser.error("No locked split mapping exists. Run lock_dataset_split.py first.")
    rows = load_finalized_rows(args.working_file, args.labeled_file)
    assignments = read_csv(args.assignments)
    validate_assignments(rows, assignments)
    split_by_id = {row["comment_id"]: row["split"] for row in assignments}
    test_rows = [row for row in rows if split_by_id[row["comment_id"]] == "test"]
    if len(test_rows) != 43:
        raise ValueError(f"Expected 43 locked test rows, found {len(test_rows)}.")
    for path in (args.predictions_output, args.results_output, args.failure_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite experiment artifact: {path}")
    args.prompt_output.parent.mkdir(parents=True, exist_ok=True)
    args.prompt_output.write_text(PROMPT_TEMPLATE, encoding="utf-8")

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured; no baseline request was made.")
    client = Groq(api_key=api_key)
    predictions: list[dict[str, str]] = []
    try:
        for row in test_rows:
            predicted, raw_response, attempts = request_label(
                client, args.model, row["text"]
            )
            predictions.append(
                {
                    "comment_id": row["comment_id"],
                    "true_label": row["final_label"],
                    "predicted_label": predicted or "",
                    "parse_status": "valid" if predicted else "parse_failure",
                    "api_attempts": str(attempts),
                    "raw_model_response": raw_response,
                }
            )
    except Exception as error:
        save_failure(args.failure_output, error)
        print(json.dumps({"status": "prescribed model unavailable", "failure_artifact": str(args.failure_output)}, indent=2))
        return

    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    valid = [row for row in predictions if row["parse_status"] == "valid"]
    y_true = [row["true_label"] for row in valid]
    y_pred = [row["predicted_label"] for row in valid]
    report = classification_report(y_true, y_pred, labels=list(VALID_LABELS), output_dict=True, zero_division=0)
    results = {
        "evaluated_at_utc": now_utc(),
        "model": args.model,
        "temperature": 0,
        "prompt_template_file": str(args.prompt_output.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "locked_test_assignment_file": str(args.assignments.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "test_examples": len(test_rows),
        "successful_parses": len(valid),
        "parse_failures": len(test_rows) - len(valid),
        "accuracy_on_successful_parses": accuracy_score(y_true, y_pred) if valid else None,
        "macro_f1_on_successful_parses": f1_score(y_true, y_pred, labels=list(VALID_LABELS), average="macro", zero_division=0) if valid else None,
        "per_class": {label: report[label] for label in VALID_LABELS} if valid else {},
        "confusion_matrix_label_order": list(VALID_LABELS),
        "confusion_matrix_on_successful_parses": confusion_matrix(y_true, y_pred, labels=list(VALID_LABELS)).tolist() if valid else [],
        "prediction_file": str(args.predictions_output.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    args.results_output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "baseline completed", **results}, indent=2))


if __name__ == "__main__":
    main()
