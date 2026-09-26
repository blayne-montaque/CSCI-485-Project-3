"""Optionally add Groq AI suggestions to unreviewed TakeMeter candidates.

This never writes final labels or human-review state. A model name is required
so every pre-labeling run records the exact model actually used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from takemeter.workflow import VALID_LABELS, WORKING_COLUMNS, load_working

TAXONOMY = """Classify one r/3Dprinting comment using exactly one label.

reasoned_explanation: A comment that meaningfully explains a conclusion, diagnosis,
recommendation, or viewpoint using causal reasoning, observable evidence, a technical principle,
an explicit tradeoff, or relevant experience that explains why the conclusion follows. Simply
naming a problem or attaching a very brief descriptive phrase does not automatically qualify.
There must be meaningful supporting rationale.

actionable_advice: A comment that provides a concrete recommendation, procedure, setting change,
resource, or next step that a reader could act on, but does not provide substantial reasoning
explaining why that action should work.

bare_claim: A comment that states an answer, diagnosis, opinion, preference, or conclusion
without substantial supporting reasoning and without sufficiently concrete actionable guidance.

reaction: A comment whose primary function is social or emotional rather than technical
contribution: humor, agreement, appreciation, encouragement, surprise, dismissal, or a similar
response that does not materially advance the technical discussion.

If a comment contains both advice and explanation, classify it as reasoned_explanation ONLY when
the explanation supplies a meaningful causal mechanism, observable evidence, technical principle,
explicit tradeoff, or relevant experiential rationale that explains WHY the recommendation or
conclusion follows. If it tells the reader what to do but gives little or no meaningful rationale,
classify it as actionable_advice.

If the comment gives a concrete action the reader can perform, classify it as actionable_advice.
If it merely supplies an answer, diagnosis, conclusion, preference, or opinion without a concrete
next step, classify it as bare_claim.

Personal experience does not automatically count as reasoned_explanation. It qualifies only when
it is used to explain WHY a conclusion or recommendation follows. A bare preference or outcome
without a developed rationale is bare_claim.

Return JSON only: {"proposed_label":"one valid label","ambiguous":true|false,
"note":"at most 20 words"}."""


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def parse_reply(content: str) -> dict[str, object]:
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or parsed.get("proposed_label") not in VALID_LABELS:
        raise ValueError("Response did not contain a valid proposed_label.")
    if not isinstance(parsed.get("ambiguous", False), bool):
        raise ValueError("Response ambiguity flag must be a JSON boolean.")
    return {
        "proposed_label": parsed["proposed_label"],
        "ai_ambiguous": bool(parsed.get("ambiguous", False)),
        "ai_note": str(parsed.get("note", "")).strip()[:240],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exact Groq model for this annotation run.")
    parser.add_argument("--working-file", type=Path, default=PROJECT_ROOT / "data" / "annotation_working.csv")
    parser.add_argument("--summary-output", type=Path, default=PROJECT_ROOT / "data" / "prelabel_summary.json")
    parser.add_argument("--limit", type=int, default=0, help="0 means all currently unclassified rows.")
    parser.add_argument("--retries", type=int, default=2, help="Retries after the initial attempt for invalid or failed responses.")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY is not configured; no pre-labels were generated.")
        raise SystemExit(2)
    frame = load_working(args.working_file)
    candidates = frame.index[frame["proposed_label"].eq("")].tolist()
    if args.limit:
        candidates = candidates[: args.limit]
    client = Groq(api_key=api_key)
    outcomes: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    total_api_attempts = 0
    for completed, row in enumerate(candidates, start=1):
        comment_id = frame.at[row, "comment_id"]
        last_error = ""
        for attempt in range(args.retries + 1):
            try:
                total_api_attempts += 1
                completion = client.chat.completions.create(
                    model=args.model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": TAXONOMY},
                        {"role": "user", "content": f"Comment:\n{frame.at[row, 'text']}"},
                    ],
                )
                result = parse_reply(completion.choices[0].message.content or "")
                for key, value in result.items():
                    frame.at[row, key] = value
                outcomes[str(result["proposed_label"])] += 1
                if result["ai_ambiguous"]:
                    outcomes["ambiguous"] += 1
                break
            except Exception as error:
                last_error = str(error)[:500]
                if attempt < args.retries:
                    time.sleep(1 + attempt)
        else:
            failures.append({"comment_id": comment_id, "attempts": str(args.retries + 1), "error": last_error})
        if completed % 25 == 0 or completed == len(candidates):
            print(f"Processed {completed}/{len(candidates)} candidates.", flush=True)
    frame.loc[:, list(WORKING_COLUMNS)].to_csv(args.working_file, index=False)
    summary = {
        "ran_at_utc": now_utc(),
        "tool": "Groq chat completions",
        "model": args.model,
        "configuration": {"temperature": 0, "response_format": "json_object", "retries_after_initial_attempt": args.retries},
        "attempted": len(candidates),
        "api_attempts": total_api_attempts,
        "successfully_pre_labeled": sum(outcomes[label] for label in VALID_LABELS),
        "failed_or_unclassified": len(failures),
        "proposed_label_distribution": {label: outcomes[label] for label in VALID_LABELS},
        "ambiguity_count": outcomes["ambiguous"],
        "failures": failures,
        "human_reviewed_rows_modified": 0,
        "final_labels_modified": 0,
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
