"""Optionally add Groq AI suggestions to unreviewed TakeMeter candidates.

This never writes final labels or human-review state. A model name is required
so every pre-labeling run records the exact model actually used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from takemeter.workflow import VALID_LABELS, WORKING_COLUMNS, load_working

TAXONOMY = """Classify one r/3Dprinting comment using exactly one label.
reasoned_explanation: meaningful causal reasoning, observable evidence, technical principle,
explicit tradeoff, or experience that explains why a conclusion follows.
actionable_advice: a concrete action, procedure, setting, resource, or next step with no
substantial explanation of why it works.
bare_claim: an answer, diagnosis, opinion, preference, or conclusion without substantial
reasoning and without concrete actionable guidance.
reaction: primarily social or emotional humor, agreement, appreciation, encouragement,
surprise, or dismissal that does not materially advance technical discussion.
If advice includes a meaningful causal mechanism, evidence, principle, tradeoff, or experiential
rationale explaining why it follows, use reasoned_explanation; otherwise use actionable_advice.
A concrete next step is actionable_advice, while a diagnosis or answer without one is bare_claim.
Experience alone is not reasoning unless it explains why the conclusion follows.
Return JSON only: {"proposed_label":"one valid label","ambiguous":true|false,
"note":"at most 20 words"}."""


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def parse_reply(content: str) -> dict[str, object]:
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or parsed.get("proposed_label") not in VALID_LABELS:
        raise ValueError("Response did not contain a valid proposed_label.")
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
    for row in candidates:
        comment_id = frame.at[row, "comment_id"]
        try:
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
        except Exception as error:
            failures.append({"comment_id": comment_id, "error": str(error)[:500]})
    frame.loc[:, list(WORKING_COLUMNS)].to_csv(args.working_file, index=False)
    summary = {
        "ran_at_utc": now_utc(),
        "tool": "Groq chat completions",
        "model": args.model,
        "attempted": len(candidates),
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
