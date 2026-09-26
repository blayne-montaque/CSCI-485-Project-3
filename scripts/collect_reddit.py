"""Collect public r/3Dprinting comments from Arctic Shift with provenance.

This creates a candidate pool, not a human-labeled assignment dataset. No author
identity is saved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from takemeter.workflow import RAW_COLUMNS, initialize_working_file, normalize_text, validate_candidate_rows

ARCHIVE_BASE = "https://arctic-shift.photon-reddit.com/api"
USER_AGENT = "TakeMeterAcademicProject/0.1 (public research; contact: repository maintainer)"
TOPIC_QUERIES = {
    "troubleshooting": "troubleshoot",
    "print_quality": "layer",
    "first_layer": '"first layer"',
    "materials_filament": "filament",
    "tpu": "TPU",
    "strength_orientation": "orientation",
    "slicer_settings": "slicer",
    "printer_setup": "calibration",
    "cad_design": "CAD",
}


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def request_json(session: requests.Session, path: str, params: dict[str, object]) -> dict:
    response = session.get(f"{ARCHIVE_BASE}{path}", params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError(f"Unexpected archive response for {path}: {payload!r}")
    return payload


def is_clear_question_only(text: str) -> bool:
    words = normalize_text(text).split()
    first_word = words[0] if words else ""
    question_starters = {
        "how", "what", "when", "where", "why", "which", "who", "is", "are", "do", "does",
        "did", "can", "could", "would", "should", "will", "anyone", "has", "have",
    }
    return len(words) <= 32 and ("?" in text or first_word in question_starters) and first_word in question_starters


def collect(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    retrieved_at = now_utc()
    records_by_topic: dict[str, list[dict[str, object]]] = {topic: [] for topic in TOPIC_QUERIES}
    filter_counts: Counter[str] = Counter()
    api_errors: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    per_thread: Counter[str] = Counter()
    thread_ids: set[str] = set()
    per_topic_retrieved: Counter[str] = Counter()

    for topic, query in TOPIC_QUERIES.items():
        try:
            payload = request_json(
                session,
                "/comments/search",
                {"subreddit": "3Dprinting", "body": query, "limit": args.per_topic_limit, "sort": "desc"},
            )
        except (requests.RequestException, ValueError) as error:
            api_errors.append({"topic": topic, "error": str(error)})
            continue
        comments = payload["data"]
        per_topic_retrieved[topic] += len(comments)
        for comment in comments:
            comment_id = str(comment.get("id") or "").strip()
            body = str(comment.get("body") or "").strip()
            normalized = normalize_text(body)
            thread_id = str(comment.get("link_id") or "").removeprefix("t3_")
            author = str(comment.get("author") or "").casefold()
            if not comment_id or not body or body.casefold() in {"[deleted]", "[removed]"}:
                filter_counts["empty_deleted_or_removed"] += 1
            elif author == "automoderator" or comment.get("distinguished") in {"moderator", "admin"} or comment.get("stickied"):
                filter_counts["bot_or_moderator_boilerplate"] += 1
            elif is_clear_question_only(body):
                filter_counts["clear_question_only"] += 1
            elif comment_id in seen_ids:
                filter_counts["duplicate_comment_id"] += 1
            elif normalized in seen_texts:
                filter_counts["duplicate_normalized_text"] += 1
            elif not thread_id:
                filter_counts["missing_thread_id"] += 1
            elif per_thread[thread_id] >= args.per_thread_cap:
                filter_counts["per_thread_cap"] += 1
            else:
                seen_ids.add(comment_id)
                seen_texts.add(normalized)
                per_thread[thread_id] += 1
                thread_ids.add(thread_id)
                records_by_topic[topic].append(
                    {
                        "comment_id": comment_id,
                        "text": body,
                        "source_url": f"https://www.reddit.com{comment.get('permalink', '')}",
                        "thread_id": thread_id,
                        "thread_url": "",
                        "thread_title": "",
                        "created_utc": comment.get("created_utc", ""),
                        "score": comment.get("score", ""),
                        "retrieved_at_utc": retrieved_at,
                        "source_topic": topic,
                    }
                )
        time.sleep(args.request_delay)

    records = [record for topic in TOPIC_QUERIES for record in records_by_topic[topic]]
    thread_details: dict[str, dict[str, str]] = {}
    for index in range(0, len(thread_ids), 500):
        batch = sorted(thread_ids)[index : index + 500]
        try:
            payload = request_json(session, "/posts/ids", {"ids": ",".join(batch)})
        except (requests.RequestException, ValueError) as error:
            api_errors.append({"topic": "post_lookup", "error": str(error)})
            continue
        for post in payload["data"]:
            post_id = str(post.get("id") or "")
            if post_id:
                permalink = str(post.get("permalink") or f"/r/3Dprinting/comments/{post_id}/")
                thread_details[post_id] = {
                    "thread_url": f"https://www.reddit.com{permalink}",
                    "thread_title": str(post.get("title") or ""),
                }
        time.sleep(args.request_delay)

    for record in records:
        details = thread_details.get(str(record["thread_id"]), {})
        record["thread_url"] = details.get(
            "thread_url", f"https://www.reddit.com/r/3Dprinting/comments/{record['thread_id']}/"
        )
        record["thread_title"] = details.get("thread_title", "")

    # Round-robin selection keeps a bounded candidate pool from being dominated by
    # the first high-volume query while preserving a reproducible retrieval order.
    selected: list[dict[str, object]] = []
    cursor = {topic: 0 for topic in TOPIC_QUERIES}
    while len(selected) < args.max_candidates:
        added = False
        for topic in TOPIC_QUERIES:
            if cursor[topic] < len(records_by_topic[topic]) and len(selected) < args.max_candidates:
                selected.append(records_by_topic[topic][cursor[topic]])
                cursor[topic] += 1
                added = True
        if not added:
            break
    frame = pd.DataFrame(selected, columns=RAW_COLUMNS)
    if frame.empty:
        raise RuntimeError("No usable public candidates were returned; see the collection errors.")
    validate_candidate_rows(frame)
    titled_threads = frame.loc[frame["thread_title"].ne(""), "thread_id"].nunique()
    summary: dict[str, object] = {
        "retrieval_timestamp_utc": retrieved_at,
        "method": "Arctic Shift public Reddit archive API (/api/comments/search and /api/posts/ids)",
        "community": "r/3Dprinting",
        "author_data_saved": False,
        "retrieved_before_filtering": int(sum(per_topic_retrieved.values())),
        "excluded_by_deterministic_filters": int(sum(filter_counts.values())),
        "filter_breakdown": dict(sorted(filter_counts.items())),
        "duplicates_removed": int(filter_counts["duplicate_comment_id"] + filter_counts["duplicate_normalized_text"]),
        "retained_candidates": int(len(frame)),
        "eligible_before_target_cap": int(len(records)),
        "not_selected_due_to_target_cap": int(len(records) - len(frame)),
        "unique_source_threads": int(frame["thread_id"].nunique()),
        "topic_query_retrieval_counts": dict(sorted(per_topic_retrieved.items())),
        "retained_by_topic_query": dict(sorted(frame["source_topic"].value_counts().to_dict().items())),
        "threads_with_title": int(titled_threads),
        "api_errors": api_errors,
        "limits": {
            "per_topic_limit": args.per_topic_limit,
            "per_thread_cap": args.per_thread_cap,
            "max_candidates": args.max_candidates,
            "request_delay_seconds": args.request_delay,
        },
    }
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output", type=Path, default=PROJECT_ROOT / "data" / "raw_candidates.csv")
    parser.add_argument("--working-output", type=Path, default=PROJECT_ROOT / "data" / "annotation_working.csv")
    parser.add_argument("--summary-output", type=Path, default=PROJECT_ROOT / "data" / "collection_summary.json")
    parser.add_argument("--per-topic-limit", type=int, default=100)
    parser.add_argument("--per-thread-cap", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=400)
    parser.add_argument("--request-delay", type=float, default=0.75)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing candidate and working files.")
    args = parser.parse_args()
    if not 1 <= args.per_topic_limit <= 100:
        parser.error("--per-topic-limit must be between 1 and 100")
    if args.per_thread_cap < 1 or args.max_candidates < 1 or args.request_delay < 0:
        parser.error("--per-thread-cap and --max-candidates must be positive; --request-delay cannot be negative")
    for output in (args.raw_output, args.working_output, args.summary_output):
        if output.exists() and not args.overwrite:
            parser.error(f"{output} already exists; use --overwrite only for a deliberate recollection.")
    frame, summary = collect(args)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.raw_output, index=False)
    initialize_working_file(args.raw_output, args.working_output)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
