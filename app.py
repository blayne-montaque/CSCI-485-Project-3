"""Streamlit interface for explicit human review of TakeMeter candidates."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from takemeter.workflow import (
    TARGET_PER_LABEL,
    VALID_LABELS,
    accepted_rows,
    apply_review_action,
    load_working,
)

WORKING_FILE = PROJECT_ROOT / "data" / "annotation_working.csv"


@st.cache_data(show_spinner=False)
def read_working_file(modified_at_ns: int) -> pd.DataFrame:
    return load_working(WORKING_FILE)


def refresh() -> None:
    st.cache_data.clear()
    st.rerun()


st.set_page_config(page_title="TakeMeter annotation review", layout="wide")
st.title("TakeMeter annotation review")
st.caption("AI suggestions are optional. A row is reviewed only after you explicitly save or exclude it.")

if not WORKING_FILE.exists():
    st.error("annotation_working.csv is missing. Run scripts/collect_reddit.py first.")
    st.stop()

frame = read_working_file(WORKING_FILE.stat().st_mtime_ns)
accepted = accepted_rows(frame)
reviewed = int(frame["human_reviewed"].sum())
excluded = int(frame["excluded"].sum())

with st.sidebar:
    st.header("Progress")
    st.metric("Reviewed", f"{reviewed} / {len(frame)}")
    st.metric("Accepted", len(accepted))
    st.metric("Excluded", excluded)
    st.divider()
    st.caption("Human-reviewed accepted labels")
    for label in VALID_LABELS:
        st.write(f"{label}: {int((accepted['final_label'] == label).sum())} / {TARGET_PER_LABEL}")
    st.divider()
    filter_status = st.selectbox("Show", ("Unreviewed", "All", "Reviewed", "Accepted", "Excluded", "Ambiguous"))
    proposed_label_filter = st.selectbox(
        "AI proposed label filter", ("All proposed labels",) + VALID_LABELS
    )
    label_filter = st.selectbox("Final label filter", ("All labels",) + VALID_LABELS)
    st.caption("Unreviewed AI proposed labels")
    unreviewed = frame.loc[~frame["human_reviewed"]]
    for label in VALID_LABELS:
        st.write(f"{label}: {int((unreviewed['proposed_label'] == label).sum())}")

visible = frame.copy()
if filter_status == "Unreviewed":
    visible = visible.loc[~visible["human_reviewed"]]
elif filter_status == "Reviewed":
    visible = visible.loc[visible["human_reviewed"]]
elif filter_status == "Accepted":
    visible = accepted
elif filter_status == "Excluded":
    visible = visible.loc[visible["excluded"]]
elif filter_status == "Ambiguous":
    visible = visible.loc[visible["ambiguous"] | visible["ai_ambiguous"]]
if proposed_label_filter != "All proposed labels":
    visible = visible.loc[visible["proposed_label"] == proposed_label_filter]
if label_filter != "All labels":
    visible = visible.loc[visible["final_label"] == label_filter]

if visible.empty:
    st.info("No candidates match the current filters.")
    st.stop()

visible = visible.reset_index(drop=True)
if "candidate_position" not in st.session_state:
    st.session_state.candidate_position = 0
st.session_state.candidate_position = min(st.session_state.candidate_position, len(visible) - 1)
position = st.session_state.candidate_position
row = visible.iloc[position]

nav_left, nav_middle, nav_right = st.columns((1, 3, 1))
with nav_left:
    if st.button("Previous", disabled=position == 0):
        st.session_state.candidate_position -= 1
        st.rerun()
with nav_middle:
    st.write(f"Candidate {position + 1} of {len(visible)} in this view")
with nav_right:
    if st.button("Next", disabled=position >= len(visible) - 1):
        st.session_state.candidate_position += 1
        st.rerun()

st.subheader("Comment")
st.text_area("", value=row["text"], height=190, disabled=True, label_visibility="collapsed")
metadata_left, metadata_right = st.columns(2)
with metadata_left:
    st.write(f"**Thread:** {row['thread_title'] or 'Title unavailable from archive'}")
    st.link_button("Open original Reddit comment", row["source_url"])
with metadata_right:
    st.write(f"**AI proposed label:** {row['proposed_label'] or 'Not pre-labeled'}")
    st.write(f"**AI ambiguity flag:** {'Yes' if row['ai_ambiguous'] else 'No'}")
    if row["ai_note"]:
        st.caption(f"AI note: {row['ai_note']}")

st.divider()
st.subheader("Your review")
options = ("Choose a label",) + VALID_LABELS
default_label = row["final_label"] if row["final_label"] in VALID_LABELS else ""
choice = st.radio("Final label", options, index=options.index(default_label) if default_label else 0, horizontal=True)
ambiguous = st.checkbox("Flag as genuinely difficult or ambiguous", value=bool(row["ambiguous"]))
note = st.text_area("Optional review note", value=row["review_notes"], max_chars=1000)

action_left, action_middle, action_right = st.columns(3)
with action_left:
    if st.button("Accept proposed label", disabled=row["proposed_label"] not in VALID_LABELS, type="primary"):
        apply_review_action(WORKING_FILE, row["comment_id"], action="save", final_label=row["proposed_label"], ambiguous=ambiguous, review_notes=note)
        st.session_state.candidate_position = min(position, max(len(visible) - 2, 0))
        refresh()
with action_middle:
    if st.button("Save selected label"):
        if choice == "Choose a label":
            st.error("Choose one final label before saving.")
        else:
            apply_review_action(WORKING_FILE, row["comment_id"], action="save", final_label=choice, ambiguous=ambiguous, review_notes=note)
            st.session_state.candidate_position = min(position, max(len(visible) - 2, 0))
            refresh()
with action_right:
    if st.button("Exclude candidate"):
        apply_review_action(WORKING_FILE, row["comment_id"], action="exclude", ambiguous=ambiguous, review_notes=note)
        st.session_state.candidate_position = min(position, max(len(visible) - 2, 0))
        refresh()

st.caption("Saving a label or excluding a candidate is a deliberate human-review action. Viewing or navigating never changes review state.")
