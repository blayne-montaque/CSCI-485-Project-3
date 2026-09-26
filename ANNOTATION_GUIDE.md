# TakeMeter collection and annotation guide

## Data files

- `data/raw_candidates.csv` contains real public r/3Dprinting candidate comments and retrieved provenance. It contains no author names and no labels.
- `data/annotation_working.csv` starts as a copy of the raw pool and holds AI suggestions, review state, and the student's eventual final labels. It is not the final assignment dataset.
- `data/collection_summary.json` records the actual retrieval and deterministic-filter counts for the collection run.
- `data/prelabel_summary.json` is created only after a configured Groq pre-labeling run.
- `data/labeled_posts.csv` must not exist until every accepted example has been personally reviewed and finalization is deliberately run.

The collector uses the Arctic Shift public Reddit archive API. It queries r/3Dprinting comments using multiple technical-topic terms, then retrieves source post titles by thread ID. It saves comment text, comment permalink, thread ID and URL, title, timestamps, public score, retrieval time, and source-topic query. It deliberately does not save authors.

The collection filters deleted or empty entries, practical bot or moderator boilerplate, clearly pure short questions, duplicate IDs, normalized-text duplicates, and comments above its per-thread cap. It does not remove comments because they are difficult to classify. Review `data/collection_summary.json` for the exact results of a run.

## Start the review interface

From the repository root in PowerShell, run:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

The interface opens locally. It reads and saves `data/annotation_working.csv` continuously. Stop with `Ctrl+C`, then rerun the same command to resume.

Each screen shows the comment, source thread, original comment link, any AI suggestion, and review progress. Viewing a comment or using Previous/Next never marks it reviewed.

Use one deliberate action for every candidate:

- **Accept proposed label** saves the displayed AI suggestion as your final label.
- **Save selected label** saves the label you choose.
- **Exclude candidate** records that you reviewed the candidate and chose not to include it.

You can flag a valid but difficult example as ambiguous and add a review note. This preserves genuinely difficult real cases for later analysis. AI suggestions are suggestions only; `human_reviewed` changes only through an explicit review action.

The sidebar gives reviewed and accepted totals and human-reviewed accepted counts toward the planned 60 examples per label. Aim for about 240 accepted examples with no class dominating. If a class is short, review more candidates or collect from more relevant threads; do not change labels merely to manufacture balance.

## Optional AI pre-labeling

Pre-labeling requires a local `.env` containing `GROQ_API_KEY`, which must remain uncommitted. The assignment baseline model is not automatically used for annotation. After deliberately choosing an available annotation model, run:

```powershell
.\.venv\Scripts\python.exe scripts\prelabel_groq.py --model YOUR_CHOSEN_GROQ_MODEL
```

The script uses the taxonomy and boundary rules from `planning.md`, requests one valid proposed label plus an ambiguity flag and short note, and writes no final labels or human-review actions. It records the exact model and outcomes in `data/prelabel_summary.json`. Invalid or failed responses remain unclassified rather than being guessed.

## Finalize only after personal review

After personally reviewing every accepted row, validate and produce the assignment-compatible two-column CSV:

```powershell
.\.venv\Scripts\python.exe scripts\validate_annotation_data.py
.\.venv\Scripts\python.exe scripts\finalize_dataset.py --confirm-review-complete
```

The finalizer refuses to run without the explicit flag and refuses to overwrite an existing final CSV. It accepts only rows where `human_reviewed` is true, `excluded` is false, and `final_label` is one of the four taxonomy labels. It validates duplicate IDs and normalized text, prints class counts, identifies classes below 60, and flags a class above half of the accepted set. Do not create train/validation/test splits or start model work until review and finalization are genuinely complete.

## Recollection

The committed raw pool is the provenance record for this checkpoint. Recollect only deliberately:

```powershell
.\.venv\Scripts\python.exe scripts\collect_reddit.py --overwrite
```

This replaces the raw and working candidate files. Do not use it after review begins unless you have preserved the current review work.
