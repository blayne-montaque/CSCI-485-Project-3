# Non-demo repository audit

Audited against the assignment checklist supplied by the student. Status:
**non-demo checks passed with documented limitations**. The demo is pending.
No test-set rerun, retraining, relabeling, or split replacement was performed.

| Area | Substantive evidence and verdict |
|---|---|
| Planning | Original community rationale, four definitions, two examples per class, explicit hard boundary, sampling/balance targets, metric rationale, success thresholds and AI plan remain intact. Planning-only commit `c11e86058d9a42693b282231b067da6937fe04b1` precedes implementation; new stretch paths are absent from that tree. |
| Dataset | 282 reviewed examples; counts 89/50/83/60; unique IDs/normalized text, provenance, and finalized projection verified by core checks. All data files unchanged from the experiment checkpoint. No class dominates. Three real post-audit difficult cases, two corrections, initial-versus-audit timing disclosed. |
| Collection limitation | Original retrieval/deduplication summaries and supplemental archived collection documented. Question-like accepted `gxiw1ug` remains a sampling-frame tension. No human label was changed during this audit. |
| Baseline | Actual Scout 404 evidence preserved; explicit authorized gpt-oss-20b substitution disclosed. Request hashes reconstruct prompt+text, with no truth metadata; all 43 correct locked IDs, zero unresolved parse failures. Saved metrics recomputed. |
| Training | Recorded pretrained revision, seed/config, actual CUDA environment, 39 optimization steps, train-only optimization IDs, validation-only selection and model hashes verified. No retraining in this run. |
| Evaluation | Both saved prediction files contain identical 43 IDs. Accuracy, all class metrics, macro-F1 and confusion matrices independently recomputed; probability rows valid. README metrics match artifacts; poor result and failed thresholds explicit. |
| Calibration | Reproduced 10-bin ECE and multiclass Brier; bin endpoints, empty bins and known synthetic cases tested. Reliability plot visually inspected. No calibration transform or test tuning. |
| Error analysis | All 29 errors reproduced; prediction/confusion counts, lengths, confidence, paired-model results and declared lexical/structural measurements. Three exact errors and all four probabilities included. Interpretations distinguish observations from hypotheses. |
| Local interface | Separate classifier_app.py preserves annotation app byte-for-byte. Real CPU/CUDA offline smoke inference; Streamlit submit/empty/missing-model tests passed. Four probabilities, stable labels, truncation notice, poor-result disclosure and recovery instructions present. |
| README | Covers dataset/annotation/audit history, methods, observed comparisons, actual examples, learned-vs-intended reflection, calibration, stretch scope, setup, AI disclosure, limits and pending demo. Local links checked. No invented video URL. |
| Reproducibility | Pinned installed requirements checked. Setup's stale empty-project wording fixed. EXPERIMENTS explains fixed revision and guarded reproduction. Cache/weights ignored; local checkpoint required for actual inference and full model-hash audit. |
| Integrity | 33 pre-existing core data/artifact/app/requirements files unchanged. Planning validator now verifies only the authorized append-only plan while retaining the historical original-plan identity. No metrics or predictions altered. |
| Security | Exact-key and key-pattern scan over tracked/untracked deliverable candidates passed; .env untracked/ignored; model weights ignored; no oversized deliverable files. |
| Tests | **19 tests passed**, zero failures, including actual local Streamlit inference. Core artifact validation also passed. |
| Remaining deliverable | Student records the 3–5 minute demo; no recording or upload performed. |

Known limitations remain visible in README. In particular, this split is disjoint
by comment ID, not by thread: **19 thread IDs overlap train and test**. This limits
independence but does not change recorded optimization membership. Small-sample
calibration, AI-label anchoring, local-only weights, and the poor model result are
not silently presented as resolved. These are study limitations, not evidence of
a computational bug invalidating the saved metrics.

Final staging/commit/push and clean-tree checks are performed after this audit.
Their observed status is reported to the student; this document does not preclaim
a future commit hash or successful push.
