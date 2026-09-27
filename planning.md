# TakeMeter: project plan and taxonomy

## Project scope

TakeMeter will classify individual public comments from [r/3Dprinting](https://www.reddit.com/r/3Dprinting/) by the kind of contribution they make to a technical discussion. The unit of analysis is one comment, especially from troubleshooting, print or design critique, materials or settings, and related technical conversations.

r/3Dprinting is a good fit because it has active, text-heavy technical discourse. People answer the same question in notably different ways: some explain a cause or tradeoff, some give a concrete recommendation with little explanation, some state an answer or opinion, and some respond socially. Those distinctions matter because readers often use comments to diagnose failed prints, choose a material or setting, or make a design decision. The topic also connects to the student's Mechanical Engineering background and hands-on additive-manufacturing experience.

This is a pre-data-collection specification. No dataset, labels, baseline results, model results, or evaluation results are claimed here.

## Initial discourse review and sampling frame

Before finalizing the taxonomy, approximately 35 real public r/3Dprinting comments were reviewed across orientation and part-strength, first-layer and general troubleshooting, filament and materials, TPU, first-printer, and CAD/design discussions. The review showed recurring differences between explanations of why a recommendation follows, concrete but unexplained recommendations, unsupported answers or diagnoses, and social reactions.

It also showed that pure information-seeking questions do not naturally fit this taxonomy. For example, a question about how forces are applied to an object or whether a vendor sells a particular spool size may not itself express a take. The collection frame will therefore include comments that express a take, recommendation, diagnosis, explanation, or reaction. Pure information-seeking questions without one of those functions will be excluded. This is a definition of the sampling frame, not a way to remove difficult examples after reviewing model behavior.

## Label taxonomy

### `reasoned_explanation`

A comment that meaningfully explains a conclusion, diagnosis, recommendation, or viewpoint using causal reasoning, observable evidence, a technical principle, an explicit tradeoff, or relevant experience that explains why the conclusion follows. Merely naming a problem or adding a very short descriptive phrase is not enough; the comment must provide meaningful support.

Examples from the discourse review:

1. A comment recommends printing a functional clip on its edge and explains that layer lines create the weak direction, so the other orientation would put the weakness at the clip legs.
2. In a TPU troubleshooting discussion, a comment explains that flexible filament can buckle or snag in a Bowden setup because of its flexibility and the tube geometry, connects that mechanism to inconsistent extrusion or clogging, and recommends direct drive.

### `actionable_advice`

A comment that gives a concrete recommendation, procedure, setting change, resource, or next step a reader could perform, but does not provide substantial reasoning for why that action should work.

Examples:

1. “Try adjusting the Z offset.”
2. “Pitch it 45 degrees and design a support.”

### `bare_claim`

A comment that states an answer, diagnosis, opinion, preference, or conclusion without substantial supporting reasoning and without sufficiently concrete actionable guidance.

Examples from an orientation-strength discussion:

1. “Edge.”
2. “The one to the right.”

### `reaction`

A comment whose primary function is social or emotional rather than a technical contribution. This includes humor, agreement, appreciation, encouragement, surprise, dismissal, and similar responses that do not materially advance the technical discussion.

Examples:

1. “This is the way.”
2. “Got mine few days ago, best thing ever!”

## Boundary rules

The main boundary is `reasoned_explanation` versus `actionable_advice`. A comment that has both advice and explanation is `reasoned_explanation` only if the explanation supplies a meaningful causal mechanism, observable evidence, technical principle, explicit tradeoff, or relevant experiential rationale showing why the recommendation or conclusion follows. Advice with little or no meaningful rationale is `actionable_advice`.

For example:

```text
“Lower your Z-offset.”
→ actionable_advice

“Lower your Z-offset because the visible gaps between adjacent extrusion lines indicate that the nozzle is too far from the bed.”
→ reasoned_explanation
```

The secondary boundary is `actionable_advice` versus `bare_claim`. A concrete action that the reader can perform is `actionable_advice`; an answer, diagnosis, conclusion, preference, or opinion without a concrete next step is `bare_claim`.

```text
“Your Z-offset is wrong.”
→ bare_claim

“Lower your Z-offset.”
→ actionable_advice
```

Experience alone does not make a comment a `reasoned_explanation`. Experience-backed reasoning may qualify only when it explains why a conclusion or recommendation follows. A statement such as “I use Brand X exclusively and never have problems,” without a developed rationale, is a `bare_claim`.

## Collection and review plan

The target final human-reviewed dataset is 240 examples, aiming for approximately 60 per label:

| Label | Target |
| --- | ---: |
| `reasoned_explanation` | 60 |
| `actionable_advice` | 60 |
| `bare_claim` | 60 |
| `reaction` | 60 |

The exact final counts may vary after review, but no class should dominate. Comments will come from public r/3Dprinting threads across multiple discussion types and threads, rather than relying heavily on one thread. Relevant sources include troubleshooting, print quality, first-layer issues, materials and filament, strength and orientation, slicer and settings, printer setup, CAD/design, and general technical discussion.

Collection will avoid duplicate and near-duplicate comments. It will exclude deleted or removed content, bot or moderator boilerplate where practical, pure information-seeking questions outside the sampling frame, and comments that cannot reasonably be assigned under the taxonomy. If a class is underrepresented, additional relevant public threads likely to contain that discourse type will be sampled. Already reviewed labels will not be changed merely to manufacture balance.

AI-assisted pre-labeling will propose a `proposed_label`, an uncertainty or ambiguity flag, and optionally a brief annotation note. The student will personally review every collected example before its label is final: accepting or changing the proposal, marking difficult cases, or excluding comments that violate the established sampling frame. The final assignment dataset will contain the human-reviewed final labels. Its required columns will be `text` and `label`; provenance and annotation-note fields may be retained to document the workflow and difficult cases accurately.

## Evaluation plan

The final reviewed data will be split deterministically with a fixed random seed into stratified 70% training, 15% validation, and 15% test sets. The test set will be locked before evaluation. Exactly the same locked test comments will be used for the zero-shot Groq baseline and the fine-tuned classifier.

The primary overall metric is accuracy. Additional metrics will be per-class precision, recall, and F1; macro-averaged F1; and a confusion matrix. Accuracy gives an intuitive overall success rate. Per-class metrics are necessary because the task requires learning all four discourse distinctions rather than overpredicting a frequent class. Macro-F1 gives each label equal importance even if the reviewed dataset is not perfectly balanced. The confusion matrix will help locate failed discourse boundaries, especially `reasoned_explanation` versus `actionable_advice` and `actionable_advice` versus `bare_claim`.

The fine-tuned classifier will be considered practically successful only if it achieves all of the following on the locked test set:

- at least 70% accuracy;
- at least 0.65 macro-F1; and
- no individual class F1 below 0.50.

These criteria are set before training and will not be changed after observing model results. The fine-tuned model will be compared honestly with the zero-shot baseline on that same test set.

## Planned model and baseline

The planned fine-tuned model is `distilbert-base-uncased`. The initial configuration will follow the assignment defaults unless later evidence supports a documented change: 3 epochs, learning rate `2e-5`, and batch size 16. Training is planned locally using the verified RTX 4070 CUDA environment.

The planned zero-shot baseline is Groq classification with `meta-llama/llama-4-scout-17b-16e-instruct`, as prescribed by the assignment. That exact model must be attempted first. A prior course project received a real Groq `404 model_not_found` response for it; if that happens here, the actual error will be preserved and documented, the baseline step will stop, and authorization will be obtained before a substitute model is used. No substitution will be silent.

## AI tool plan

### Label stress-testing

Before full annotation, AI will generate 5–10 deliberately ambiguous *synthetic* cases near the `reasoned_explanation`/`actionable_advice`, `actionable_advice`/`bare_claim`, and experience-backed-reasoning/unsupported-preference boundaries. The student will review them. If the rules cannot classify them consistently, definitions will be tightened before full annotation begins. These synthetic cases will never enter the real Reddit dataset.

### Annotation assistance

AI will receive the finalized taxonomy and decision rules alongside real collected comment text, then propose one label per comment and may flag uncertainty. The student will make every final label decision. The annotation record will retain sufficient metadata to disclose this assistance honestly in the future README.

### Failure-analysis assistance

After real evaluation, the actual wrong predictions may be provided to an AI tool to suggest systematic patterns, including confused label pairs, short or low-information comments, technical-vocabulary cues, length, mixed advice and reasoning, sarcasm or humor, and experience-backed statements. The student will re-read the real errors to verify these suggestions. Only verified patterns will be reported.

## Planned stretch features

The project intends to pursue confidence calibration, systematic error-pattern analysis, and a deployed classification interface. Inter-annotator reliability is not planned. It may be added only if a real second human independently labels at least 30 examples; no second annotator or agreement statistic will be invented.

Before implementing any stretch feature, this file will receive a dated or clearly identified update that defines that feature’s implementation and evaluation plan. That update will be committed before the feature is implemented.

## Integrity commitments

All claims in the project will be based on observed work. The project will not fabricate Reddit comments, student review, annotation decisions, API calls or results, training, metrics, confusion matrices, wrong predictions, confidence scores, demo completion, Git state, or another human annotator.

## Post-core-experiment stretch plan — 2026-09-27

This section is a post-experiment addition. The original specification above,
taxonomy, labels, split and success criteria remain unchanged. Core results are
frozen at `a55b3ec73bb33b22314c82f8ebf7aa220c977d30`. Read-only diagnostics found
38/43 fine-tuned predictions were reasoned_explanation, 5 were bare_claim, and
29 were wrong. The baseline and fine-tuned accuracies were 28/43 and 14/43.
Training and validation losses decreased over three epochs, but these observations
do not establish the cause of poor generalization. No retraining, test-driven
tuning, or replacement of the recorded experiment is planned.

### Confidence calibration analysis

Use the saved 43 fine-tuned test probability vectors, without new test inference
or fitting a calibration transformation. Confidence is the maximum of the four
probabilities. Use 10 equal-width bins: [0,.1), [.1,.2), ..., [.9,1], including 1
in the final bin. Report sample count, mean confidence, accuracy and signed gap
per bin; empty bins have null means/accuracy and zero contribution to ECE.
ECE is sum over bins of (bin count / 43) times the absolute accuracy-confidence
gap. Also report the multiclass Brier score as the mean per-example sum of four
squared probability-minus-one-hot errors (not divided by four). Save JSON and
a reliability plot showing the identity line and counts. Distinguish observed
over/underconfidence from proof of calibration; 43 cases and sparse bins limit
precision. Validate probability sums, bin endpoints, empty bins, known synthetic
metric cases, and reproduction from saved predictions.

### Systematic error analysis

Use saved baseline/fine-tuned predictions and errors joined by comment_id to
unchanged finalized text/provenance. Quantify prediction and confusion-pair
frequencies, errors by true/predicted class, paired correctness, disagreements,
and the IDs each model alone gets right. Measure whitespace-delimited word and
character lengths by class and correctness, plus confidence summaries and the
five highest-confidence errors. Include descriptive lexical/structural checks:
printing vocabulary (print*, filament*, nozzle*, layer*, bed, slicer*, PETG, PLA,
TPU, support*), question marks, newlines, and short comments (at most 20 words).
These are exploratory diagnostics, not a fitted explanation or significance test.

Analyze at least three real mistakes illustrating advice, unsupported assertion,
and social reaction: candidates pbz923l, pc1nuul and gyxhfex, all verified against
saved errors. Preserve exact text, source URL, true/predicted labels, confidence,
and all probabilities. Separate measured facts from plausible interpretations;
small training size, imbalance, vocabulary overlap and semantic boundaries are
limitations/hypotheses unless evidence establishes more. Record decreasing loss
without calling it sufficient learning. Save reproducible JSON/CSV and validate
joins, totals, exact examples, error membership, and agreement counts. AI-assisted
interpretations will be checked against the saved text and taxonomy; do not imply
independent student discovery or unperformed human verification.

### Local classification interface

Implement a separate classifier_app.py, preserving the annotation app.py.
Load the existing checkpoint/tokenizer from checkpoints/takemeter-distilbert,
with the recorded label mapping and maximum length 256. Accept arbitrary text;
classify only after a button press; display the label, raw confidence and all four
probabilities. Identify the model as experimental fine-tuned DistilBERT and show
the observed poor test performance. Use CUDA when available and CPU otherwise.
No Groq call or key is needed. Handle empty input, missing/corrupt checkpoints,
and long-text truncation clearly. Explain checkpoint reproduction if missing.
This is local deployment, not a hosted service or completed demo recording.

Validate the classifier with actual local inference on newly supplied smoke-test
text (not another test-set evaluation), probability/mapping checks, CPU inference,
and Streamlit interaction tests for submission, empty input and missing model.
Retain observed smoke-test evidence. No calibration transform is applied.

### Delivery and validation order

Commit and push this planning-only update as Checkpoint A before creating any
stretch scripts/artifacts. After remote verification, implement the three planned
features, write an evidence-grounded README, audit the supplied project rubric,
run tests and read-only core-artifact validation, then commit/push Checkpoint B.
Core artifact hashes and split IDs must remain unchanged. Validators may be
extended to recognize this append-only plan while preserving the original plan
snapshot used by the core experiment. Inter-annotator reliability is excluded:
there is no real second independent human annotator. The student must still
record the 3–5 minute demo; no recording or demo URL will be fabricated.
