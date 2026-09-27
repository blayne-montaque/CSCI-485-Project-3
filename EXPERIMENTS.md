# Reproducing the locked TakeMeter experiment

The authoritative split is `data/split_assignments.csv`, committed at
`24b9451b0bf3f00f697ff61436721795dbf45d51`: 197 train, 42 validation, 43 test.
Experiment code loads this mapping and checks protected inputs against that
checkpoint. It never invokes the split generator. Stable IDs come from the
reviewed working rows, whose text/label projection must exactly match
`data/labeled_posts.csv`. The final dataset, labels, taxonomy, saved prompt,
split, and prescribed-model failure evidence remain unchanged. The original plan
is preserved verbatim; a later stretch section was committed before stretch work.
Validation checks that extension against planning checkpoint
`c11e86058d9a42693b282231b067da6937fe04b1` while retaining the original planning hash
in the historical experiment identity.

The fixed label/ID order for every metric, matrix, and probability row is:
`reasoned_explanation` (0), `actionable_advice` (1), `bare_claim` (2), `reaction` (3).
Confusion matrices use true labels as rows and predicted labels as columns.

## Baseline provenance

The prescribed `meta-llama/llama-4-scout-17b-16e-instruct` returned an actual Groq
HTTP 404 `model_not_found`; the original evidence remains in
`artifacts/prescribed_baseline_failure.json`. The student explicitly authorized
`openai/gpt-oss-20b` because of that failure. This is an authorized substitute,
not a result from the prescribed model.

The runner reads `artifacts/baseline_prompt_template.txt` verbatim and uses
temperature 0. It sends only the template and one locked test comment, with no
label metadata or demonstrations. A format-only retry is allowed for malformed
output; bounded rate-limit retries repeat the same request. The request journal
records prompt hashes, response IDs, returned model names, actual raw content,
and sanitized API errors. True labels are joined after inference. Metrics require
all 43 predictions to parse successfully; there is no automatic model fallback.

## Training and selection

The base is `distilbert-base-uncased`, resolved to revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`. The recorded run uses 3 epochs,
learning rate `2e-5`, batch size 16, seed 42, and float32 CUDA on the RTX 4070.
The optimizer is AdamW with zero weight decay; the learning rate decays linearly
over 39 optimizer steps with no warmup. Gradients are clipped to norm 1.
The full configuration, package versions, and actual device are in
`artifacts/training_config.json`.

Maximum sequence length is chosen before training/test inference from the 95th
percentile of train+validation token lengths, rounded up to 64 and bounded to
128–512. The observed choice was 256: 10/197 train and 1/42 validation comments
are truncated. Test text lengths and correctness do not affect this choice.

Only training rows enter optimization. Validation cross-entropy is computed
after each epoch; the lowest loss selects the checkpoint, with the earliest
epoch winning a tie. Epoch 3 was selected. Training history records actual
optimization IDs per epoch, validation metrics, and elapsed wall time. Seeds,
deterministic algorithms, eager attention, and CUDA determinism settings are
explicit; exact replay across different hardware/software is not guaranteed.

`evaluate_distilbert.py` loads the selected saved model and performs one final
inference pass on the same 43 test IDs as the baseline. An exclusive run marker
and output guards prevent repeating or overwriting final test evaluation.
Confidence values are raw softmax probabilities, without calibration.

## Local model handling and commands

The selected model and tokenizer are in `checkpoints/takemeter-distilbert/`.
That entire directory and the download cache `.cache/huggingface/` are ignored.
Weights are not committed. Their SHA-256 hashes are recorded in
`artifacts/training_results.json`, alongside the selected epoch and runtime.
Use the existing CUDA environment setup in `SETUP.md` and `requirements.txt`.

Read-only verification of the completed run (requires the local saved model):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\validate_experiment.py
```

The production scripts intentionally refuse to overwrite experiment artifacts.
For a separately authorized reproduction, use a disposable checkout and archive
the prior generated run outputs there first. Preserve `data/`, `planning.md`,
`artifacts/split_summary.json`, `artifacts/baseline_prompt_template.txt`, and
`artifacts/prescribed_baseline_failure.json`. Do not regenerate the split.
To reproduce training without another API evaluation, keep the original
baseline predictions, results, and request journal, and archive only training
and fine-tuned evaluation outputs and the ignored local model directory.

The ordered execution commands for a fresh experiment workspace are:

```powershell
# Requires the private, ignored .env and explicit substitute authorization.
.\.venv\Scripts\python.exe scripts\run_groq_zero_shot_baseline.py --model openai/gpt-oss-20b --confirm-authorized-substitute
# Training validates the complete baseline evidence before proceeding.
.\.venv\Scripts\python.exe scripts\train_distilbert.py --revision 12040accade4e8a0f71eabdb258fecc2e7e948be
.\.venv\Scripts\python.exe scripts\evaluate_distilbert.py
.\.venv\Scripts\python.exe scripts\validate_experiment.py
```

## Evidence artifacts

- `baseline_requests.jsonl`, `baseline_predictions.csv`, `baseline_results.json`,
  `baseline_confusion_matrix.png`: actual substitute baseline evidence. Baseline
  errors are exactly the prediction rows where true and predicted labels differ.
- `training_config.json`, `training_results.json`, `validation_predictions.csv`:
  configuration fixed before optimization, observed training history, selection,
  environment, model hashes, and validation predictions for all three epochs.
- `finetuned_test_run.json`, `finetuned_predictions.csv`, `finetuned_results.json`,
  `confusion_matrix.png`: final test evidence, with confidence and all four class
  probabilities per prediction.
- `finetuned_errors.csv`: all actual wrong predictions, exact text, provenance,
  confidence and class probabilities. No qualitative error analysis is included.
- `model_comparison.json`: observed metrics and arithmetic differences.
- `baseline_validation.json`, `experiment_validation.json`: completed checks.

All listed evidence files are under `artifacts/`. The validator independently
recomputes precision, recall, F1, accuracy, and confusion counts from saved
predictions without invoking either model. It checks probabilities, exact error
exports, model hashes, training/validation membership, and unchanged protected
inputs. Model downloads/API responses are actual observations, not test fixtures.

Implementation references: [Hugging Face DistilBERT documentation](https://huggingface.co/docs/transformers/model_doc/distilbert)
and [PyTorch reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html).
