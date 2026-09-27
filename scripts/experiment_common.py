"""Read-only locked data access and shared evaluation definitions."""
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from lock_dataset_split import VALID_LABELS, load_finalized_rows, read_csv, validate_assignments

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'
LOCK_CHECKPOINT = '24b9451b0bf3f00f697ff61436721795dbf45d51'
STRETCH_PLAN_CHECKPOINT = 'c11e86058d9a42693b282231b067da6937fe04b1'
LABELS = list(VALID_LABELS)
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
PROTECTED = ['data/labeled_posts.csv', 'data/annotation_working.csv',
             'data/split_assignments.csv', 'artifacts/split_summary.json',
             'artifacts/baseline_prompt_template.txt',
             'artifacts/prescribed_baseline_failure.json', 'planning.md']


def now():
    return datetime.now(UTC).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def write_csv(path, rows, fields=None):
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_locked_data():
    # Validate against the committed lock, not merely matching split sizes.
    for name in PROTECTED:
        committed = subprocess.check_output(['git', 'show', f'{LOCK_CHECKPOINT}:{name}'], cwd=ROOT)
        current = (ROOT / name).read_bytes()
        if name == 'planning.md':
            extended = subprocess.check_output(
                ['git', 'show', f'{STRETCH_PLAN_CHECKPOINT}:planning.md'], cwd=ROOT)
            if not extended.replace(b'\r\n', b'\n').startswith(committed.replace(b'\r\n', b'\n')):
                raise ValueError('Stretch plan changed the original specification')
            committed = extended
        if current.replace(b'\r\n', b'\n') != committed.replace(b'\r\n', b'\n'):
            raise ValueError(f'Authoritative checkpoint changed: {name}')
    rows = load_finalized_rows(ROOT / 'data/annotation_working.csv', ROOT / 'data/labeled_posts.csv')
    if Counter(row['final_label'] for row in rows) != dict(zip(LABELS, [89, 50, 83, 60])):
        raise ValueError('Unexpected final distribution')
    assignments = read_csv(ROOT / 'data/split_assignments.csv')
    validate_assignments(rows, assignments)
    split_by_id = {row['comment_id']: row['split'] for row in assignments}
    return {name: [row for row in rows if split_by_id[row['comment_id']] == name]
            for name in ['train', 'validation', 'test']}


def identities(splits):
    # The historical core experiment references its original planning snapshot.
    hashes = {name: sha256(ROOT / name) for name in PROTECTED}
    hashes['planning.md'] = hashlib.sha256(subprocess.check_output(
        ['git', 'show', f'{LOCK_CHECKPOINT}:planning.md'], cwd=ROOT)).hexdigest()
    return {
        'lock_checkpoint': LOCK_CHECKPOINT,
        'input_sha256': hashes,
        'split_ids': {name: [r['comment_id'] for r in rows] for name, rows in splits.items()},
        'label_order': LABELS,
        'label2id': LABEL2ID,
    }


def metrics(truth, predictions):
    if not truth or len(truth) != len(predictions):
        raise ValueError('Metrics require complete actual predictions')
    if any(label not in LABELS for label in truth + predictions):
        raise ValueError('Invalid metric label')
    report = classification_report(truth, predictions, labels=LABELS, output_dict=True, zero_division=0)
    return {'count': len(truth), 'accuracy': accuracy_score(truth, predictions),
            'macro_f1': f1_score(truth, predictions, labels=LABELS, average='macro', zero_division=0),
            'per_class': {label: report[label] for label in LABELS},
            'label_order': LABELS,
            'confusion_matrix': confusion_matrix(truth, predictions, labels=LABELS).tolist()}


def plot_confusion(result, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    import numpy as np
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(np.array(result['confusion_matrix']), display_labels=LABELS).plot(
        ax=ax, cmap='Blues', colorbar=False, values_format='d', xticks_rotation=25)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def validate_baseline(splits):
    from run_groq_zero_shot_baseline import build_prompt, parse_label, SUBSTITUTE_MODEL
    predictions = read_csv(ARTIFACTS / 'baseline_predictions.csv')
    result = json.loads((ARTIFACTS / 'baseline_results.json').read_text(encoding='utf-8'))
    test = {r['comment_id']: r for r in splits['test']}
    if len(predictions) != 43 or {p['comment_id'] for p in predictions} != set(test):
        raise ValueError('Baseline IDs do not match the locked test set')
    template = (ARTIFACTS / 'baseline_prompt_template.txt').read_text(encoding='utf-8')
    journal = [json.loads(line) for line in (ARTIFACTS / 'baseline_requests.jsonl').read_text().splitlines()]
    last_responses = {}
    for request in journal:
        if request['comment_id'] not in test:
            raise ValueError('Non-test baseline request detected')
        prompt = build_prompt(template, test[request['comment_id']]['text'], request['format_attempt'] == 2)
        assert request['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
        assert request['model'] == SUBSTITUTE_MODEL and request['temperature'] == 0
        if 'raw_response' in request:
            assert request['returned_model'] == SUBSTITUTE_MODEL
            last_responses[request['comment_id']] = request
    for prediction in predictions:
        assert prediction['true_label'] == test[prediction['comment_id']]['final_label']
        assert prediction['parse_status'] == 'valid'
        assert prediction['predicted_label'] == parse_label(prediction['raw_model_response'])
        assert last_responses[prediction['comment_id']]['raw_response'] == prediction['raw_model_response']
    computed = metrics([p['true_label'] for p in predictions], [p['predicted_label'] for p in predictions])
    assert all(result[key] == value for key, value in computed.items())
    assert result['input_sha256'] == identities(splits)['input_sha256']
    assert result['parse_failures'] == 0 and result['model'] == SUBSTITUTE_MODEL
    assert result['substitution']['student_authorized']
    return result
