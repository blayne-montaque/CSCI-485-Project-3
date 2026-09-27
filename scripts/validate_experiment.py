"""Validate saved experiment evidence without running either model again."""
import json
import math
import re
from pathlib import Path
from dotenv import dotenv_values
from experiment_common import (ROOT, ARTIFACTS, LABELS, LABEL2ID, identities, load_locked_data,
                               now, read_csv, sha256, validate_baseline, write_json)
from train_distilbert import MODEL_DIR


def independent_metrics(predictions):
    matrix = [[0]*4 for _ in range(4)]
    for p in predictions:
        matrix[LABEL2ID[p['true_label']]][LABEL2ID[p['predicted_label']]] += 1
    per_class = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {'precision': precision, 'recall': recall, 'f1-score': f1, 'support': support}
    return {'accuracy': sum(matrix[i][i] for i in range(4))/len(predictions),
            'macro_f1': sum(m['f1-score'] for m in per_class.values())/4,
            'per_class': per_class, 'confusion_matrix': matrix}


def assert_metrics(actual, saved):
    for metric in ['accuracy', 'macro_f1']:
        assert math.isclose(actual[metric], saved[metric], abs_tol=1e-12)
    assert actual['confusion_matrix'] == saved['confusion_matrix']
    assert sum(map(sum, actual['confusion_matrix'])) == saved['count']
    for label in LABELS:
        for key, value in actual['per_class'][label].items():
            assert math.isclose(value, saved['per_class'][label][key], abs_tol=1e-12)


def validate():
    splits = load_locked_data()
    baseline = validate_baseline(splits)
    final = json.loads((ARTIFACTS/'finetuned_results.json').read_text())
    config = json.loads((ARTIFACTS/'training_config.json').read_text())
    training = json.loads((ARTIFACTS/'training_results.json').read_text())
    comparison = json.loads((ARTIFACTS/'model_comparison.json').read_text())
    test_run = json.loads((ARTIFACTS/'finetuned_test_run.json').read_text())
    expected = identities(splits)
    for result in [baseline, final, config]:
        assert result['input_sha256'] == expected['input_sha256']
        assert result['split_ids'] == expected['split_ids']
        assert result['label2id'] == LABEL2ID
    assert training['training_config_sha256'] == sha256(ARTIFACTS/'training_config.json')
    assert test_run['training_config_sha256'] == training['training_config_sha256']
    assert test_run['model_files_sha256'] == training['model_files_sha256']
    assert test_run['selected_epoch'] == training['selected_epoch']
    for filename, digest in training['model_files_sha256'].items():
        assert sha256(MODEL_DIR/filename) == digest
    assert [h['epoch'] for h in training['history']] == [1, 2, 3]
    assert training['test_forward_passes'] == 0 and training['optimization_steps'] == 39
    val_predictions = read_csv(ARTIFACTS/'validation_predictions.csv')
    train_ids = set(expected['split_ids']['train'])
    val_ids = set(expected['split_ids']['validation'])
    test_ids = expected['split_ids']['test']
    for epoch in training['history']:
        assert len(epoch['optimization_ids']) == 197 and set(epoch['optimization_ids']) == train_ids
        rows = [r for r in val_predictions if int(r['epoch']) == epoch['epoch']]
        assert len(rows) == 42 and {r['comment_id'] for r in rows} == val_ids
        truth = {r['comment_id']: r['final_label'] for r in splits['validation']}
        assert all(row['true_label'] == truth[row['comment_id']] for row in rows)
        assert_metrics(independent_metrics(rows), epoch['validation_metrics'])
    best = min(training['history'], key=lambda h:h['validation_loss'])
    assert best['epoch'] == training['selected_epoch']
    assert config['epochs'] == 3 and config['learning_rate'] == 2e-5 and config['batch_size'] == 16
    assert set(config['length_statistics']) == {'train', 'validation'}
    assert config['max_sequence_length'] == 256 and config['seed'] == 42
    tables = {}
    truth = {r['comment_id']: r for r in splits['test']}
    for name, result in [('baseline', baseline), ('finetuned', final)]:
        rows = read_csv(ARTIFACTS/f'{name}_predictions.csv')
        assert len(rows) == 43 and [r['comment_id'] for r in rows] == test_ids
        assert all(r['true_label'] == truth[r['comment_id']]['final_label'] for r in rows)
        assert all(r['predicted_label'] in LABELS for r in rows)
        assert_metrics(independent_metrics(rows), result)
        tables[name] = rows
    for row in tables['finetuned']:
        probs = [float(row['probability_'+label]) for label in LABELS]
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probs)
        assert math.isclose(sum(probs), 1.0, abs_tol=1e-6)
        assert LABELS[probs.index(max(probs))] == row['predicted_label']
        assert float(row['confidence']) == max(probs)
    errors = read_csv(ARTIFACTS/'finetuned_errors.csv')
    expected_errors = [r for r in tables['finetuned'] if r['true_label'] != r['predicted_label']]
    assert len(errors) == len(expected_errors)
    for error, prediction in zip(errors, expected_errors):
        assert all(error[k] == v for k,v in prediction.items())
        assert error['text'] == truth[error['comment_id']]['text']
        assert error['source_url'] == truth[error['comment_id']]['source_url']
    assert comparison['test_ids'] == test_run['test_ids'] == test_ids
    for name, result in [('baseline', baseline), ('finetuned', final)]:
        assert comparison[name] == {k:result[k] for k in ['model','accuracy','macro_f1','per_class']}
    for metric in ['accuracy','macro_f1']:
        assert math.isclose(comparison['absolute_metric_differences_finetuned_minus_baseline'][metric],
                            final[metric]-baseline[metric], abs_tol=1e-12)
    for label in LABELS:
        for metric in ['precision', 'recall', 'f1-score']:
            assert math.isclose(comparison['per_class_differences_finetuned_minus_baseline'][label][metric],
                                final['per_class'][label][metric]-baseline['per_class'][label][metric],
                                abs_tol=1e-12)
    expected_thresholds = {
        'accuracy_at_least_0.70': final['accuracy'] >= 0.70,
        'macro_f1_at_least_0.65': final['macro_f1'] >= 0.65,
        'every_class_f1_at_least_0.50': all(p['f1-score'] >= 0.50 for p in final['per_class'].values())}
    assert final['success_thresholds'] == expected_thresholds
    assert final['all_success_thresholds_met'] == all(expected_thresholds.values())
    secret = dotenv_values(ROOT/'.env').get('GROQ_API_KEY', '')
    scanned = []
    for path in list(ARTIFACTS.glob('*')) + list((ROOT/'scripts').glob('*.py')):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        assert not secret or secret.encode() not in raw, f'Secret in {path.name}'
        assert not re.search(rb'gsk_[a-zA-Z0-9_-]{20,}', raw), f'Secret-like value in {path.name}'
        scanned.append(str(path.relative_to(ROOT)))
    return {'validated_at': now(), 'passed': True, 'total_examples': 282,
            'split_counts': {k:len(v) for k,v in splits.items()},
            'identical_test_ids': test_ids, 'label2id': LABEL2ID,
            'baseline_predictions': 43, 'finetuned_predictions': 43,
            'finetuned_error_count': len(errors),
            'protected_input_sha256': expected['input_sha256'],
            'checks': ['committed dataset, labels, split, prompt and Scout failure unchanged',
                       'disjoint persisted IDs; optimization visits only 197 train IDs per epoch',
                       'model selected only by 42 validation rows',
                       'baseline request hashes contain only saved prompt and test text',
                       'both prediction files contain identical 43 test IDs',
                       'metrics independently recomputed; confusion totals match',
                       'softmax probabilities finite, bounded, sum to one',
                       'error export exactly matches incorrect fine-tuned predictions',
                       'model checkpoint and training configuration hashes verified',
                       'secret scan passed'],
            'secret_scanned_files': scanned}


if __name__ == '__main__':
    result = validate()
    output = ARTIFACTS/'experiment_validation.json'
    if not output.exists():
        write_json(output, result)
    print(json.dumps(result, indent=2))
