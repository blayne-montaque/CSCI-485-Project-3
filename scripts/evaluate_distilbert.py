"""One final locked-test inference pass; export metrics, comparison and errors."""
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from experiment_common import (ARTIFACTS, LABELS, LABEL2ID, identities, load_locked_data,
                               metrics, now, plot_confusion, sha256, validate_baseline,
                               write_csv, write_json)
from train_distilbert import MODEL_DIR, CommentDataset, infer, seed_everything


def main():
    splits = load_locked_data()
    baseline = validate_baseline(splits)
    config = json.loads((ARTIFACTS/'training_config.json').read_text())
    training = json.loads((ARTIFACTS/'training_results.json').read_text())
    assert training['training_config_sha256'] == sha256(ARTIFACTS/'training_config.json')
    assert config['input_sha256'] == identities(splits)['input_sha256']
    for filename, digest in training['model_files_sha256'].items():
        assert sha256(MODEL_DIR/filename) == digest
    for name in ['finetuned_test_run.json', 'finetuned_predictions.csv', 'finetuned_results.json',
                 'finetuned_errors.csv', 'model_comparison.json']:
        if (ARTIFACTS/name).exists():
            raise FileExistsError(f'Refusing to repeat final test evaluation: {name}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True, attn_implementation='eager').to('cuda:0')
    assert model.config.label2id == LABEL2ID
    assert model.config.id2label == dict(enumerate(LABELS))
    data = CommentDataset(splits['test'], tokenizer, config['max_sequence_length'])
    loader = DataLoader(data, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    # An exclusive marker prevents a second test pass, even following a crash.
    write_json(ARTIFACTS/'finetuned_test_run.json', {
        'started_at': now(), 'test_ids': [r['comment_id'] for r in splits['test']],
        'selected_epoch': training['selected_epoch'], 'device': 'cuda:0',
        'model_files_sha256': training['model_files_sha256'],
        'training_config_sha256': training['training_config_sha256'],
        'purpose': 'single final evaluation; no model selection or retuning'})
    start = time.perf_counter()
    probs, test_loss = infer(model, loader, torch.device('cuda:0'))
    torch.cuda.synchronize()
    runtime = time.perf_counter()-start
    predictions = []
    for row, distribution in zip(splits['test'], probs):
        index = int(np.argmax(distribution))
        predictions.append({
            'comment_id': row['comment_id'], 'true_label': row['final_label'],
            'predicted_label': LABELS[index], 'confidence': distribution[index],
            **{'probability_'+label: float(distribution[i]) for i, label in enumerate(LABELS)}})
    assert len(predictions) == 43
    write_csv(ARTIFACTS/'finetuned_predictions.csv', predictions)
    result = metrics([p['true_label'] for p in predictions], [p['predicted_label'] for p in predictions])
    result.update(model=config['base_model'], selected_epoch=training['selected_epoch'],
                  evaluated_at=now(), evaluation_runtime_seconds=runtime, test_loss=test_loss,
                  confidence_note='Uncalibrated softmax probability for the predicted class.',
                  training_config_sha256=training['training_config_sha256'], **identities(splits))
    result['success_thresholds'] = {
        'accuracy_at_least_0.70': result['accuracy'] >= 0.70,
        'macro_f1_at_least_0.65': result['macro_f1'] >= 0.65,
        'every_class_f1_at_least_0.50': all(m['f1-score'] >= 0.50 for m in result['per_class'].values())}
    result['all_success_thresholds_met'] = all(result['success_thresholds'].values())
    write_json(ARTIFACTS/'finetuned_results.json', result)
    plot_confusion(result, ARTIFACTS/'confusion_matrix.png', 'Fine-tuned DistilBERT (43 locked test comments)')
    lookup = {r['comment_id']: r for r in splits['test']}
    errors = [{**p, 'text': lookup[p['comment_id']]['text'], 'source_url': lookup[p['comment_id']]['source_url']}
              for p in predictions if p['true_label'] != p['predicted_label']]
    write_csv(ARTIFACTS/'finetuned_errors.csv', errors, list(predictions[0])+['text','source_url'])
    comparison = {
        'test_ids': [r['comment_id'] for r in splits['test']],
        'baseline': {k: baseline[k] for k in ['model','accuracy','macro_f1','per_class']},
        'finetuned': {k: result[k] for k in ['model','accuracy','macro_f1','per_class']},
        'absolute_metric_differences_finetuned_minus_baseline': {
            k: result[k]-baseline[k] for k in ['accuracy','macro_f1']},
        'per_class_differences_finetuned_minus_baseline': {
            label: {metric: result['per_class'][label][metric]-baseline['per_class'][label][metric]
                    for metric in ['precision','recall','f1-score']} for label in LABELS},
        'note': 'Differences are arithmetic score differences, not relative percentage changes.',
    }
    write_json(ARTIFACTS/'model_comparison.json', comparison)
    print(json.dumps({k: result[k] for k in ['accuracy','macro_f1','success_thresholds']}, indent=2))
    print(f'Exported {len(errors)} actual test errors.')


if __name__ == '__main__':
    main()
