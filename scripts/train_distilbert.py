"""Fine-tune DistilBERT on the persisted train split; select by validation loss."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
import json
import argparse
import math
import random
import time

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from experiment_common import (ROOT, ARTIFACTS, LABELS, LABEL2ID, identities, load_locked_data,
                               metrics, now, sha256, validate_baseline, write_csv, write_json)

BASE_MODEL = 'distilbert-base-uncased'
MODEL_DIR = ROOT / 'checkpoints' / 'takemeter-distilbert'
CACHE = ROOT / '.cache' / 'huggingface'
SEED = 42


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class CommentDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.encoded = tokenizer([r['text'] for r in rows], padding='max_length',
                                 truncation=True, max_length=max_length, return_tensors='pt')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {**{k: v[index] for k, v in self.encoded.items()},
                'labels': torch.tensor(LABEL2ID[self.rows[index]['final_label']]),
                'row_index': index}


def infer(model, loader, device):
    model.eval()
    probabilities = []
    loss_sum = 0.0
    seen = []
    with torch.inference_mode():
        for batch in loader:
            indices = batch.pop('row_index').tolist()
            targets = batch.pop('labels').to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            logits = model(**inputs).logits
            loss_sum += torch.nn.functional.cross_entropy(logits, targets, reduction='sum').item()
            probabilities.extend(logits.softmax(-1).cpu().tolist())
            seen.extend(indices)
    assert seen == list(range(len(loader.dataset)))
    return probabilities, loss_sum / len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='main', help='Use the recorded resolved revision for replay.')
    args = parser.parse_args()
    started = time.perf_counter()
    splits = load_locked_data()
    validate_baseline(splits)
    if (ARTIFACTS / 'training_config.json').exists() or MODEL_DIR.exists():
        raise FileExistsError('Refusing to overwrite an existing training run')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing CPU fallback')
    device = torch.device('cuda:0')
    seed_everything()
    environment = {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                   'cuda_available': True, 'gpu_name': torch.cuda.get_device_name(0),
                   'transformers': transformers.__version__, 'device': str(device)}
    print(json.dumps(environment), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=args.revision, cache_dir=CACHE, token=False)
    # Derive the length from train/validation only. Test text is never tokenized here.
    lengths = {}
    for name in ['train', 'validation']:
        lengths[name] = [len(tokenizer(r['text'], truncation=False)['input_ids']) for r in splits[name]]
    pooled = lengths['train'] + lengths['validation']
    max_length = min(512, max(128, 64 * math.ceil(float(np.percentile(pooled, 95)) / 64)))
    length_stats = {name: {'count': len(values), 'median': float(np.median(values)),
                           'p95': float(np.percentile(values, 95)), 'maximum': max(values),
                           'truncated_at_selected_length': sum(v > max_length for v in values)}
                    for name, values in lengths.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, revision=args.revision, cache_dir=CACHE, token=False, num_labels=4, id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID, attn_implementation='eager')
    revision = model.config._commit_hash
    # Pin the tokenizer to the same actual base-model revision used for weights.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=revision, cache_dir=CACHE, token=False)
    model.to(device)
    assert next(model.parameters()).device.type == 'cuda'
    config = {
        'created_at': now(), 'base_model': BASE_MODEL, 'resolved_revision': revision,
        'epochs': 3, 'learning_rate': 2e-5, 'batch_size': 16, 'max_sequence_length': max_length,
        'seed': SEED, 'optimizer': 'torch.optim.AdamW', 'weight_decay': 0.0,
        'adam_betas': [0.9, 0.999], 'adam_epsilon': 1e-8,
        'scheduler': 'linear decay to zero over all optimization steps', 'warmup_steps': 0,
        'gradient_clip_norm': 1.0, 'precision': 'float32', 'attention': 'eager',
        'selection_rule': 'lowest validation cross-entropy after an epoch; earliest epoch wins ties',
        'max_length_rule': 'train+validation token-length p95 rounded up to 64; bounded to [128,512]',
        'length_statistics': length_stats, 'environment': environment,
        'deterministic_algorithms': True,
        'reproducibility_note': 'Seeds and deterministic kernels set; exact cross-platform replay is not guaranteed.',
        'model_directory': 'checkpoints/takemeter-distilbert',
        **identities(splits)}
    write_json(ARTIFACTS / 'training_config.json', config)
    print(json.dumps({'max_sequence_length': max_length, 'length_statistics': length_stats,
                      'resolved_revision': revision}), flush=True)
    train_data = CommentDataset(splits['train'], tokenizer, max_length)
    validation_data = CommentDataset(splits['validation'], tokenizer, max_length)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, generator=generator, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=16, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0)
    steps = 3 * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1 - step / steps))
    best_loss, selected_epoch = float('inf'), None
    history, validation_predictions = [], []
    torch.cuda.synchronize()
    training_start = time.perf_counter()
    for epoch in range(1, 4):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = 0.0
        seen_ids = []
        for batch in train_loader:
            indices = batch.pop('row_index').tolist()
            seen_ids.extend(splits['train'][i]['comment_id'] for i in indices)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(indices)
        assert len(seen_ids) == 197 and set(seen_ids) == set(config['split_ids']['train'])
        probs, validation_loss = infer(model, validation_loader, device)
        preds = [LABELS[int(np.argmax(p))] for p in probs]
        val_metrics = metrics([r['final_label'] for r in splits['validation']], preds)
        for row, pred in zip(splits['validation'], preds):
            validation_predictions.append({'epoch': epoch, 'comment_id': row['comment_id'],
                                           'true_label': row['final_label'], 'predicted_label': pred})
        if validation_loss < best_loss:
            best_loss, selected_epoch = validation_loss, epoch
            model.save_pretrained(MODEL_DIR, safe_serialization=True)
            tokenizer.save_pretrained(MODEL_DIR)
        torch.cuda.synchronize()
        record = {'epoch': epoch, 'train_loss': loss_sum/197, 'validation_loss': validation_loss,
                  'validation_metrics': val_metrics, 'optimization_ids': seen_ids,
                  'epoch_runtime_seconds': time.perf_counter()-epoch_start}
        history.append(record)
        print(json.dumps({k: record[k] for k in ['epoch','train_loss','validation_loss','epoch_runtime_seconds']}), flush=True)
    torch.cuda.synchronize()
    write_csv(ARTIFACTS / 'validation_predictions.csv', validation_predictions)
    write_json(ARTIFACTS / 'training_results.json', {
        'completed_at': now(), 'selected_epoch': selected_epoch,
        'best_validation_loss': best_loss, 'history': history, 'optimization_steps': steps,
        'training_runtime_seconds': time.perf_counter()-training_start,
        'total_runtime_seconds_including_downloads': time.perf_counter()-started,
        'environment': environment, 'training_config_sha256': sha256(ARTIFACTS/'training_config.json'),
        'model_files_sha256': {p.name: sha256(p) for p in MODEL_DIR.iterdir() if p.is_file()},
        'test_forward_passes': 0})
    print(f'Training complete. Selected epoch {selected_epoch} using validation loss.', flush=True)


if __name__ == '__main__':
    main()
