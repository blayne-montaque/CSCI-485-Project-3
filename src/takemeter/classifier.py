"""Offline inference for the saved course classifier; no API credentials used."""
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ('reasoned_explanation', 'actionable_advice', 'bare_claim', 'reaction')
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / 'checkpoints' / 'takemeter-distilbert'


def choose_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def load_classifier(path=DEFAULT_MODEL, device=None):
    path = Path(path)
    if not (path / 'config.json').is_file() or not (path / 'model.safetensors').is_file():
        raise FileNotFoundError('Saved DistilBERT checkpoint is missing.')
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True, attn_implementation='eager')
    if model.config.label2id != dict(zip(LABELS, range(4))):
        raise ValueError('Checkpoint label mapping does not match TakeMeter.')
    if model.config.id2label != dict(enumerate(LABELS)):
        raise ValueError('Checkpoint ID mapping does not match TakeMeter.')
    target = device or choose_device()
    model.to(target).eval()
    return tokenizer, model, target


def classify_text(text, bundle, max_length=256):
    if not text.strip():
        raise ValueError('Enter a comment before classifying.')
    tokenizer, model, device = bundle
    full_length = len(tokenizer(text, truncation=False)['input_ids'])
    encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
    with torch.inference_mode():
        logits = model(**{key:value.to(device) for key,value in encoded.items()}).logits[0]
        probabilities = logits.softmax(-1).cpu().tolist()
    if not all(0 <= p <= 1 for p in probabilities) or abs(sum(probabilities)-1) > 1e-5:
        raise RuntimeError('Model returned invalid probabilities.')
    index = max(range(4), key=lambda i: probabilities[i])
    return {'predicted_label': LABELS[index], 'confidence': probabilities[index],
            'probabilities': dict(zip(LABELS, probabilities)), 'device':device,
            'input_tokens':full_length, 'truncated':full_length>max_length,
            'max_sequence_length':max_length}
