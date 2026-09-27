"""Run the explicitly authorized substitute using the unchanged saved prompt.

No split generation, automatic model fallback, prompt tuning, or partial-set
metrics. Every request/response is journaled; unresolved failure blocks training.
"""
import argparse
import hashlib
import json
import os
import re
import time

from dotenv import load_dotenv
from groq import Groq, RateLimitError
from experiment_common import (ARTIFACTS, ROOT, LABELS, identities, load_locked_data,
                               metrics, now, plot_confusion, write_csv, write_json)

PRESCRIBED_MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct'
SUBSTITUTE_MODEL = 'openai/gpt-oss-20b'


def sanitize(value):
    text = str(value)
    key = os.getenv('GROQ_API_KEY')
    if key:
        text = text.replace(key, '[REDACTED]')
    return re.sub(r'gsk_[A-Za-z0-9_-]+', '[REDACTED]', text)


def parse_label(content):
    value = content.strip()
    return value if value in LABELS else None


def build_prompt(template, text, format_retry=False):
    prompt = template.format(comment_text=text)
    if format_retry:
        prompt += '\nFormat correction: output only one exact valid label name.'
    return prompt


def run_baseline(client, test_inputs, template, journal):
    # Only (ID, text) enter this function. Truth is joined after all inference.
    predictions = []
    for comment_id, text in test_inputs:
        predicted = None
        for format_attempt in range(2):
            prompt = build_prompt(template, text, bool(format_attempt))
            for rate_attempt in range(4):
                record = {'timestamp': now(), 'comment_id': comment_id,
                          'model': SUBSTITUTE_MODEL, 'temperature': 0,
                          'format_attempt': format_attempt + 1,
                          'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
                try:
                    response = client.chat.completions.create(
                        model=SUBSTITUTE_MODEL, temperature=0,
                        messages=[{'role': 'user', 'content': prompt}])
                except Exception as error:
                    record.update(error_type=type(error).__name__,
                                  http_status=getattr(error, 'status_code', None),
                                  error=sanitize(error))
                    journal.write(json.dumps(record) + '\n')
                    journal.flush()
                    if isinstance(error, RateLimitError) and rate_attempt < 3:
                        time.sleep(20)
                        continue
                    raise
                raw = response.choices[0].message.content or ''
                predicted = parse_label(raw)
                record.update(response_id=response.id, returned_model=response.model,
                              raw_response=sanitize(raw), predicted_label=predicted,
                              parse_status='valid' if predicted else 'parse_failure')
                journal.write(json.dumps(record) + '\n')
                journal.flush()
                break
            if predicted:
                break
        predictions.append({'comment_id': comment_id, 'predicted_label': predicted or '',
                            'parse_status': 'valid' if predicted else 'parse_failure',
                            'raw_model_response': sanitize(raw),
                            'format_attempts': format_attempt + 1})
        print(f'Classified {len(predictions)}/{len(test_inputs)}', flush=True)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=[SUBSTITUTE_MODEL], required=True)
    parser.add_argument('--confirm-authorized-substitute', action='store_true', required=True)
    args = parser.parse_args()
    splits = load_locked_data()
    failure = json.loads((ARTIFACTS / 'prescribed_baseline_failure.json').read_text())
    if failure['requested_model'] != PRESCRIBED_MODEL or failure['http_status'] != 404:
        raise ValueError('Prescribed-model failure evidence missing')
    for name in ['baseline_predictions.csv', 'baseline_results.json', 'baseline_requests.jsonl',
                 'substitute_baseline_failure.json']:
        if (ARTIFACTS / name).exists():
            raise FileExistsError(f'Refusing to overwrite {name}')
    template = (ARTIFACTS / 'baseline_prompt_template.txt').read_text(encoding='utf-8')
    load_dotenv(ROOT / '.env')
    client = Groq(api_key=os.environ['GROQ_API_KEY'], max_retries=0, timeout=90)
    inputs = [(r['comment_id'], r['text']) for r in splits['test']]
    start = time.perf_counter()
    try:
        with (ARTIFACTS / 'baseline_requests.jsonl').open('x', encoding='utf-8') as journal:
            predictions = run_baseline(client, inputs, template, journal)
    except Exception as error:
        write_json(ARTIFACTS / 'substitute_baseline_failure.json', {
            'timestamp': now(), 'model': args.model, 'error_type': type(error).__name__,
            'http_status': getattr(error, 'status_code', None), 'error': sanitize(error)})
        raise SystemExit('Substitute baseline failed; see sanitized failure artifact. Training must stop.')
    by_id = {r['comment_id']: r for r in splits['test']}
    for prediction in predictions:
        prediction['true_label'] = by_id[prediction['comment_id']]['final_label']
    write_csv(ARTIFACTS / 'baseline_predictions.csv', predictions)
    if any(p['parse_status'] != 'valid' for p in predictions):
        raise SystemExit('Unresolved parse failures; training must stop.')
    result = metrics([p['true_label'] for p in predictions], [p['predicted_label'] for p in predictions])
    result.update(model=args.model, temperature=0, parse_failures=0, timestamp=now(),
                  runtime_seconds=time.perf_counter()-start,
                  prescribed_model=PRESCRIBED_MODEL,
                  substitution={'reason': 'Prescribed model returned HTTP 404 model_not_found',
                                'student_authorized': True,
                                'authorization_source': 'Explicit student instruction for this run',
                                'failure_evidence': 'artifacts/prescribed_baseline_failure.json'},
                  **identities(splits))
    write_json(ARTIFACTS / 'baseline_results.json', result)
    plot_confusion(result, ARTIFACTS / 'baseline_confusion_matrix.png', 'Groq zero-shot baseline (43 locked test comments)')
    print(json.dumps({k: result[k] for k in ['model', 'count', 'accuracy', 'macro_f1', 'parse_failures']}))


if __name__ == '__main__':
    main()
