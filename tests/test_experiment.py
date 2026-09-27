import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_common import LABELS, LABEL2ID, metrics, load_locked_data
from run_groq_zero_shot_baseline import run_baseline, parse_label, SUBSTITUTE_MODEL


class ExperimentTests(unittest.TestCase):
    def test_strict_parser(self):
        for label in LABELS:
            self.assertEqual(parse_label(' \n' + label + '\n'), label)
        for invalid in ['reaction.', '"reaction"', 'Reaction', 'reaction\nbare_claim', '', '{}']:
            self.assertIsNone(parse_label(invalid))

    def test_format_retry_uses_only_text_and_frozen_template(self):
        client = Mock()
        def response(raw):
            return SimpleNamespace(id='test-only', model=SUBSTITUTE_MODEL,
                                   choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])
        client.chat.completions.create.side_effect = [response('reaction.'), response('reaction')]
        journal = io.StringIO()
        predictions = run_baseline(client, [('synthetic_test_id', 'synthetic test content')],
                                   'Frozen template\n{comment_text}', journal)
        calls = client.chat.completions.create.call_args_list
        self.assertEqual(calls[0].kwargs['messages'],
                         [{'role': 'user', 'content': 'Frozen template\nsynthetic test content'}])
        self.assertEqual(calls[1].kwargs['messages'][0]['content'],
                         calls[0].kwargs['messages'][0]['content'] +
                         '\nFormat correction: output only one exact valid label name.')
        self.assertNotIn('true_label', predictions[0])
        self.assertEqual(predictions[0]['predicted_label'], 'reaction')
        self.assertEqual(len(journal.getvalue().splitlines()), 2)

    def test_metrics_fixed_label_order(self):
        result = metrics(LABELS, [LABELS[0]] * 4)
        self.assertEqual(result['accuracy'], 0.25)
        self.assertEqual(result['confusion_matrix'], [[1, 0, 0, 0]] * 4)
        self.assertAlmostEqual(result['macro_f1'], 0.1)
        self.assertEqual(LABEL2ID, dict(zip(LABELS, range(4))))

    def test_real_lock_disjoint_complete(self):
        splits = load_locked_data()
        ids = [{r['comment_id'] for r in splits[name]} for name in ['train', 'validation', 'test']]
        self.assertEqual([len(group) for group in ids], [197, 42, 43])
        self.assertEqual(len(set.union(*ids)), 282)
        self.assertFalse(ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2])


if __name__ == '__main__':
    unittest.main()
