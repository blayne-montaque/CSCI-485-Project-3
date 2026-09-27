import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
from analyze_calibration import calibration, build as calibration_build
from analyze_errors import build as errors_build
from takemeter.classifier import choose_device, classify_text, load_classifier


class CalibrationTests(unittest.TestCase):
    def test_perfect_predictions_and_empty_bins(self):
        result=calibration([[1,0,0,0],[0,1,0,0]],[0,1])
        self.assertEqual(result['ece'],0)
        self.assertEqual(result['multiclass_brier_score'],0)
        self.assertEqual(result['bins'][-1]['count'],2)
        self.assertIsNone(result['bins'][0]['accuracy'])

    def test_bin_boundary_and_known_error(self):
        result=calibration([[.3,.25,.25,.2]],[1])
        self.assertEqual(result['bins'][3]['count'],1)
        self.assertAlmostEqual(result['ece'],.3)
        self.assertAlmostEqual(result['multiclass_brier_score'],.755)

    def test_reject_invalid_probabilities(self):
        with self.assertRaises(ValueError): calibration([[.9,.9,0,0]],[0])
        with self.assertRaises(ValueError): calibration([[float('nan'),0,0,1]],[0])

    def test_saved_calibration_reproduces(self):
        saved=json.loads((ROOT/'artifacts/calibration_results.json').read_text())
        self.assertEqual(saved,calibration_build())
        self.assertEqual(sum(b['count'] for b in saved['bins']),43)


class ErrorAnalysisTests(unittest.TestCase):
    def test_saved_analysis_reproduces(self):
        result,rows=errors_build()
        self.assertEqual(result,json.loads((ROOT/'artifacts/error_analysis.json').read_text(encoding='utf-8')))
        self.assertEqual(len(rows),43)
        self.assertEqual(sum(result['paired_correctness_counts'].values()),43)
        self.assertEqual(result['prediction_agreement']+result['prediction_disagreement'],43)
        self.assertEqual(sum(p['count'] for p in result['confusion_pairs']),29)
        self.assertTrue(all(not e['correct'] for e in result['representative_errors']))


class ClassifierTests(unittest.TestCase):
    def test_empty_input_prevents_inference(self):
        with self.assertRaises(ValueError): classify_text('  ',None)

    def test_missing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError): load_classifier(directory)

    def test_cpu_fallback(self):
        with patch('torch.cuda.is_available',return_value=False):
            self.assertEqual(choose_device(),'cpu')


class StreamlitTests(unittest.TestCase):
    def test_empty_submit_and_missing_checkpoint(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ',{'TAKEMETER_MODEL_DIR':directory}):
            app=AppTest.from_file(str(ROOT/'classifier_app.py'),default_timeout=30).run()
            self.assertFalse(app.exception)
            app.button[0].click().run()
            self.assertTrue(any('Enter a comment' in str(w.value) for w in app.warning))
            app.text_area[0].set_value('A newly written smoke test comment.').run()
            app.button[0].click().run()
            self.assertTrue(app.error)
            self.assertFalse(app.exception)

    def test_real_local_classification(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(ROOT/'classifier_app.py'),default_timeout=60).run()
        self.assertFalse(app.exception)
        app.text_area[0].set_value('Try printing a small calibration cube before the next large part.').run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.metric),1)
        self.assertEqual(len(app.dataframe[0].value),4)


if __name__=='__main__': unittest.main()
