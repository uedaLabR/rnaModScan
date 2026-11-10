import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ARROW_AVAILABLE = importlib.util.find_spec('pyarrow') is not None
TF_AVAILABLE = importlib.util.find_spec('tensorflow') is not None


@unittest.skipUnless(ARROW_AVAILABLE, 'pyarrow not installed')
class ParquetTrainingTest(unittest.TestCase):
    def prepare(self, root):
        from tests.site_bag_fixture import prepare_training_bags
        return prepare_training_bags(root)

    def test_index_random_parquet_ranges_and_three_model_inputs(self):
        from training.cross_validation import build_cross_validation
        from utils.features.site_dataset import SiteBagBatches as SiteBatches
        from utils.nn.inputs import CHANNEL_NAMES
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.prepare(root)
            build_cross_validation({'adenine': index}, root / 'cv')
            dataset = SiteBatches(index, root / 'cv/adenine/fold_0/train/sites.tsv', batch_sites=3)
            inputs, labels = next(dataset.batches())
            self.assertEqual(set(inputs), {'sequence_tokens', 'read_features', 'read_mask', 'sequence_repeat_mask'})
            self.assertEqual(inputs['read_features'].shape, (3, 300, 19, 18))
            self.assertTrue(inputs['read_mask'][:, :2].all())
            self.assertFalse(inputs['read_mask'][:, 2:].any())
            self.assertEqual(str(inputs['sequence_tokens'].dtype), 'int32')
            self.assertEqual(str(inputs['read_features'].dtype), 'float32')
            self.assertEqual(labels.shape, (3,))
            row = dataset.rows[0]
            self.assertAlmostEqual(inputs['read_features'][0, 0, 9, 0],
                                   (40 if row['label'] == '2' else 240) / 255, places=6)
            base_start = CHANNEL_NAMES.index('base_match')
            self.assertEqual(inputs['read_features'][0, 0, 9, base_start:base_start+6].tolist(),
                             [1, 0, 0, 0, 0, 0])
            for name in inputs:
                self.assertNotIn(name, ('label', 'sample_id', 'group_id', 'condition'))

    def test_audit_annotations_and_site_statistics_do_not_change_inputs(self):
        import numpy as np
        from utils.features.site_dataset import SiteBagBatches as SiteBatches
        with tempfile.TemporaryDirectory() as tmp:
            index = self.prepare(Path(tmp))
            baseline = SiteBatches(index, None, batch_sites=3)
            # These objects cannot provide annotations. Training must not inspect
            # them, or parse auxiliary site statistics from the raw index.
            audited = SiteBatches(index, None, batch_sites=3)
            for row in audited.rows:
                row['af'] = 'audit-only'
                row['af_m6A'] = 'audit-only'
                row['af_inosine'] = 'audit-only'
                row['eligible_depth'] = 'audit-only'
            expected, expected_labels = next(baseline.batches())
            actual, actual_labels = next(audited.batches())
            self.assertEqual(set(actual), {'sequence_tokens', 'read_features', 'read_mask', 'sequence_repeat_mask'})
            for name in expected:
                np.testing.assert_array_equal(actual[name], expected[name])
            np.testing.assert_array_equal(actual_labels, expected_labels)

    @unittest.skipUnless(TF_AVAILABLE, 'TensorFlow not installed')
    def test_one_fold_training_from_parquet_to_saved_keras_and_test_predictions(self):
        from training.cross_validation import build_cross_validation
        from training.fit import train_fold
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.prepare(root)
            build_cross_validation({'adenine': index}, root / 'cv')
            report = train_fold(index, root / 'cv/adenine/fold_0', root / 'model', 'adenine',
                                epochs=1, batch_sites=8)
            self.assertEqual(report['status'], 'complete')
            self.assertTrue((root / 'model/train/best.keras').is_file())
            self.assertTrue((root / 'model/validation/predictions.tsv').is_file())
            self.assertTrue((root / 'model/test/predictions.tsv').is_file())
            self.assertTrue((root / 'model/validation/metrics.json').is_file())
            self.assertTrue((root / 'model/test/metrics.json').is_file())
            for split in ('validation', 'test'):
                directory = root / 'model' / split
                self.assertTrue((directory / 'metrics_summary.tsv').is_file())
                self.assertTrue((directory / 'confusion_matrix.tsv').is_file())
                with (directory / 'metrics_summary.tsv').open() as handle:
                    summary = list(csv.DictReader(handle, delimiter='\t'))
                metrics = json.loads((directory / 'metrics.json').read_text())
                self.assertEqual(int(summary[0]['sites']), metrics['sites'])
                self.assertAlmostEqual(float(summary[0]['accuracy']), metrics['accuracy'])
            progress = json.loads((root / 'model/train/progress.json').read_text())
            self.assertEqual(progress['status'], 'complete')
            self.assertEqual(progress['completed_epochs'], 1)
            self.assertFalse(any(key.startswith('test_') for key in progress['metrics']))
            for kind in ('png', 'pdf', 'svg'):
                self.assertTrue((root / 'model/train' / ('learning_curves.' + kind)).is_file())
            self.assertFalse((root / 'model/metrics.json').exists())
            self.assertFalse((root / 'model/INCOMPLETE.txt').exists())
            with (root / 'model/test/predictions.tsv').open() as handle:
                predictions = list(csv.DictReader(handle, delimiter='\t'))
            with (root / 'cv/adenine/fold_0/test/sites.tsv').open() as handle:
                test_ids = {r['row_id'] for r in csv.DictReader(handle, delimiter='\t')}
            self.assertEqual({r['row_id'] for r in predictions}, test_ids)

    @unittest.skipUnless(TF_AVAILABLE, 'TensorFlow not installed')
    def test_fit_exception_stops_heartbeat_and_keeps_failure_audit(self):
        from training.cross_validation import build_cross_validation
        from training.fit import train_fold
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.prepare(root)
            build_cross_validation({'adenine': index}, root / 'cv')
            callbacks = []

            def fail_fit(*args, **kwargs):
                progress = kwargs['callbacks'][-1]
                callbacks.append(progress)
                progress.set_params(dict(epochs=1, steps=1))
                progress.on_train_begin()
                raise RuntimeError('synthetic fit failure')

            with patch('training.fit.compile_site_model') as compile_model:
                compile_model.return_value.fit.side_effect = fail_fit
                with self.assertRaisesRegex(RuntimeError, 'synthetic fit failure'):
                    train_fold(index, root / 'cv/adenine/fold_0', root / 'model', 'adenine', epochs=1)
            saved = json.loads((root / 'model/train/progress.json').read_text())
            self.assertEqual((saved['status'], saved['phase']), ('failed', 'failed'))
            self.assertIn('synthetic fit failure', saved['error'])
            self.assertFalse(callbacks[0]._heartbeat_thread.is_alive())
            self.assertTrue((root / 'model/FAILED.txt').is_file())
            self.assertTrue((root / 'model/INCOMPLETE.txt').is_file())


if __name__ == '__main__':
    unittest.main()
