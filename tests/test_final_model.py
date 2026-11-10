import csv
import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

TF_AVAILABLE = importlib.util.find_spec('tensorflow') is not None
ARROW_AVAILABLE = importlib.util.find_spec('pyarrow') is not None


@unittest.skipUnless(TF_AVAILABLE, 'TensorFlow not installed')
class FinalModelTest(unittest.TestCase):
    def cv_records(self, directory, index, epochs, position_annotation=None):
        from utils.nn.keras_site import MODEL_INPUT_SCHEMA, ModelConfig
        from utils.references.catalog import sha256
        for fold, best in enumerate(epochs):
            path = directory / f'fold_{fold}'
            path.mkdir(parents=True)
            (path / 'train').mkdir()
            (path / 'test').mkdir()
            # Synthetic records test refit wiring, not CV predictive performance.
            (path / 'run.json').write_text(json.dumps(dict(status='complete', target='adenine',
                index_sha256=sha256(index), model=asdict(ModelConfig(target='adenine', max_reads=300)), feature_set=MODEL_INPUT_SCHEMA)))
            with (path / 'train/history.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['epoch', 'val_loss'])
                writer.writerows((e, .5 if e == best-1 else .8) for e in range(5))
            (path / 'test/metrics.json').write_text(json.dumps(dict(test_accuracy=fold / 5)))

    def test_final_epoch_choice_uses_validation_history_not_test_metrics(self):
        from training.final_model import cv_training_choice
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / 'sites.tsv'
            index.write_text('synthetic index for provenance\n')
            self.cv_records(root / 'cv', index, [2, 4, 3, 5, 1])
            epochs, config, sources = cv_training_choice(root / 'cv', index, 'adenine')
            self.assertEqual(epochs, 3)
            self.assertEqual([s['best_epoch'] for s in sources], [2, 4, 3, 5, 1])
            index.write_text('changed index\n')
            with self.assertRaisesRegex(ValueError, 'current labeled index'):
                cv_training_choice(root / 'cv', index, 'adenine')

    @unittest.skipUnless(ARROW_AVAILABLE, 'pyarrow not installed')
    def test_refit_all_labels_exports_loadable_deployment_bundle(self):
        from tests.site_bag_fixture import prepare_training_bags
        from training.final_model import train_final_model
        from prediction.predict import _load_model
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = prepare_training_bags(root)
            self.cv_records(root / 'cv', index, [1] * 5)
            result = train_final_model(index, root / 'cv', root / 'final', 'adenine', epochs=1,
                                        batch_sites=8)
            self.assertEqual(result['training_rows'], 48)
            self.assertEqual(result['role'], 'final_all_labeled_sites')
            self.assertEqual(result['epoch_policy'], 'explicit setting')
            model, vocabulary, saved = _load_model(root / 'final')
            self.assertEqual(tuple(model.output_shape), (None, 3))
            self.assertEqual(saved['training_rows'], 48)
            with (root / 'final/training_ids.tsv').open() as handle:
                rows = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(len(rows), 48)
            self.assertEqual({r['sample_id'] for r in rows}, {'HeLa_WT', 'HeLa_IVT', 'HEK293T_WT', 'HEK293T_IVT'})
            self.assertFalse((root / 'final/metrics.json').exists())  # No held-out accuracy claim after full refit.
            self.assertFalse((root / 'final/repeat_vocabulary.json').exists())
            self.assertEqual(set(t.name for t in model.inputs),
                             {'sequence_tokens', 'read_features', 'read_mask', 'sequence_repeat_mask'})
