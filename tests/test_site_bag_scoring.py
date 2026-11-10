"""Inference consumes only prepared common Parquet, including ragged/zero-read shards."""
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq

from tests.test_site_bag_prediction_inputs import SiteBagPredictionInputsTest
from prediction.site_bag_inputs import prepare_site_bag_prediction_inputs
from prediction.scoring import score_prepared_inputs


class SiteBagScoringTest(unittest.TestCase):

    def test_old_prepared_bundle_is_rejected_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'run.json').write_text(json.dumps(dict(status='complete', schema='prediction_inputs_v6')))
            with patch('prediction.predict._load_model', side_effect=AssertionError('Do not load old inputs')):
                with self.assertRaisesRegex(ValueError, 'prepared NN input bundle'):
                    score_prepared_inputs(root, root / 'model', root / 'scores')

    def test_multiple_shards_score_without_original_sources_or_padding_on_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, model_dir, source, _, _ = SiteBagPredictionInputsTest().fixture(root)
            prepared = root / 'prepared'
            run = prepare_site_bag_prediction_inputs(index, model_dir, prepared)
            model_run = json.loads((model_dir / 'run.json').read_text())
            shutil.rmtree(source)
            shutil.rmtree(index.parent)
            masks = []
            class Model:
                def __call__(self, inputs, training=False):
                    self_shape = inputs['read_features'].shape
                    if self_shape[1:] != (300, 19, 18):
                        raise AssertionError(self_shape)
                    if (inputs['sequence_repeat_mask'].shape != (self_shape[0], 19)
                            or inputs['sequence_repeat_mask'].dtype != bool
                            or not inputs['sequence_repeat_mask'].any()):
                        raise AssertionError('Saved binary repeat masks must reach inference')
                    masks.extend(inputs['read_mask'].sum(axis=1).tolist())
                    scores = np.tile(np.array([.2, .7, .1]), (self_shape[0], 1))
                    return types.SimpleNamespace(numpy=lambda: scores)
            fake_tf = types.SimpleNamespace(keras=types.SimpleNamespace(backend=types.SimpleNamespace(clear_session=lambda: None)))
            with patch('prediction.predict._load_model', return_value=(Model(), {}, model_run)), \
                    patch('prediction.scoring.configure_inference', return_value=dict(device='cpu')), \
                    patch.dict('sys.modules', {'tensorflow': fake_tf}):
                report = score_prepared_inputs(prepared, model_dir, root / 'scores', batch_sites=2)
            self.assertEqual(sum(part['rows'] for part in report['partitions']), 2)
            self.assertTrue(all(0 < count <= 300 for count in masks))
            values = []
            for part in report['partitions']:
                values.extend(pq.ParquetFile(root / 'scores' / part['path']).read().to_pylist())
            self.assertEqual(len(values), 2)
            self.assertTrue(all(row['predicted_class'] == 'inosine' for row in values))
            self.assertFalse((root / 'scores/INCOMPLETE.txt').exists())
