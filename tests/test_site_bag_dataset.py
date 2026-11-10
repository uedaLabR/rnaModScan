"""Shared ragged site Parquet supports fitting and unlabeled inference directly."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from utils.adenine import CLASSES
from utils.features.site_dataset import SiteBagBatches
from utils.nn.inputs import CHANNEL_NAMES


class SiteBagDatasetTest(unittest.TestCase):
    def prepare(self, root, *, depths=(1, 10, 300), labels=(0, 1, 2), stored_labels=None, missing_mask=False):
        source = root / 'source'
        directory = source / 'sites/chrom=chr1/strand=plus'
        directory.mkdir(parents=True)
        path = directory / 'part-000000.parquet'
        schema = pa.schema([
            ('row_id', pa.string()),
            ('sequence_tokens', pa.list_(pa.int32(), 17)),
            ('sequence_repeat_mask', pa.list_(pa.bool_(), 19)),
            ('read_features', pa.list_(pa.list_(pa.list_(pa.float32(), len(CHANNEL_NAMES)), 19))),
            ('training_class', pa.string()),
            ('training_label_m6A', pa.bool_()), ('training_label_inosine', pa.bool_()),
            ('training_label_false_positive', pa.bool_()),
        ], metadata={b'schema_version': b'site_bags_parquet_v3'})
        records, rows = [], []
        for index, (depth, label) in enumerate(zip(depths, labels)):
            row_id = f'sample|chr1:{100 + index * 100}:+:adenine'
            stored = (stored_labels or labels)[index]
            name = CLASSES[stored] if stored is not None else None
            values = np.zeros((depth, 19, len(CHANNEL_NAMES)), np.float32)
            if index != 0:
                values[:] = index / 10
            record = dict(row_id=row_id, sequence_tokens=[index] * 17,
                          read_features=values.tolist(), training_class=name,
                          sequence_repeat_mask=[position % 3 == 0 for position in range(19)])
            if missing_mask:
                record['sequence_repeat_mask'][0] = None
            record.update({'training_label_' + value: value == name if name is not None else None
                           for value in CLASSES})
            records.append(record)
            rows.append(dict(row_id=row_id, sample_id='sample', chrom='chr1', start=100 + index * 100,
                             end=101 + index * 100, strand='+', modification='adenine',
                             label='' if label is None else label, reference_sequence='A' * 19,
                             features_dir=str(source), retained_depth=depth, eligible_depth=depth * 4,
                             site_parquet=str(path.relative_to(source)), site_row_group=index // 2,
                             site_row_offset=index % 2, group_id='group_' + str(index)))
        pq.write_table(pa.Table.from_pylist(records, schema=schema), path, row_group_size=2)
        source_run = dict(status='complete', schema='site_bags_parquet_v3', sample_id='sample',
                          max_reads=300, sampling_seed=42,
                          parquet_files=dict(sites=[dict(path=str(path.relative_to(source / 'sites')),
                                                         rows=len(rows), size_bytes=path.stat().st_size)]))
        (source / 'run.json').write_text(json.dumps(source_run))
        index_path = root / 'index.tsv'
        with index_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, list(rows[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
        return index_path, source, rows

    def test_ragged_rows_are_padded_only_when_building_nn_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            index, _, _ = self.prepare(Path(directory))
            data = SiteBagBatches(index, None, batch_sites=3)
            inputs, labels = next(data.batches())
            self.assertEqual(set(inputs), {'sequence_tokens', 'read_features', 'read_mask', 'sequence_repeat_mask'})
            self.assertEqual(inputs['read_features'].shape, (3, 300, 19, 18))
            self.assertEqual(inputs['read_features'].dtype, np.float32)
            np.testing.assert_array_equal(inputs['read_mask'].sum(axis=1), [1, 10, 300])
            np.testing.assert_array_equal(labels, [0, 1, 2])
            # An all-zero real read is still present; only mask defines padding.
            self.assertTrue(inputs['read_mask'][0, 0])
            self.assertFalse(inputs['read_features'][0].any())
            np.testing.assert_array_equal(inputs['read_features'][1, 10:], 0)
            np.testing.assert_array_equal(inputs['sequence_tokens'][:, 0], [0, 1, 2])
            self.assertEqual(len(data.cache), 2)

    def test_missing_repeat_annotation_is_not_replaced_with_zero_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            index, _, _ = self.prepare(Path(directory), missing_mask=True)
            data = SiteBagBatches(index, None)
            with self.assertRaisesRegex(ValueError, 'complete sequence repeat mask'):
                next(data.batches())

    def test_split_metadata_and_original_depth_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, _, rows = self.prepare(root)
            split = root / 'split.tsv'
            with split.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, list(rows[0]), delimiter='\t')
                writer.writeheader()
                writer.writerows(rows[1:])
            data = SiteBagBatches(index, split, batch_sites=2)
            self.assertEqual([row['group_id'] for row in data.rows], ['group_1', 'group_2'])
            self.assertEqual(data.rows[-1]['eligible_depth'], '1200')
            self.assertEqual(data.rows[-1]['retained_depth'], '300')
            np.testing.assert_array_equal(next(data.batches())[1], [1, 2])

    def test_null_training_labels_are_valid_for_inference_and_rejected_for_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            index, _, _ = self.prepare(Path(directory), depths=(1, 0), labels=(None, None))
            data = SiteBagBatches(index, None, require_labels=False)
            inputs, labels = next(data.batches())
            self.assertEqual(inputs['read_features'].shape, (1, 300, 19, 18))
            np.testing.assert_array_equal(labels, [-1])
            self.assertEqual(inputs['read_mask'].sum(), 1)
            with self.assertRaisesRegex(ValueError, 'unlabeled'):
                SiteBagBatches(index, None)

    def test_index_cannot_assign_a_class_to_an_unlabeled_inference_bag(self):
        with tempfile.TemporaryDirectory() as directory:
            index, _, _ = self.prepare(Path(directory), depths=(1,), labels=(0,), stored_labels=(None,))
            data = SiteBagBatches(index, None)
            with self.assertRaisesRegex(ValueError, 'training flags'):
                next(data.batches())

    def test_wrong_site_offset_and_sampling_policy_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            index, _, _ = self.prepare(Path(directory))
            data = SiteBagBatches(index, None)
            data.rows[0]['site_row_offset'] = '1'
            with self.assertRaisesRegex(ValueError, 'site ID'):
                next(data.batches())
            with self.assertRaisesRegex(ValueError, 'sampling policy'):
                SiteBagBatches(index, None, sampling_seed=7)

    def test_old_28_channel_site_bags_require_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            index, source, _ = self.prepare(Path(directory))
            run_path = source / 'run.json'
            run = json.loads(run_path.read_text())
            run['schema'] = 'site_bags_parquet_v1'
            run_path.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, 'regenerate old site bags'):
                SiteBagBatches(index, None)

    def test_truncated_file_and_retained_depth_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            index, _, _ = self.prepare(Path(directory))
            data = SiteBagBatches(index, None)
            data.rows[0]['retained_depth'] = '2'
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                next(data.batches())
            data = SiteBagBatches(index, None)
            path = Path(data.rows[0]['site_parquet'])
            path.write_bytes(path.read_bytes()[:-10])
            with self.assertRaisesRegex(ValueError, 'file changed'):
                next(data.batches())

    def test_training_shuffle_is_reproducible_grouped_and_label_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_index, _, _ = self.prepare(root / 'one', depths=(1,) * 6, labels=(0, 1, 2, 0, 1, 2))
            second_index, _, _ = self.prepare(root / 'two', depths=(1,) * 6, labels=(2, 0, 1, 2, 0, 1))
            first = SiteBagBatches(first_index, None, batch_sites=1)
            second = SiteBagBatches(second_index, None, batch_sites=1)
            def order(data, seed):
                return [int(inputs['sequence_tokens'][0, 0])
                        for inputs, _ in data.batches(shuffle=True, rng=np.random.default_rng(seed))]
            shuffled = order(first, 3)
            self.assertEqual(shuffled, order(first, 3))
            self.assertEqual(shuffled, order(second, 3))
            self.assertEqual(sorted(shuffled), list(range(6)))
            for start in range(0, len(shuffled), 2):
                self.assertEqual(shuffled[start] // 2, shuffled[start + 1] // 2)
            self.assertNotEqual(shuffled, order(first, 4))
            unshuffled = [int(inputs['sequence_tokens'][0, 0]) for inputs, _ in first.batches()]
            self.assertEqual(unshuffled, list(range(6)))

    def test_ragged_bags_feed_the_fixed_shape_tensorflow_model(self):
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest('TensorFlow unavailable')
        from utils.nn.keras_site import ModelConfig, build_site_model, compile_site_model
        with tempfile.TemporaryDirectory() as directory:
            index, _, _ = self.prepare(Path(directory), depths=(1, 10, 299))
            data = SiteBagBatches(index, None, batch_sites=3)
            model = compile_site_model(build_site_model(ModelConfig(target='adenine', max_reads=300)))
            history = model.fit(data.tensorflow_dataset(), epochs=1, verbose=0)
            self.assertTrue(np.isfinite(history.history['loss'][0]))
            self.assertEqual({tensor.name.split(':')[0] for tensor in model.inputs}, {'sequence_tokens', 'read_features', 'read_mask', 'sequence_repeat_mask'})


if __name__ == '__main__':
    unittest.main()
