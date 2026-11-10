"""Cell-only fitting on synthetic Parquet; no biological accuracy is tested."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) for name in ('tensorflow', 'pyarrow', 'pysam'))


def read_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def write_rows(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


class CellEpochChoiceTest(unittest.TestCase):
    def test_best_epoch_uses_finite_validation_loss(self):
        from training.cell_model import best_validation_epoch
        with tempfile.TemporaryDirectory() as temporary:
            history = Path(temporary) / 'history.csv'
            history.write_text('epoch,loss,val_loss\n0,0.01,0.7\n1,0.9,0.1\n2,0.001,0.4\n')
            # Training loss favours epoch 3; validation must select epoch 2.
            self.assertEqual(best_validation_epoch(history), 2)
            for contents in ('epoch,loss,val_loss\n', 'epoch,loss,val_loss\n0,0.1,nan\n'):
                with self.subTest(contents=contents):
                    history.write_text(contents)
                    with self.assertRaises(ValueError):
                        best_validation_epoch(history)


@unittest.skipUnless(AVAILABLE, 'TensorFlow/Arrow/pysam required')
class CellModelTest(unittest.TestCase):
    def prepare(self, root):
        from tests.site_bag_fixture import prepare_training_bags
        from training.cell_datasets import build_cell_datasets
        index = prepare_training_bags(root, high_depth=False)
        samples = [dict(sample_id=cell + '_' + condition, cell=cell,
                        reference_cell=cell, condition=condition)
                   for cell in ('HEK293T', 'HeLa') for condition in ('WT', 'IVT')]
        datasets = root / 'cell_datasets'
        build_cell_datasets({'adenine': index}, samples, datasets)
        return index, datasets, samples

    def test_hek_only_selection_refit_and_loadable_bundle_reuse_raw_features(self):
        from training.cell_model import train_cell_model
        from prediction.model_bundle import load_model_bundle
        from prediction.predict import _load_model
        from utils.features.site_dataset import SiteBagBatches as SiteBatches
        from utils.references.catalog import sha256
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index, datasets, samples = self.prepare(root)
            expected_ids = {row['row_id'] for row in read_rows(index)
                            if row['sample_id'] in ('HEK293T_WT', 'HEK293T_IVT')}
            raw_hashes = {path: sha256(path) for path in (root / 'features').rglob('*.parquet')}
            passed_to_tensorflow = []
            original = SiteBatches.tensorflow_dataset

            def observe(dataset, *args, **kwargs):
                passed_to_tensorflow.append({row['row_id'] for row in dataset.rows})
                self.assertTrue(all(row['sample_id'] in ('HEK293T_WT', 'HEK293T_IVT')
                                    for row in dataset.rows))
                return original(dataset, *args, **kwargs)

            out = root / 'cell_model'
            with patch.object(SiteBatches, 'tensorflow_dataset', observe):
                report = train_cell_model(index, datasets, samples, out, epochs=1,
                                          batch_sites=8, patience=0)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['training_cell'], 'HEK293T')
            self.assertEqual(report['best_epoch'], 1)
            self.assertEqual(set(report['training_scope']['sample_ids']),
                             {'HEK293T_WT', 'HEK293T_IVT'})
            self.assertIn(expected_ids, passed_to_tensorflow)
            self.assertEqual(len(expected_ids), 24)
            for relative in ('train/best.keras', 'train/history.csv', 'validation/predictions.tsv',
                             'validation/metrics.json', 'final_model/model.keras'):
                self.assertTrue((out / relative).is_file(), relative)
            self.assertFalse(list(out.rglob('test')))
            self.assertFalse((out / 'final_model/metrics.json').exists())
            self.assertFalse((out / 'INCOMPLETE.txt').exists())
            self.assertFalse(list(out.rglob('*.parquet')))
            self.assertEqual(raw_hashes, {path: sha256(path) for path in raw_hashes})

            final_ids = read_rows(out / 'final_model/training_ids.tsv')
            self.assertEqual({row['row_id'] for row in final_ids}, expected_ids)
            self.assertEqual({row['sample_id'] for row in final_ids}, {'HEK293T_WT', 'HEK293T_IVT'})
            _, bundle = load_model_bundle(out / 'final_model')
            self.assertEqual(bundle['training_rows'], 24)
            self.assertEqual(bundle['epochs'], 1)
            self.assertEqual(bundle['epoch_policy'], 'best validation epoch from training cell only')
            for label, expected in {'0': 4 / 3, '1': 4 / 3, '2': 2 / 3}.items():
                self.assertAlmostEqual(bundle['class_weights'][label], expected)
            model, _, _ = _load_model(out / 'final_model')
            self.assertEqual(tuple(model.output_shape), (None, 3))
            self.assertEqual({tensor.name for tensor in model.inputs},
                             {'sequence_tokens', 'read_features', 'read_mask', 'sequence_repeat_mask'})
            with (out / 'final_model/history.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_changed_split_manifest_is_rejected_before_fitting(self):
        from training.cell_model import train_cell_model
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index, datasets, samples = self.prepare(root)
            train = datasets / 'HEK293T/adenine/train/sites.tsv'
            with train.open('a', encoding='utf-8') as handle:
                handle.write('tampered\n')
            with patch('tensorflow.keras.Model.fit', side_effect=AssertionError('No fitting allowed')) as fit:
                with self.assertRaises(ValueError):
                    train_cell_model(index, datasets, samples, root / 'bad', epochs=1)
                fit.assert_not_called()
            self.assertFalse((root / 'bad/train/best.keras').exists())

    def test_actual_overlap_is_rejected_even_with_distinct_declared_group_ids(self):
        from training.cell_model import train_cell_model
        from utils.references.catalog import sha256
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index, datasets, samples = self.prepare(root)
            train_path = datasets / 'HEK293T/adenine/train/sites.tsv'
            validation_path = datasets / 'HEK293T/adenine/validation/sites.tsv'
            training, validation = read_rows(train_path), read_rows(validation_path)
            moved = next(row for row in training if row['sample_id'] == 'HEK293T_WT')
            training.remove(moved)
            moved['group_id'] = 'declared_separate_group'
            validation.append(moved)
            self.assertFalse({r['group_id'] for r in training} & {r['group_id'] for r in validation})
            self.assertTrue(any(row['chrom'] == moved['chrom'] and row['start'] == moved['start']
                                for row in training))  # Paired IVT remains at the same locus.
            write_rows(train_path, training)
            write_rows(validation_path, validation)

            assignments_path = datasets / 'training_assignments.tsv'
            assignments = read_rows(assignments_path)
            for row in assignments:
                if row['row_id'] == moved['row_id']:
                    row.update(split='validation', group_id=moved['group_id'])
            write_rows(assignments_path, assignments)
            hela_path = datasets / 'HeLa/adenine/sites.tsv'
            hela = read_rows(hela_path)
            for row in hela:
                if row['chrom'] == moved['chrom'] and row['start'] == moved['start']:
                    row.update(overlap_training_window='1', overlap_validation_window='1')
            write_rows(hela_path, hela)
            # Refresh checksums to exercise coordinate grouping, not just stale-file detection.
            run_path = datasets / 'run.json'
            run = json.loads(run_path.read_text())
            for path in (train_path, validation_path, hela_path):
                run['files_sha256'][str(path.relative_to(datasets))] = sha256(path)
            run['sizes']['train'] -= 1
            run['sizes']['validation'] += 1
            run_path.write_text(json.dumps(run))
            with patch('tensorflow.keras.Model.fit', side_effect=AssertionError('No fitting allowed')) as fit:
                with self.assertRaisesRegex(ValueError, '[Gg]roup|overlap|window|split'):
                    train_cell_model(index, datasets, samples, root / 'bad_groups', epochs=1)
                fit.assert_not_called()


@unittest.skipUnless(AVAILABLE, 'TensorFlow/Arrow/pysam required')
class SiteBagCellDataTest(unittest.TestCase):
    def prepare(self, root):
        from tests.site_bag_fixture import prepare_training_bags
        from training.cell_datasets import build_cell_datasets
        index = prepare_training_bags(root, high_depth=True)
        samples = [dict(sample_id=cell + '_' + condition, cell=cell,
                        reference_cell=cell, condition=condition)
                   for cell in ('HEK293T', 'HeLa') for condition in ('WT', 'IVT')]
        datasets = root / 'cell_datasets'
        build_cell_datasets({'adenine': index}, samples, datasets)
        return index, datasets, samples

    def test_common_bag_subset_preserves_sampling_locations_and_excludes_hela(self):
        from training.cell_model_data import prepare_cell_model_data
        from utils.features.site_dataset import SiteBagBatches
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index, datasets, samples = self.prepare(root)
            originals = {row['row_id']: row for row in read_rows(index)}
            out = root / 'cell_data'
            # Subsetting reads only TSV/JSON metadata; it must not decode tensors.
            with patch('pyarrow.parquet.ParquetFile', side_effect=AssertionError('Parquet opened during metadata subset')):
                result = prepare_cell_model_data(index, datasets, samples, out)
            subset = read_rows(result['index_path'])
            self.assertEqual(len(subset), 24)
            self.assertEqual({row['sample_id'] for row in subset}, {'HEK293T_WT', 'HEK293T_IVT'})
            self.assertFalse(list(out.rglob('*.parquet')))
            run = json.loads((out / 'index/run.json').read_text())
            self.assertEqual((run['schema'], run['max_reads'], run['sampling_seed']),
                             ('site_bag_index_v3', 300, 42))
            self.assertEqual(len(run['feature_sources']), 2)
            for row in subset:
                source = originals[row['row_id']]
                for name in ('site_parquet', 'site_row_group', 'site_row_offset', 'label',
                             'eligible_depth', 'retained_depth', 'reference_sequence'):
                    self.assertEqual(row[name], source[name])
            high = next(row for row in subset if row['sample_id'] == 'HEK293T_WT' and row['start'] == '100')
            self.assertEqual((high['eligible_depth'], high['retained_depth']), ('601', '300'))
            data = SiteBagBatches(result['index_path'], None)
            inputs, labels = data._batch([next(row for row in data.rows if row['row_id'] == high['row_id'])])
            self.assertEqual(inputs['read_features'].shape, (1, 300, 19, 18))
            self.assertEqual(int(inputs['read_mask'].sum()), 300)
            self.assertEqual(int(labels[0]), int(high['label']))

    def test_common_bag_fit_uses_only_hek_and_exports_fixed_input_bundle(self):
        from training.cell_model import train_cell_model
        from utils.features.site_dataset import SiteBagBatches
        from prediction.predict import _load_model
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index, datasets, samples = self.prepare(root)
            observed_ids = []
            original = SiteBagBatches.tensorflow_dataset
            def observe(dataset, *args, **kwargs):
                observed_ids.append({row['row_id'] for row in dataset.rows})
                self.assertEqual({row['sample_id'].split('_')[0] for row in dataset.rows}, {'HEK293T'})
                return original(dataset, *args, **kwargs)
            out = root / 'model'
            with patch.object(SiteBagBatches, 'tensorflow_dataset', observe):
                result = train_cell_model(index, datasets, samples, out, epochs=1, patience=0,
                                          max_reads=300, sampling_seed=42, batch_sites=8)
            self.assertEqual(result['status'], 'complete')
            expected = {row['row_id'] for row in read_rows(index) if row['sample_id'].startswith('HEK293T_')}
            self.assertIn(expected, observed_ids)
            bundle, _, metadata = _load_model(out / 'final_model')
            self.assertEqual(tuple(bundle.get_layer('read_features').output.shape), (None, 300, 19, 18))
            self.assertEqual(metadata['training_rows'], 24)
            self.assertFalse(list(out.rglob('*.parquet')))

    def test_changed_sampling_policy_and_invalid_retained_depth_are_rejected(self):
        from training.cell_model_data import prepare_cell_model_data
        from training.cell_datasets import build_cell_datasets
        from utils.references.catalog import sha256
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index, datasets, samples = self.prepare(root)
            run_path = index.parent / 'run.json'
            original_run = json.loads(run_path.read_text())
            # Refresh manifest checksums so this exercises sampling consistency,
            # rather than stopping only at the prior index-run checksum check.
            run = dict(original_run, sampling_seed=43)
            run_path.write_text(json.dumps(run))
            changed_datasets = root / 'changed_cell_datasets'
            build_cell_datasets({'adenine': index}, samples, changed_datasets)
            with self.assertRaisesRegex(ValueError, 'sampling policy'):
                prepare_cell_model_data(index, changed_datasets, samples, root / 'bad_policy')
            run_path.write_text(json.dumps(original_run))
            rows = read_rows(index)
            selected = next(row for row in rows if row['sample_id'] == 'HEK293T_WT' and row['start'] == '100')
            selected['retained_depth'] = '299'
            write_rows(index, rows)
            original_run['index_sha256'] = sha256(index)
            run_path.write_text(json.dumps(original_run))
            depth_datasets = root / 'depth_cell_datasets'
            build_cell_datasets({'adenine': index}, samples, depth_datasets)
            with self.assertRaisesRegex(ValueError, 'Retained depth'):
                prepare_cell_model_data(index, depth_datasets, samples, root / 'bad_depth')


if __name__ == '__main__':
    unittest.main()
