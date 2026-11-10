"""Cell-specific dataset preparation uses small synthetic metadata only."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.cell_datasets import assign_training_groups, build_cell_datasets


class CellDatasetsTest(unittest.TestCase):
    def fixture(self, root):
        samples = [dict(sample_id=cell + '_' + condition, cell=cell,
                        reference_cell=cell, condition=condition)
                   for cell in ('HEK293T', 'HeLa') for condition in ('WT', 'IVT')]
        paths = {}
        for target in ('m6A', 'inosine'):
            directory = root / 'indices' / target
            directory.mkdir(parents=True)
            path = directory / 'sites.tsv'
            fields = ['row_id', 'sample_id', 'chrom', 'start', 'end', 'strand', 'modification', 'label', 'retained_depth']
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fields, delimiter='\t')
                writer.writeheader()
                for start in range(100, 2100, 100):
                    for sample in samples:
                        writer.writerow(dict(row_id=f'{sample["sample_id"]}|chr1:{start}:+:{target}',
                            sample_id=sample['sample_id'], chrom='chr1', start=start, end=start + 1,
                            strand='+', modification=target, label=int(sample['condition'] == 'WT'), retained_depth=2))
            (directory / 'run.json').write_text(json.dumps(dict(status='complete', target=target,
                require_labels=True, counts=dict(not_extracted_sites=0), feature_sources=[])))
            paths[target] = path
        return paths, samples

    def read(self, path):
        with path.open() as handle:
            return list(csv.DictReader(handle, delimiter='\t'))

    def test_training_cell_only_and_separate_hela_not_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, samples = self.fixture(root)
            existing_cv = root / 'splits' / 'five_fold.txt'
            existing_cv.parent.mkdir()
            existing_cv.write_text('preserve existing CV')
            out = root / 'cell_datasets'
            with patch('utils.features.parquet.require_arrow', side_effect=AssertionError('No Parquet read allowed')):
                report = build_cell_datasets(paths, samples, out)
            self.assertEqual(report['sizes'], dict(train=72, validation=8, separate_analysis=80))
            self.assertFalse(list(out.rglob('test')))
            for target in paths:
                train = self.read(out / 'HEK293T' / target / 'train/sites.tsv')
                validation = self.read(out / 'HEK293T' / target / 'validation/sites.tsv')
                hela = self.read(out / 'HeLa' / target / 'sites.tsv')
                self.assertEqual({row['sample_id'] for row in train + validation}, {'HEK293T_WT', 'HEK293T_IVT'})
                self.assertEqual({row['sample_id'] for row in hela}, {'HeLa_WT', 'HeLa_IVT'})
                self.assertFalse({row['group_id'] for row in train} & {row['group_id'] for row in validation})
                self.assertNotIn('label', train[0])
                self.assertEqual({row['label'] for row in hela}, {'0', '1'})
                self.assertTrue(all(int(row['overlap_training_window']) + int(row['overlap_validation_window']) == 1
                                    for row in hela))
            self.assertEqual(existing_cv.read_text(), 'preserve existing CV')
            self.assertFalse((out / 'INCOMPLETE.txt').exists())
            with self.assertRaises(FileExistsError):
                build_cell_datasets(paths, samples, out)

    def test_labels_and_separate_cell_positions_cannot_change_training_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, samples = self.fixture(root)
            build_cell_datasets(paths, samples, root / 'first')
            for path in paths.values():
                rows = self.read(path)
                for row in rows:
                    row['label'] = '0' if row['label'] == '1' else '1'
                    if row['sample_id'].startswith('HeLa_'):
                        row['start'] = str(int(row['start']) + 50)
                        row['end'] = str(int(row['start']) + 1)
                with path.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, list(rows[0]), delimiter='\t')
                    writer.writeheader()
                    writer.writerows(reversed(rows))
            build_cell_datasets(paths, samples, root / 'second')
            self.assertEqual((root / 'first/training_assignments.tsv').read_text(),
                             (root / 'second/training_assignments.tsv').read_text())

    def test_overlapping_windows_across_targets_strands_and_conditions_stay_together(self):
        rows = [dict(row_id=str(index), chrom='chr1', start=start,
                     modification='m6A' if index % 2 else 'inosine', strand='-' if index % 2 else '+')
                for index, start in enumerate((100, 118, 136, 500, 900))]
        assign_training_groups(rows, flank=9, validation_fraction=.1, seed=42)
        self.assertEqual(len({row['group_id'] for row in rows[:3]}), 1)
        self.assertEqual(len({row['split'] for row in rows[:3]}), 1)
        self.assertEqual({row['split'] for row in rows}, {'train', 'validation'})

    def test_hek293_is_not_implicitly_mapped_to_hek293t(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, samples = self.fixture(root)
            for sample in samples:
                if sample['cell'] == 'HEK293T':
                    sample['cell'] = 'HEK293'
            with self.assertRaises(ValueError):
                build_cell_datasets(paths, samples, root / 'bad')
            self.assertFalse((root / 'bad').exists())

    def test_invalid_fraction_and_incomplete_index_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, samples = self.fixture(root)
            with self.assertRaises(ValueError):
                build_cell_datasets(paths, samples, root / 'bad', validation_fraction=0)
            (paths['m6A'].parent / 'INCOMPLETE.txt').write_text('still extracting')
            with self.assertRaises(ValueError):
                build_cell_datasets(paths, samples, root / 'bad')
            self.assertFalse((root / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
