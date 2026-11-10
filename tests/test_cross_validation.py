import csv
import json
import tempfile
import unittest
from pathlib import Path

from training.cross_validation import assign_position_folds, build_cross_validation, split_groups


class CrossValidationTest(unittest.TestCase):
    def rows(self, mod='m6A'):
        return [dict(row_id=f'{sample}|chr1:{pos}:+:{mod}', sample_id=sample, chrom='chr1', start=str(pos),
                     end=str(pos+1), strand='+', modification=mod, label='1' if sample.endswith('WT') else '0')
                for pos in range(100, 10100, 100)
                for sample in ('HEK293T_WT', 'HEK293T_IVT', 'HeLa_WT', 'HeLa_IVT')]

    def test_same_sites_stay_together_and_labels_do_not_affect_assignment(self):
        rows = self.rows()
        assign_position_folds(rows)
        expected = {r['row_id']: r['fold'] for r in rows}
        for pos in range(100, 10100, 100):
            self.assertEqual(len({r['fold'] for r in rows if r['start'] == str(pos)}), 1)
        for row in rows:
            row['label'] = '1' if row['label'] == '0' else '0'
        rows.reverse()
        assign_position_folds(rows)
        self.assertEqual(expected, {r['row_id']: r['fold'] for r in rows})
        test_ids = []
        for fold in range(5):
            assignment = split_groups(rows, fold)
            self.assertEqual(sum(s == 'train' for s in assignment.values()), 288)
            self.assertEqual(sum(s == 'validation' for s in assignment.values()), 32)
            self.assertEqual(sum(s == 'test' for s in assignment.values()), 80)
            split_groups_seen = {}
            for row in rows:
                split_groups_seen.setdefault(row['group_id'], set()).add(assignment[row['row_id']])
            self.assertTrue(all(len(s) == 1 for s in split_groups_seen.values()))
            test_ids.extend(rid for rid, split in assignment.items() if split == 'test')
        self.assertEqual(len(test_ids), len(set(test_ids)))
        self.assertEqual(set(test_ids), set(expected))

    def test_overlapping_windows_across_strands_and_targets(self):
        rows = self.rows()[:0]
        for i, pos in enumerate((10, 28, 46, 65, 200, 400, 600, 800, 1000)):
            rows.append(dict(row_id=str(i), chrom='chr1', start=pos, strand='-' if i % 2 else '+',
                             modification='m6A' if i % 2 else 'inosine'))
        assign_position_folds(rows)
        self.assertEqual(len({r['group_id'] for r in rows[:3]}), 1)
        self.assertEqual(len({r['fold'] for r in rows[:3]}), 1)
        self.assertNotEqual(rows[2]['group_id'], rows[3]['group_id'])

    def test_writes_five_id_sets_and_actual_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for target in ('m6A', 'inosine'):
                directory = root / target
                directory.mkdir()
                rows = self.rows(target)
                path = directory / 'sites.tsv'
                with path.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, list(rows[0]), delimiter='\t')
                    writer.writeheader(); writer.writerows(rows)
                (directory / 'run.json').write_text(json.dumps(dict(status='complete', target=target,
                    counts=dict(not_extracted_sites=0), feature_sources=[])))
                paths[target] = path
            report = build_cross_validation(paths, root / 'cv')
            self.assertAlmostEqual(report['approximate_fractions']['train'], .72)
            with (root / 'cv/split_sizes.tsv').open() as handle:
                sizes = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(len(sizes), 30)
            for target in paths:
                self.assertTrue((root / f'cv/{target}/fold_4/test/sites.tsv').exists())
                with (root / f'cv/{target}/fold_0/train/sites.tsv').open() as handle:
                    rows = list(csv.DictReader(handle, delimiter='\t'))
                self.assertEqual(len(rows), 288)
                self.assertNotIn('label', rows[0])
                self.assertNotIn('af', rows[0])
