"""Use synthetic validation metrics; no model fitting or test-score reads."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev
import tempfile
import unittest

from training.validation_comparison import summarize_validation_folds
from utils.io.excel import verify_workbook


class ValidationComparisonTest(unittest.TestCase):
    def save_fold(self, root, fold, rate, m6a=.75, inosine=.9):
        path = root / 'adenine' / f'fold_{fold}'
        for name in ('train', 'validation', 'test'):
            (path / name).mkdir(parents=True)
        run = dict(status='complete', target='adenine', feature_set='sequence_reads_repeat18_v1',
                   learning_rate=rate, model=dict(read_hidden_dim=16), index_sha256='same-index',
                   split_files={split: dict(sha256=f'{fold}-{split}') for split in ('train', 'validation', 'test')},
                   batch_sites=16, epochs=20, seed=42, patience=3, class_weights={'0': 1},
                   tensorflow_version='2.10.1')
        (path / 'run.json').write_text(json.dumps(run))
        (path / 'train/settings.json').write_text(json.dumps(dict(learning_rate=rate, seed=42)))
        (path / 'train/history.csv').write_text('epoch,val_loss\n0,0.2\n1,0.1\n2,0.15\n')
        metrics = dict(class_names=['m6A', 'inosine', 'false_positive'], sites=100,
            accuracy=.93, macro_f1=.85,
            by_motif=dict(non_DRACH=dict(per_class=dict(m6A=dict(f1=m6a, support=30)))),
            per_class=dict(inosine=dict(f1=inosine, support=10)))
        (path / 'validation/metrics.json').write_text(json.dumps(metrics))
        # Any attempt to use held-out test scores would make the summary fail.
        (path / 'test/metrics.json').write_text('THIS TEST FILE MUST NEVER BE READ')
        return path

    def table(self, root, filename):
        with (root / 'summary' / filename).open() as handle:
            return list(csv.DictReader(handle, delimiter='\t'))

    def test_partial_outputs_keep_missing_baselines_and_aggregates_blank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self.save_fold(root / 'old', 0, .001, .7523496240601504, .908289241622575)
            new = root / 'new'
            report = summarize_validation_folds(new, baseline)
            self.assertEqual((report['completed_folds'], report['baseline_completed_folds']), (0, 1))
            rows = self.table(new, 'validation_comparison.tsv')
            self.assertAlmostEqual(float(rows[0]['old_m6a_non_DRACH_F1_percent']), 75.23496240601504)
            self.assertEqual(rows[0]['new_m6a_non_DRACH_F1_percent'], '')
            self.assertEqual(rows[1]['old_status'], 'not_available')
            self.assertEqual(rows[5]['old_m6a_non_DRACH_F1_percent'], '')
            self.assertEqual(rows[6]['new_inosine_F1_percent'], '')
            self.save_fold(new, 0, .0003)
            refreshed = summarize_validation_folds(new, baseline)
            self.assertEqual(refreshed['completed_folds'], 1)
            self.assertIsNone(refreshed['statistics']['sample_sd']['new']['inosine_f1'])

    def test_fivefold_mean_and_sample_sd_and_single_paired_delta(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self.save_fold(root / 'old', 0, .001, .7523496240601504, .908289241622575)
            values = [.70, .75, .80, .85, .90]
            for fold, score in enumerate(values):
                self.save_fold(root / 'new', fold, .0003, score, .8 + fold * .02)
            report = summarize_validation_folds(root / 'new', baseline)
            self.assertEqual(report['status'], 'complete')
            self.assertAlmostEqual(report['statistics']['mean']['new']['m6a_non_drach_f1'], mean(values))
            self.assertAlmostEqual(report['statistics']['sample_sd']['new']['m6a_non_drach_f1'], stdev(values))
            self.assertIsNone(report['statistics']['mean']['old']['m6a_non_drach_f1'])
            rows = self.table(root / 'new', 'validation_comparison.tsv')
            self.assertAlmostEqual(float(rows[0]['m6a_delta_percentage_points']), 100 * (.70 - .7523496240601504))
            self.assertEqual(rows[5]['m6a_delta_percentage_points'], '')
            workbook = verify_workbook(root / 'new/summary/validation_comparison.xlsx')
            self.assertEqual(workbook['Validation_folds'][1][2], '0.0003')
            self.assertEqual(report['folds'][0]['best_epoch'], 2)

    def test_incomplete_marker_excludes_even_existing_metric_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.save_fold(root / 'new', 0, .0003)
            (path / 'INCOMPLETE.txt').touch()
            (path / 'validation/metrics.json').write_text('INCOMPLETE JSON')
            report = summarize_validation_folds(root / 'new', root / 'old')
            self.assertEqual(report['completed_folds'], 0)
            self.assertEqual(report['folds'][0]['status'], 'running')
            self.assertIsNone(report['folds'][0]['m6a_non_drach_f1'])

    def test_wrong_learning_rate_and_incompatible_splits_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.save_fold(root / 'new', 0, .001)
            baseline = self.save_fold(root / 'old', 0, .001)
            with self.assertRaisesRegex(ValueError, 'learning rate'):
                summarize_validation_folds(root / 'new', baseline)
            settings = json.loads((path / 'train/settings.json').read_text())
            settings['learning_rate'] = .0003
            (path / 'train/settings.json').write_text(json.dumps(settings))
            run = json.loads((path / 'run.json').read_text())
            run.update(learning_rate=.0003)
            run['split_files']['validation']['sha256'] = 'changed'
            (path / 'run.json').write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, 'validation manifest differs'):
                summarize_validation_folds(root / 'new', baseline)

    def test_undefined_metric_does_not_average_only_four_folds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for fold in range(5):
                self.save_fold(root / 'new', fold, .0003, None if fold == 2 else .8, .9)
            report = summarize_validation_folds(root / 'new', root / 'old')
            self.assertEqual(report['completed_folds'], 5)
            self.assertIsNone(report['statistics']['mean']['new']['m6a_non_drach_f1'])
            self.assertIsNone(report['statistics']['sample_sd']['new']['m6a_non_drach_f1'])
            self.assertAlmostEqual(report['statistics']['mean']['new']['inosine_f1'], .9)

    def test_wrong_class_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.save_fold(root / 'new', 0, .0003)
            metrics = json.loads((path / 'validation/metrics.json').read_text())
            metrics['class_names'] = ['false_positive', 'm6A', 'inosine']
            (path / 'validation/metrics.json').write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, 'classes'):
                summarize_validation_folds(root / 'new', root / 'old')
