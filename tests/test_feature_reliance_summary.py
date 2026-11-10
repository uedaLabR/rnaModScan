"""Scientific-unit checks for frozen-model reliance reports (synthetic data)."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.feature_reliance.summarize import (
    CONDITIONS, effect_rows, export, metric_rows, read_completed_run,
)
from utils.io.excel import verify_workbook


def class_values(f1, support=10):
    return dict(support=support, tp=8, fp=2, fn=2, tn=18,
                precision=f1, recall=f1, f1=f1, one_vs_rest_accuracy=.8)


def metrics(f1=.8, loss=.2):
    def scope(sites):
        return dict(sites=sites, accuracy=.8, macro_f1=f1, log_loss=loss,
            per_class={name: class_values(f1) for name in ('m6A', 'inosine', 'false_positive')},
            confusion_matrix=[[8, 1, 1], [1, 8, 1], [1, 1, 8]])
    return dict(overall=scope(30), by_motif=dict(DRACH=scope(16), non_DRACH=scope(14)))


def full_run():
    seeds = [17, 43, 101]
    entries = [dict(condition='baseline', seed=None, metrics=metrics())]
    for condition in CONDITIONS:
        for index, seed in enumerate(seeds):
            f1 = [.6, .7, .8][index] if condition == 'sequence' else .85
            entries.append(dict(condition=condition, seed=seed,
                metrics=metrics(f1=f1, loss=.3), class_flip_rate=.1,
                mean_true_class_probability_change=-.04,
                mean_absolute_probability_change=.03))
    return dict(status='complete', cohort=dict(sites=30, full_test_sites=30, pilot=False, max_sites=None),
                fold=0, variant_seeds=seeds, evaluations=entries)


class FeatureRelianceSummaryTest(unittest.TestCase):
    def test_paired_drop_and_sd_use_percentage_points(self):
        source = full_run()
        values = {(row['condition'], row['scope'], row['metric']): row
                  for row in effect_rows(source)}
        effect = values['sequence', 'overall', 'macro_f1']
        self.assertAlmostEqual(effect['baseline'], 80)
        self.assertAlmostEqual(effect['perturbed_mean'], 70)
        self.assertAlmostEqual(effect['delta_mean'], 10)
        self.assertAlmostEqual(effect['delta_sd'], 10)
        self.assertEqual(effect['defined_repeats'], 3)
        # An improved score is negative reliance, never clipped or renormalized.
        self.assertAlmostEqual(values['repeat', 'inosine_all', 'f1']['delta_mean'], -5)
        self.assertNotAlmostEqual(sum(values[c, 'overall', 'macro_f1']['delta_mean']
                                      for c in CONDITIONS), 100)

    def test_loss_and_true_probability_have_explicit_directions(self):
        values = {(r['condition'], r['scope'], r['metric']): r for r in effect_rows(full_run())}
        self.assertAlmostEqual(values['sequence', 'overall', 'log_loss']['delta_mean'], .1)
        self.assertAlmostEqual(values['sequence', 'overall', 'class_flip_rate']['delta_mean'], 10)
        value = values['sequence', 'overall', 'mean_true_class_probability_change']
        self.assertAlmostEqual(value['perturbed_mean'], -4)
        self.assertAlmostEqual(value['delta_mean'], 4)

    def test_motif_scope_uses_all_class_opportunities_and_null_stays_null(self):
        source = full_run()
        source['evaluations'][1]['metrics']['by_motif']['DRACH']['per_class']['m6A']['precision'] = None
        rows = metric_rows(source)
        selected = next(r for r in rows if r['condition'] == 'sequence' and r['seed'] == 17
                        and r['scope'] == 'm6A_DRACH')
        self.assertEqual(selected['sites'], 16)
        self.assertEqual(selected['support'], 10)
        self.assertIsNone(selected['precision'])
        effect = next(r for r in effect_rows(source) if r['condition'] == 'sequence'
                      and r['scope'] == 'm6A_DRACH' and r['metric'] == 'precision')
        self.assertEqual(effect['defined_repeats'], 2)

    def test_incomplete_pilot_duplicate_and_missing_seed_are_rejected(self):
        variants = []
        running = full_run(); running['status'] = 'running'; variants.append(running)
        pilot = full_run(); pilot['cohort']['pilot'] = True; variants.append(pilot)
        capped = full_run(); capped['cohort']['max_sites'] = 512; variants.append(capped)
        partial = full_run(); partial['cohort']['full_test_sites'] = 31; variants.append(partial)
        incomplete = full_run(); incomplete['evaluations'].pop(); variants.append(incomplete)
        duplicate = full_run(); duplicate['evaluations'].append(deepcopy(duplicate['evaluations'][1])); variants.append(duplicate)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run.json'
            for source in variants:
                with self.subTest(status=source['status'], sites=source['cohort']):
                    path.write_text(json.dumps(source))
                    with self.assertRaises(ValueError):
                        read_completed_run(tmp)

    def test_native_export_preserves_source_counts_and_existing_outputs(self):
        source = full_run()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / 'source'; original.mkdir()
            content = json.dumps(source).encode()
            (original / 'run.json').write_bytes(content)
            data, figures = root / 'data', root / 'figures'
            with patch('analysis.feature_reliance.summarize.draw_effects', return_value=[]):
                report = export(original, data, figures)
            self.assertEqual((original / 'run.json').read_bytes(), content)
            self.assertTrue(report['native_cells_verified'])
            book = verify_workbook(Path(report['workbook']))
            self.assertEqual(len(book), 4)
            repeated = book['Metrics_by_repeat']
            columns = repeated[0]
            rows = [dict(zip(columns, row)) for row in repeated[1:]]
            selected = next(r for r in rows if r['condition'] == 'sequence' and r['seed'] == 17
                            and r['scope'] == 'm6A_DRACH')
            self.assertEqual(selected['sites'], 16)
            self.assertEqual(selected['tp'], 8)
            self.assertAlmostEqual(selected['f1'], .6)
            self.assertEqual(selected['seed'], 17)
            with self.assertRaises(FileExistsError):
                export(original, data, figures)


if __name__ == '__main__':
    unittest.main()
