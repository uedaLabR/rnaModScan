"""Hand-counted weak-label metrics and unambiguous summary files."""
import csv
from pathlib import Path
import tempfile
import unittest

from training.validation import three_class_metrics, write_metrics_summary, label_metrics


def examples(matrix):
    labels, scores = [], []
    for actual, row in enumerate(matrix):
        for predicted, count in enumerate(row):
            for _ in range(count):
                labels.append(actual)
                scores.append([float(i == predicted) for i in range(3)])
    return labels, scores


class ValidationMetricsTest(unittest.TestCase):
    def test_hand_counted_three_class_metrics_and_labelled_tsv(self):
        matrix = [[5, 1, 0], [2, 3, 1], [0, 2, 6]]
        report = three_class_metrics(*examples(matrix))
        self.assertEqual(report['confusion_matrix'], matrix)
        self.assertEqual(report['sites'], 20)
        self.assertAlmostEqual(report['accuracy'], .7)
        expected = {'m6A': (5, 2, 1, 12, 6, .85, 5/7, 5/6, 10/13),
                    'inosine': (3, 3, 3, 11, 6, .7, .5, .5, .5),
                    'false_positive': (6, 1, 2, 11, 8, .85, 6/7, .75, .8)}
        names = ('tp', 'fp', 'fn', 'tn', 'support', 'one_vs_rest_accuracy', 'precision', 'recall', 'f1')
        for name, values in expected.items():
            for field, expected_value in zip(names, values):
                self.assertAlmostEqual(report['per_class'][name][field], expected_value, msg=name+' '+field)
        self.assertAlmostEqual(report['macro_f1'], (10/13 + .5 + .8)/3)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_metrics_summary(report, directory)
            with Path(paths['metrics_summary']).open() as handle:
                rows = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual([r['class'] for r in rows], ['all', 'm6A', 'inosine', 'false_positive'])
            self.assertEqual(float(rows[0]['accuracy']), .7)
            self.assertTrue(all(r['accuracy'] == '' for r in rows[1:]))
            self.assertEqual(float(rows[1]['one_vs_rest_accuracy']), .85)
            self.assertTrue(all('weak labels' in r['interpretation'] for r in rows))
            with Path(paths['confusion_matrix']).open() as handle:
                confusion = list(csv.reader(handle, delimiter='\t'))
            self.assertEqual(confusion[0], ['true_class / predicted_class', 'm6A', 'inosine', 'false_positive'])
            self.assertEqual(confusion[2], ['inosine', '2', '3', '1'])

    def test_empty_split_is_undefined_and_mismatched_lengths_are_rejected(self):
        report = three_class_metrics([], [])
        self.assertEqual(report['confusion_matrix'], [[0, 0, 0]] * 3)
        self.assertIsNone(report['accuracy'])
        self.assertIsNone(report['macro_f1'])
        for value in report['per_class'].values():
            self.assertEqual(value['support'], 0)
            self.assertEqual((value['tp'], value['fp'], value['fn'], value['tn']), (0, 0, 0, 0))
            for field in ('precision', 'recall', 'f1', 'one_vs_rest_accuracy'):
                self.assertIsNone(value[field])
        with tempfile.TemporaryDirectory() as directory:
            write_metrics_summary(report, directory)
            with (Path(directory) / 'metrics_summary.tsv').open() as handle:
                rows = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(r['accuracy'] == '' and r['one_vs_rest_accuracy'] == '' for r in rows))
        with self.assertRaises(ValueError):
            three_class_metrics([0], [])

    def test_absent_class_has_no_positive_accuracy_claim_and_binary_writer_still_works(self):
        report = three_class_metrics([2, 2, 2], [[0, 0, 1]] * 3)
        self.assertEqual(report['accuracy'], 1)
        absent = report['per_class']['inosine']
        self.assertEqual(absent['support'], 0)
        self.assertEqual(absent['tn'], 3)
        self.assertEqual(absent['one_vs_rest_accuracy'], 1)
        self.assertIsNone(absent['precision'])
        self.assertIsNone(absent['recall'])
        self.assertIsNone(absent['f1'])
        self.assertEqual(report['macro_f1'], 1)  # Existing defined-class convention.
        self.assertIn('defined', report['macro_f1_policy'])
        with tempfile.TemporaryDirectory() as directory:
            binary = label_metrics([0, 1, 1], [.1, .8, .2], .5)
            write_metrics_summary(binary, directory)
            with (Path(directory) / 'confusion_matrix.tsv').open() as handle:
                rows = list(csv.reader(handle, delimiter='\t'))
            self.assertEqual(rows, [['true_class / predicted_class', 'unmodified', 'modified'],
                                    ['unmodified', '1', '0'], ['modified', '1', '1']])


if __name__ == '__main__':
    unittest.main()
