"""Export saved epoch histories and best-model class counts, without inference."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from utils.io.artifacts import completed_run
from utils.io.excel import write_workbook, verify_workbook
from utils.references.catalog import sha256

METRICS = ('one_vs_rest_accuracy', 'precision', 'recall', 'f1')
CLASS_FIELDS = ['fold', 'best_epoch', 'scope', 'class', 'sites', 'support', 'predicted_sites',
                'tp', 'fp', 'fn', 'tn', *METRICS]


def sheet(name, headers, rows):
    return dict(name=name, headers=headers, rows=rows)


def validate_counts(metrics):
    """Reconcile class metrics with the saved three-class confusion matrix."""
    names, matrix, total = metrics['class_names'], metrics['confusion_matrix'], metrics['sites']
    assert sum(sum(row) for row in matrix) == total
    for index, name in enumerate(names):
        row = metrics['per_class'][name]
        tp = matrix[index][index]
        fn = sum(matrix[index]) - tp
        fp = sum(values[index] for values in matrix) - tp
        tn = total - tp - fn - fp
        assert [row[key] for key in ('tp', 'fp', 'fn', 'tn')] == [tp, fp, fn, tn]
        assert row['support'] == tp + fn
        for key, numerator, denominator in [
                ('one_vs_rest_accuracy', tp + tn, total), ('precision', tp, tp + fp),
                ('recall', tp, tp + fn), ('f1', 2 * tp, 2 * tp + fp + fn)]:
            expected = numerator / denominator if denominator else None
            actual = row[key]
            assert (expected is None and actual is None) or (
                expected is not None and actual is not None and math.isclose(actual, expected, abs_tol=1e-12))


def export_report(run_dir, output_name='training_details.xlsx', include_fp_origins=False):
    run_dir = Path(run_dir).resolve()
    completed_run(run_dir)
    if Path(output_name).name != output_name or not output_name.endswith('.xlsx'):
        raise ValueError('output_name must be an Excel filename without a directory')
    destination = run_dir / 'summary' / output_name
    if destination.exists():
        raise FileExistsError(destination)
    epochs, overview, sources = [], [], []
    classes = {split: [] for split in ('validation', 'test')}
    overall, confusions = [], []
    for fold in range(5):
        directory = run_dir / 'adenine' / f'fold_{fold}'
        completed_run(directory)
        settings_path = directory / 'train/settings.json'
        settings = json.loads(settings_path.read_text())
        history_path = directory / 'train/history.csv'
        with history_path.open(newline='') as handle:
            history = list(csv.DictReader(handle))
        best = min(history, key=lambda row: float(row['val_loss']))
        best_epoch = int(best['epoch']) + 1
        overview.append([fold, str(settings['learning_rate']), settings['seed'], best_epoch,
                         len(history), *[settings['split_sites'][name] for name in ('train', 'validation', 'test')]])
        for row in history:
            epoch = int(row['epoch']) + 1
            epochs.append([fold, epoch, float(row['accuracy']), float(row['loss']),
                           float(row['val_loss']), float(row['val_accuracy']), epoch == best_epoch])
        paths = [history_path, settings_path, directory / 'run.json']
        for split in ('validation', 'test'):
            metrics_path = directory / split / 'metrics.json'
            paths.append(metrics_path)
            metrics = json.loads(metrics_path.read_text())
            if split == 'validation':
                assert math.isclose(metrics['accuracy'], float(best['val_accuracy']), abs_tol=1e-6)
            overall.append([fold, split, best_epoch, metrics['sites'], metrics['accuracy'], metrics['macro_f1']])
            scopes = [('all', metrics), *[(scope, values) for scope, values in metrics['by_motif'].items()
                                         if values['sites'] > 0]]
            for scope, values in scopes:
                validate_counts(values)
                for name in values['class_names']:
                    v = values['per_class'][name]
                    classes[split].append([fold, best_epoch, scope, name, values['sites'], v['support'],
                        v['tp'] + v['fp'], *[v[key] for key in ('tp', 'fp', 'fn', 'tn', *METRICS)]])
                for true, counts in zip(values['class_names'], values['confusion_matrix']):
                    for predicted, count in zip(values['class_names'], counts):
                        confusions.append([fold, split, scope, true, predicted, count])
        for path in paths:
            sources.append([fold, str(path), sha256(path)])
    means = []
    # Validation-only descriptive summaries; test does not choose models or settings.
    for scope in ('all', 'DRACH', 'non_DRACH'):
        for name in ('m6A', 'inosine', 'false_positive'):
            selected = [dict(zip(CLASS_FIELDS, row)) for row in classes['validation']
                        if row[2:4] == [scope, name]]
            assert len(selected) == 5
            for metric in METRICS:
                values = [row[metric] for row in selected if row[metric] is not None]
                means.append([scope, name, metric, len(values),
                              mean(values) if len(values) == 5 else None,
                              stdev(values) if len(values) == 5 else None])
    notes = [
        ['Run', str(run_dir)],
        ['Epoch_history', 'All recorded epochs, one-based epoch numbers; best chosen by minimum val_loss.'],
        ['Validation_best / Test_best', 'Best-model metrics only; no per-epoch class confusion matrices were saved.'],
        ['Units', 'Accuracy, precision, recall and F1 retain original numeric fractions (0–1). Multiply by 100 for percent.'],
        ['Raw precision', 'Values are not rounded before storage; Excel displays six decimal places. TSVs retain the original float values.'],
        ['Class accuracy', 'One-versus-rest accuracy=(TP+TN)/N. For rare classes this can be high because TN dominates; consider recall/F1 too.'],
        ['Overall accuracy', 'Correct three-class argmax predictions / N; listed separately in Overall_best.'],
        ['Recall / F1', 'Recall=TP/(TP+FN); F1=2TP/(2TP+FP+FN). Counts are sample/site observations, not reads.'],
        ['Loss', 'Training loss is class-weighted; validation loss is unweighted. They are not directly interchangeable.'],
        ['Confusion_counts', 'Rows describe true label and predicted class; each count is a site observation.'],
        ['Mean / SD', 'Unweighted validation mean and sample SD (ddof=1), only when all five scores are defined.'],
        ['Test', 'Reported separately; test scores are not used for learning-rate or checkpoint selection.'],
        ['Interpretation', 'Internal grouped CV against operational weak labels, before DB/DRACH rescue; not independent biological accuracy.']]
    sheets = [sheet('Readme', ['Item', 'Meaning'], notes),
        sheet('Fold_overview', ['fold', 'learning_rate', 'seed', 'best_epoch', 'completed_epochs',
                               'train_sites', 'validation_sites', 'test_sites'], overview),
        sheet('Epoch_history', ['fold', 'epoch', 'accuracy', 'loss', 'val_loss', 'val_accuracy', 'best_epoch'], epochs),
        sheet('Overall_best', ['fold', 'split', 'best_epoch', 'sites', 'accuracy', 'macro_f1'], overall),
        sheet('Validation_best', CLASS_FIELDS, [row for row in classes['validation'] if row[2] == 'all']),
        sheet('Validation_by_motif', CLASS_FIELDS, [row for row in classes['validation'] if row[2] != 'all']),
        sheet('Test_best', CLASS_FIELDS, [row for row in classes['test'] if row[2] == 'all']),
        sheet('Test_by_motif', CLASS_FIELDS, [row for row in classes['test'] if row[2] != 'all']),
        sheet('Validation_mean_SD', ['scope', 'class', 'metric', 'defined_folds', 'mean', 'sample_sd'], means),
        sheet('Confusion_counts', ['fold', 'split', 'scope', 'true_class', 'predicted_class', 'sites'], confusions),
        sheet('Sources', ['fold', 'source_path', 'sha256'], sources)]
    if include_fp_origins:
        from training.negative_origin_report import negative_origin_sheets
        extra, extra_sources = negative_origin_sheets(run_dir)
        sheets.extend(extra)
        sources.extend([None, path, digest] for path, digest in extra_sources)
        notes.extend([
            ['FP origins', 'Paired_IVT: same cell/coordinate/strand as a WT positive; modification is the WT positive class. This group has priority.'],
            ['Other IVT', 'Additional_IVT: original IVT candidates, separated by actual m6A/inosine depth>=10,Nmod>=3,AF>=0.1. Both calls form one separate group.'],
            ['WT negatives', 'WT_IVT_override: both target labels explicitly negative by the IVT comparison; absent overlap/coverage is not a negative.'],
            ['FP subgroup metrics', 'Each subgroup contains FP-labelled sites only. Conditional accuracy equals FP recall=predicted_FP/sites. Precision/F1 are not estimated from this negative-only subset; use full-class best metrics.']])
    destination.parent.mkdir(exist_ok=True)
    write_workbook(destination, sheets, float_format='0.000000')
    saved = verify_workbook(destination)
    for data in sheets:
        assert saved[data['name']] == [data['headers'], *data['rows']], data['name']
    tabular = destination.parent / destination.stem
    tabular.mkdir(exist_ok=True)
    for data in sheets:
        with (tabular / (data['name'] + '.tsv')).open('w', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(data['headers'])
            writer.writerows(data['rows'])
    report = dict(status='complete', workbook=str(destination), epoch_rows=len(epochs),
                  best_class_rows={key: len(rows) for key, rows in classes.items()},
                  tables=[data['name'] for data in sheets], original_counts_reconciled=True,
                  source_sha256=sha256(run_dir / 'run.json'))
    (tabular / 'export.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-name', default='training_details.xlsx')
    parser.add_argument('--include-fp-origins', action='store_true')
    args = parser.parse_args()
    print(json.dumps(export_report(args.run_dir, args.output_name, args.include_fp_origins), indent=2))


if __name__ == '__main__':
    main()
