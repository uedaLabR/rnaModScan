"""Summarize the five validation folds without using held-out test scores."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
import tempfile

from utils.io.excel import write_workbook
from utils.references.catalog import sha256

FOLDS = tuple(range(5))
OLD_LEARNING_RATE = 0.001
NEW_LEARNING_RATE = 0.0003
METRICS = ('m6a_non_drach_f1', 'inosine_f1', 'accuracy', 'macro_f1')
CLASSES = ['m6A', 'inosine', 'false_positive']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def probability(value, name):
    """Undefined metrics stay blank rather than becoming zero."""
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f'Invalid validation {name}: {value}')
    return float(value)


def best_epoch(path):
    """CSVLogger stores zero-based epochs; the report uses one-based epochs."""
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    candidates = [(float(row['val_loss']), int(row['epoch']) + 1) for row in rows
                  if math.isfinite(float(row['val_loss']))]
    if not candidates:
        raise ValueError(f'No finite validation loss in {path}')
    return min(candidates)[1]


def fold_path(directory, fold):
    """Accept a CV root or the exact existing baseline fold_0 directory."""
    directory = Path(directory).resolve()
    if directory.name == 'fold_0' and directory.parent.name == 'adenine':
        return directory.parent / f'fold_{fold}'
    return directory / 'adenine' / f'fold_{fold}'


def read_fold(directory, fold, expected_rate):
    """Ignore unfinished metrics, and require the final model completion marker."""
    path = fold_path(directory, fold)
    result = dict(fold=fold, status='pending', learning_rate=expected_rate,
                  seed=None, best_epoch=None, validation_sites=None,
                  m6a_non_drach_support=None, inosine_support=None,
                  **dict.fromkeys(METRICS))
    if not path.exists():
        if expected_rate == OLD_LEARNING_RATE:
            result['status'] = 'not_available'
        return result
    if (path / 'FAILED.txt').exists():
        result['status'] = 'failed'
        return result
    if (path / 'INCOMPLETE.txt').exists() or not (path / 'run.json').is_file():
        result['status'] = 'running'
        return result
    run = read_json(path / 'run.json')
    if run.get('status') != 'complete':
        result['status'] = str(run.get('status', 'running'))
        return result
    if run.get('target') != 'adenine' or run.get('feature_set') != 'sequence_reads_repeat18_v1':
        raise ValueError(f'Expected the current joint adenine model: {path}')
    settings = read_json(path / 'train/settings.json')
    if 'learning_rate' not in settings or 'learning_rate' not in run:
        raise ValueError(f'Missing recorded learning rate in {path}')
    rate = settings['learning_rate']
    if (not isinstance(rate, (int, float)) or isinstance(rate, bool)
            or not math.isclose(rate, expected_rate, rel_tol=1e-12, abs_tol=1e-15)
            or not math.isclose(run['learning_rate'], rate, rel_tol=1e-12, abs_tol=1e-15)):
        raise ValueError(f'Unexpected learning rate in {path}: expected {expected_rate}')
    metrics_path = path / 'validation/metrics.json'
    metrics = read_json(metrics_path)
    if metrics.get('class_names') != CLASSES:
        raise ValueError(f'Unexpected validation classes in {metrics_path}')
    if not {'sites', 'by_motif', 'per_class', 'accuracy', 'macro_f1'} <= metrics.keys():
        raise ValueError(f'Incomplete validation metric schema: {metrics_path}')
    try:
        non_drach = metrics['by_motif']['non_DRACH']['per_class']['m6A']
        inosine = metrics['per_class']['inosine']
    except (KeyError, TypeError) as error:
        raise ValueError(f'Incomplete validation class metrics: {metrics_path}') from error
    if any(not isinstance(item, dict) or not {'f1', 'support'} <= item.keys()
           for item in (non_drach, inosine)):
        raise ValueError(f'Incomplete validation class metrics: {metrics_path}')
    for name, value in [('sites', metrics['sites']), ('m6A support', non_drach['support']),
                        ('inosine support', inosine['support'])]:
        if type(value) is not int or value < 0:
            raise ValueError(f'Invalid validation {name} in {metrics_path}')
    result.update(status='complete', learning_rate=float(rate), seed=settings.get('seed'),
        best_epoch=best_epoch(path / 'train/history.csv'), validation_sites=metrics['sites'],
        m6a_non_drach_support=non_drach['support'], inosine_support=inosine['support'],
        m6a_non_drach_f1=probability(non_drach['f1'], 'non-DRACH m6A F1'),
        inosine_f1=probability(inosine['f1'], 'Inosine F1'),
        accuracy=probability(metrics['accuracy'], 'accuracy'),
        macro_f1=probability(metrics['macro_f1'], 'macro F1'),
        sources={str(file): sha256(file) for file in
                 (path / 'run.json', path / 'train/settings.json',
                  path / 'train/history.csv', metrics_path)},
        run=run, settings=settings)
    return result


def check_paired_settings(old, new):
    """A paired comparison must reuse the architecture, data and other settings."""
    if old['status'] != 'complete' or new['status'] != 'complete':
        return
    for key in ('model', 'feature_set', 'index_sha256', 'batch_sites', 'epochs',
                'seed', 'patience', 'class_weights', 'tensorflow_version'):
        if old['run'].get(key) != new['run'].get(key):
            raise ValueError(f'Fold {new["fold"]}: paired runs differ in {key}, not just learning rate')
    for split in ('train', 'validation', 'test'):
        previous = old['run'].get('split_files', {}).get(split, {}).get('sha256')
        current = new['run'].get('split_files', {}).get(split, {}).get('sha256')
        if previous != current:
            raise ValueError(f'Fold {new["fold"]}: {split} manifest differs between learning rates')


def aggregate(records, statistic):
    """Only a complete, defined set of five scores gets a mean or sample SD."""
    function = mean if statistic == 'mean' else stdev
    result = dict.fromkeys(METRICS)
    if not all(record['status'] == 'complete' for record in records):
        return result
    for metric in METRICS:
        values = [record[metric] for record in records]
        if all(value is not None for value in values):
            result[metric] = function(values)
    return result


def percent(value):
    return 100 * value if value is not None else None


def write_tsv(path, headers, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(headers)
        writer.writerows(rows)


def summarize_validation_folds(run_dir, baseline_dir):
    """Refresh partial tables after each fold, then finalize mean/SD after all five."""
    run_dir, baseline_dir = Path(run_dir).resolve(), Path(baseline_dir).resolve()
    current = [read_fold(run_dir, fold, NEW_LEARNING_RATE) for fold in FOLDS]
    previous = [read_fold(baseline_dir, fold, OLD_LEARNING_RATE) for fold in FOLDS]
    for old, new in zip(previous, current):
        check_paired_settings(old, new)
    completed = sum(record['status'] == 'complete' for record in current)
    previous_completed = sum(record['status'] == 'complete' for record in previous)
    new_headers = ['fold', 'status', 'learning_rate', 'seed', 'best_epoch', 'validation_sites',
        'm6a_non_DRACH_support', 'inosine_support', 'm6a_non_DRACH_F1_percent',
        'inosine_F1_percent', 'validation_accuracy_percent', 'validation_macro_F1_percent']
    new_rows = [[record['fold'], record['status'], record['learning_rate'], record['seed'],
                 record['best_epoch'], record['validation_sites'], record['m6a_non_drach_support'],
                 record['inosine_support'], *[percent(record[key]) for key in METRICS]] for record in current]
    comparison_headers = ['fold', 'old_status', 'new_status', 'old_best_epoch', 'new_best_epoch',
        'old_m6a_non_DRACH_F1_percent', 'new_m6a_non_DRACH_F1_percent', 'm6a_delta_percentage_points',
        'old_inosine_F1_percent', 'new_inosine_F1_percent', 'inosine_delta_percentage_points']
    comparison_rows = []
    for old, new in zip(previous, current):
        values = []
        for metric in METRICS[:2]:
            before, after = old[metric], new[metric]
            delta = after - before if before is not None and after is not None else None
            values.extend([percent(before), percent(after), percent(delta)])
        comparison_rows.append([new['fold'], old['status'], new['status'], old['best_epoch'], new['best_epoch'], *values])
    summaries = {}
    for statistic in ('mean', 'sample_sd'):
        old_values, new_values = aggregate(previous, statistic), aggregate(current, statistic)
        summaries[statistic] = dict(old=old_values, new=new_values)
        new_rows.append([statistic, f'{completed}/5 complete', NEW_LEARNING_RATE,
                         None, None, None, None, None, *[percent(new_values[key]) for key in METRICS]])
        values = []
        for metric in METRICS[:2]:
            delta = None
            if completed == previous_completed == 5 and old_values[metric] is not None and new_values[metric] is not None:
                differences = [new[metric] - old[metric] for old, new in zip(previous, current)]
                delta = (mean if statistic == 'mean' else stdev)(differences)
            values.extend([percent(old_values[metric]), percent(new_values[metric]), percent(delta)])
        comparison_rows.append([statistic, f'{previous_completed}/5 complete', f'{completed}/5 complete', None, None, *values])
    notes = [
        ['Learning rates', 'Old=0.001; new=0.0003. Architecture, data and other training settings are held fixed.'],
        ['Metric', 'Validation only: non-DRACH m6A F1 and overall Inosine F1. No held-out test scores are read.'],
        ['Units', 'F1/accuracy displayed in percent; deltas are percentage points. JSON retains fractions.'],
        ['Mean/SD', 'Unweighted mean and sample SD (ddof=1) across all five completed, defined folds.'],
        ['SD interpretation', 'Fold SD describes CV split variation, not biological-replicate variation or a confidence interval.'],
        ['Incomplete runs', 'Missing and unfinished scores stay blank; five-fold means/SD are withheld until all five finish.'],
        ['Baseline', 'Existing learning-rate 0.001 results are preserved. Absent old folds are not fabricated or retrained.'],
        ['Interpretation', 'Operational weak-label internal grouped CV; not independently established biological accuracy.']]
    out = run_dir / 'summary'
    out.mkdir(parents=True, exist_ok=True)
    outputs = ['validation_folds.tsv', 'validation_comparison.tsv', 'validation_comparison.xlsx', 'summary.json']
    report = dict(status='complete' if completed == 5 else 'partial', completed_folds=completed,
        expected_folds=5, baseline_completed_folds=previous_completed,
        learning_rate=NEW_LEARNING_RATE, baseline_learning_rate=OLD_LEARNING_RATE,
        mean_sd_policy='unweighted five-fold mean and sample SD, ddof=1; undefined or incomplete stays null',
        evaluation='validation only; no test scores used',
        folds=[{key: value for key, value in record.items() if key not in ('run', 'settings')} for record in current],
        baseline_folds=[{key: value for key, value in record.items() if key not in ('run', 'settings')} for record in previous],
        statistics=summaries, output_files={name: str(out / name) for name in outputs})
    # A controller can refresh these while other folds train; readers never see a half-written file.
    with tempfile.TemporaryDirectory(prefix='.validation-summary-', dir=out) as temporary:
        temp = Path(temporary)
        write_tsv(temp / outputs[0], new_headers, new_rows)
        write_tsv(temp / outputs[1], comparison_headers, comparison_rows)
        excel_rows = [list(row) for row in new_rows]
        for row in excel_rows:
            row[2] = str(row[2])  # A two-decimal Excel float format would hide 0.0003.
        write_workbook(temp / outputs[2], [
            dict(name='Validation_folds', headers=new_headers, rows=excel_rows),
            dict(name='LR_comparison', headers=comparison_headers, rows=comparison_rows),
            dict(name='Notes', headers=['Item', 'Meaning'], rows=notes, widths=[24, 100])])
        (temp / outputs[3]).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        for name in outputs:
            os.replace(temp / name, out / name)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_validation_folds(args.run_dir, args.baseline_dir), indent=2))


if __name__ == '__main__':
    main()
