"""Summarize a completed frozen-model input-reliance experiment.

The four requested feature groups and the additional controls stay separate.
All changes are paired to the baseline on the same test cohort; their values
are not additive percentages of biological importance.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.io.excel import write_workbook, verify_workbook


CONDITIONS = ('sequence', 'repeat', 'neighbor_ml', 'quality',
              'sequence_repeat', 'center_ml', 'all_ml')
PRIMARY = frozenset(CONDITIONS[:4])
LABELS = {
    'sequence': 'Reference sequence',
    'repeat': 'Repeat annotation',
    'neighbor_ml': 'Neighboring modification scores',
    'quality': 'Basecall quality',
    'sequence_repeat': 'Sequence + repeat',
    'center_ml': 'Centre modification scores',
    'all_ml': 'All modification scores',
}
SCOPES = ('overall', 'm6A_all', 'm6A_DRACH', 'm6A_non_DRACH',
          'inosine_all', 'false_positive')
PANELS = (
    ('overall', 'macro_f1', 'Overall macro-F1'),
    ('m6A_DRACH', 'f1', 'm6A in DRACH'),
    ('m6A_non_DRACH', 'f1', 'm6A outside DRACH'),
    ('inosine_all', 'f1', 'Inosine'),
)
METRIC_FIELDS = ('condition', 'role', 'seed', 'scope', 'sites', 'support',
                 'tp', 'fp', 'fn', 'tn', 'accuracy', 'one_vs_rest_accuracy',
                 'precision', 'recall', 'f1', 'macro_f1', 'log_loss',
                 'class_flip_rate', 'mean_true_class_probability_change',
                 'mean_absolute_probability_change')
EFFECT_FIELDS = ('condition', 'role', 'scope', 'metric', 'unit', 'baseline',
                 'perturbed_mean', 'perturbed_sd', 'delta_mean', 'delta_sd',
                 'defined_repeats', 'requested_repeats', 'delta_definition')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_completed_run(directory):
    """Refuse unfinished runs, pilot cohorts and incomplete condition/seed sets."""
    path = Path(directory) / 'run.json'
    source = json.loads(path.read_text())
    if source.get('status') != 'complete':
        raise ValueError('Only a completed experiment can be summarized')
    cohort = source.get('cohort', {})
    if cohort.get('pilot') is not False or cohort.get('max_sites') not in (None, 0):
        raise ValueError('Pilot or sampled runs must not be exported as full-test paper results')
    sites = cohort.get('sites')
    if not isinstance(sites, int) or sites <= 0:
        raise ValueError('A positive, explicit full-test site count is required')
    if cohort.get('full_test_sites') != sites:
        raise ValueError('The exported cohort must equal the entire original test split')
    seeds = source.get('variant_seeds', [])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Repetition seeds must be nonempty and unique')
    evaluations = source.get('evaluations', [])
    baseline = [entry for entry in evaluations if entry.get('condition') == 'baseline']
    if len(baseline) != 1 or baseline[0].get('seed') is not None:
        raise ValueError('Exactly one unmodified baseline is required')
    expected = {(condition, seed) for condition in CONDITIONS for seed in seeds}
    actual = [(entry.get('condition'), entry.get('seed')) for entry in evaluations
              if entry.get('condition') != 'baseline']
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Every requested condition must have exactly one result per seed')
    for entry in evaluations:
        if entry['metrics']['overall']['sites'] != sites:
            raise ValueError('All conditions must use the identical full-test cohort')
    return source, path


def scope_metrics(evaluation, scope):
    """Use all classes in a motif subset when evaluating m6A one versus rest."""
    metrics = evaluation['metrics']
    if scope == 'overall':
        values = dict(metrics['overall'])
        for name in ('class_flip_rate', 'mean_true_class_probability_change',
                     'mean_absolute_probability_change'):
            values[name] = evaluation.get(name)
        return values
    class_name = {'m6A_all': 'm6A', 'm6A_DRACH': 'm6A',
                  'm6A_non_DRACH': 'm6A', 'inosine_all': 'inosine',
                  'false_positive': 'false_positive'}[scope]
    selected = (metrics['by_motif'][scope.removeprefix('m6A_')]
                if scope in ('m6A_DRACH', 'm6A_non_DRACH') else metrics['overall'])
    return dict(sites=selected['sites'], **selected['per_class'][class_name])


def metric_rows(source):
    """Keep per-repeat confusion counts and undefined ratios in a native table."""
    rows = []
    for entry in source['evaluations']:
        condition = entry['condition']
        role = ('baseline' if condition == 'baseline' else
                'primary' if condition in PRIMARY else 'control')
        for scope in SCOPES:
            values = scope_metrics(entry, scope)
            row = dict(condition=condition, role=role, seed=entry.get('seed'), scope=scope)
            row.update({key: values.get(key) for key in METRIC_FIELDS if key not in row})
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError('Non-finite metric: ' + key)
            rows.append(row)
    return rows


def describe_change(metric):
    """Declare scale and direction instead of normalizing effects to sum to100%."""
    if metric == 'log_loss':
        return 1.0, 'nats per site', 'perturbed minus baseline', 1
    if metric in ('class_flip_rate', 'mean_absolute_probability_change'):
        return 100.0, 'percentage points', 'perturbed minus baseline', 1
    return 100.0, 'percentage points', 'baseline minus perturbed', -1


def effect_rows(source, rows=None):
    rows = metric_rows(source) if rows is None else rows
    baseline = {row['scope']: row for row in rows if row['condition'] == 'baseline'}
    grouped = defaultdict(list)
    for row in rows:
        if row['condition'] != 'baseline':
            grouped[row['condition'], row['scope']].append(row)
    effects = []
    changes = ('accuracy', 'macro_f1', 'log_loss', 'class_flip_rate',
               'mean_true_class_probability_change', 'mean_absolute_probability_change')
    for condition in CONDITIONS:
        for scope in SCOPES:
            metrics = changes if scope == 'overall' else ('one_vs_rest_accuracy', 'precision', 'recall', 'f1')
            for metric in metrics:
                original = baseline[scope].get(metric)
                # Flip/probability changes are defined against an unchanged input.
                if metric in ('class_flip_rate', 'mean_true_class_probability_change',
                              'mean_absolute_probability_change'):
                    original = 0.0
                scale, unit, definition, direction = describe_change(metric)
                values = [row[metric] * scale for row in grouped[condition, scope]
                          if row.get(metric) is not None]
                mean = statistics.mean(values) if values else None
                sd = statistics.stdev(values) if len(values) > 1 else None
                old = original * scale if original is not None else None
                delta = direction * (mean - old) if mean is not None and old is not None else None
                effects.append(dict(condition=condition, role='primary' if condition in PRIMARY else 'control',
                    scope=scope, metric=metric, unit=unit, baseline=old,
                    perturbed_mean=mean, perturbed_sd=sd, delta_mean=delta,
                    delta_sd=sd if delta is not None else None,
                    defined_repeats=len(values), requested_repeats=len(source['variant_seeds']),
                    delta_definition=definition))
    return effects


def definitions(source):
    return [
        ['Evaluation', 'Frozen model on the complete held-out fold test split; operational weak labels, no reference or DRACH rescue.'],
        ['Fold', source.get('fold')],
        ['Sites', source['cohort']['sites']],
        ['Repetition seeds', ', '.join(map(str, source['variant_seeds']))],
        ['Primary groups', ', '.join(CONDITIONS[:4])],
        ['Controls', ', '.join(CONDITIONS[4:])],
        ['Sequence', 'Whole coherent 19nt window (17 overlapping3mer tokens) is exchanged between sites within retained-depth bins. This tests the reference branch; read MATCH/base features retain sequence information.'],
        ['Repeat', 'Whole19nt boolean repeat mask is exchanged; sequence+repeat uses a common donor for both fields.'],
        ['Neighbor ML', 'The joint vector of eight scores is resampled at real neighboring read positions, excluding center9, within observed-base, relative-offset and retained-depth strata.'],
        ['Central/all ML', 'The same replacement acts at center9 only or all19positions. These are separate controls, not neighboring-only effects.'],
        ['Quality', 'Only positive observed inverse-quality values are resampled within base/offset/depth strata; missing/absent zeros, read masks and depth remain fixed.'],
        ['Empirical donor replacement', 'Read-feature replacement uses bounded label-free empirical donor pools. Within-read/site correlations can be disrupted; this is not a perfect conditional distribution or a strict whole-field permutation.'],
        ['Metrics_by_repeat scale', 'Accuracy, precision, recall, F1 and probability changes use their original0–1 values; confusion counts are integers.'],
        ['Effects scale', 'Classification/probability changes are percentage points. Positive F1/accuracy delta means degradation after replacing that input. Negative values are retained.'],
        ['Loss', 'Natural-log multiclass loss; positive delta means increased loss.'],
        ['Uncertainty', 'SD is sample SD across donor randomizations for this one trained fold model, not SD across folds or a confidence interval.'],
        ['Motif metrics', 'DRACH and non-DRACH m6A F1 are one-versus-rest metrics using all classes in the respective motif subset. Inosine uses the complete test cohort.'],
        ['Interpretation', 'Model reliance depends on the chosen replacement, correlated inputs and this weak-label cohort; it does not establish chemical truth or causal biological feature effects.'],
        ['Non-additivity', 'Feature effects are not normalized to sum to100%; the four groups do not exhaust all inputs, and interactions/redundancy remain possible.'],
        ['Missing ratios', 'Undefined precision/recall/F1 stays empty; defined_repeats records how many seed results enter each mean and SD.'],
        ['Training/data', 'No model fitting, BAM extraction, saved-input rewriting or production-prediction changes are performed by this summarizer.'],
    ]


def write_tsv(path, rows, fields):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def draw_effects(effects, repeats, folder):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    indexed = {(row['condition'], row['scope'], row['metric']): row for row in effects}
    extent = []
    for scope, metric, _ in PANELS:
        for condition in CONDITIONS:
            row = indexed[condition, scope, metric]
            if row['delta_mean'] is not None:
                sd = row['delta_sd'] or 0.0
                extent.extend((row['delta_mean'] - sd, row['delta_mean'] + sd))
    low, high = min([0] + extent), max([0] + extent)
    padding = max(0.5, (high - low) * .18)
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.2), sharex=True)
    colors = ['#277f8e' if name in PRIMARY else '#8b9298' for name in CONDITIONS]
    labels = [LABELS[name] for name in CONDITIONS]
    ys = np.arange(len(CONDITIONS))
    for index, (ax, (scope, metric, title)) in enumerate(zip(axes.flat, PANELS)):
        selected = [indexed[name, scope, metric] for name in CONDITIONS]
        for y, row, color in zip(ys, selected, colors):
            if row['delta_mean'] is None:
                ax.text(0, y, 'undefined', fontsize=9, va='center')
                continue
            ax.barh(y, row['delta_mean'], color=color, height=.56, alpha=.85)
            if row['delta_sd'] is not None:
                ax.errorbar(row['delta_mean'], y, xerr=row['delta_sd'], fmt='none',
                            ecolor='#25313b', capsize=3, elinewidth=1)
            offset = max(.04, (high - low + 2 * padding) * .012)
            direction = 1 if row['delta_mean'] >= 0 else -1
            ax.text(row['delta_mean'] + direction * ((row['delta_sd'] or 0) + offset), y,
                    f"{row['delta_mean']:.2f}", va='center',
                    ha='left' if direction > 0 else 'right', fontsize=8, color='#26343e')
        ax.axvline(0, color='#26343e', linewidth=.9)
        ax.axhline(3.5, color='#b9bfc2', linestyle='--', linewidth=.7)
        ax.set(yticks=ys, yticklabels=labels, xlim=(low - padding, high + padding))
        ax.invert_yaxis()
        baseline = selected[0]['baseline']
        suffix = f'baseline {baseline:.2f}%' if baseline is not None else 'baseline undefined'
        ax.set_title(f'{chr(65 + index)}  {title}\n{suffix}', fontsize=12, weight='bold', loc='left')
        ax.tick_params(labelsize=9)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color='#e5e8ea', linewidth=.6)
        for edge in ('top', 'right'):
            ax.spines[edge].set_visible(False)
        ax.set_xlabel('F1 decrease from unchanged input (percentage points)', fontsize=9)
    fig.legend(handles=[Patch(color='#277f8e', label='Requested feature group'),
                        Patch(color='#8b9298', label='Additional control')],
               ncol=2, loc='lower center', bbox_to_anchor=(.5, .037), frameon=False)
    fig.text(.5, .012,
             f'Mean ± sample SD across {repeats} donor randomizations; one frozen model and the same complete test cohort.',
             ha='center', fontsize=9, color='#45545e')
    fig.subplots_adjust(left=.20, right=.985, top=.94, bottom=.15, hspace=.55, wspace=.98)
    files = []
    for suffix in ('.svg', '.pdf', '.png'):
        path = folder / ('Feature_reliance_F1' + suffix)
        fig.savefig(path, dpi=220, bbox_inches='tight')
        files.append(path)
    plt.close(fig)
    return files


def export(run_directory, output, figures):
    source, source_path = read_completed_run(run_directory)
    output, figures = Path(output), Path(figures)
    if output.exists() or figures.exists():
        raise FileExistsError('Use fresh output and figure directories; preserve earlier summaries')
    source_sha = sha256(source_path)
    rows = metric_rows(source)
    effects = effect_rows(source, rows)
    baseline = [row for row in rows if row['condition'] == 'baseline']
    repeated = [row for row in rows if row['condition'] != 'baseline']
    notes = definitions(source)
    output.mkdir(parents=True)
    figures.mkdir(parents=True)
    sheets = [
        dict(name='Effects', headers=list(EFFECT_FIELDS),
             rows=[[row.get(key) for key in EFFECT_FIELDS] for row in effects]),
        dict(name='Metrics_by_repeat', headers=list(METRIC_FIELDS),
             rows=[[row.get(key) for key in METRIC_FIELDS] for row in repeated]),
        dict(name='Baseline', headers=list(METRIC_FIELDS),
             rows=[[row.get(key) for key in METRIC_FIELDS] for row in baseline]),
        dict(name='Definitions', headers=['Item', 'Definition'], rows=notes, widths=[29, 110]),
    ]
    fold = source.get('fold')
    book = output / f'Feature_reliance_fold{fold}_20261109.xlsx'
    write_workbook(book, sheets, float_format='0.0000')
    actual = verify_workbook(book)
    expected = {sheet['name']: [sheet['headers']] + sheet['rows'] for sheet in sheets}
    if actual != expected:
        raise ValueError('Native Excel cells do not match the source values')
    write_tsv(output / 'effects.tsv', effects, EFFECT_FIELDS)
    write_tsv(output / 'metrics_by_repeat.tsv', repeated, METRIC_FIELDS)
    write_tsv(output / 'baseline.tsv', baseline, METRIC_FIELDS)
    files = draw_effects(effects, len(source['variant_seeds']), figures)
    (output / 'source_run.json').write_bytes(source_path.read_bytes())
    if sha256(source_path) != source_sha:
        raise ValueError('Completed source run changed during summarization')
    (output / 'README.txt').write_text('\n'.join(f'{name}: {value}' for name, value in notes) + '\n')
    report = dict(status='complete', source_run=str(source_path.resolve()), source_sha256=source_sha,
        fold=source.get('fold'), sites=source['cohort']['sites'],
        variant_seeds=source['variant_seeds'], conditions=list(CONDITIONS),
        workbook=str(book.resolve()), workbook_sha256=sha256(book), worksheet_names=list(actual),
        native_cells_verified=True,
        figures=[dict(path=str(path.resolve()), sha256=sha256(path)) for path in files],
        uncertainty='Sample SD across donor randomizations, not folds or confidence intervals',
        interpretation='Frozen-model reliance under the specified input replacement on operational weak labels',
        old_outputs_changed=False, training_performed=False, source_unchanged=True)
    (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Completed full-test experiment directory')
    parser.add_argument('--output', type=Path, required=True, help='Fresh paper data directory')
    parser.add_argument('--figures', type=Path, required=True, help='Fresh scientific-figure directory')
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.output, args.figures), indent=2))


if __name__ == '__main__':
    main()
