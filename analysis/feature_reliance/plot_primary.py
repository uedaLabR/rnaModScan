"""Add a readable primary-group view without changing the all-condition report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


PRIMARY = ('sequence', 'repeat', 'neighbor_ml', 'quality')
LABELS = ('Reference sequence', 'Repeat annotation',
          'Neighboring modification scores', 'Basecall quality')
PANELS = (('overall', 'macro_f1', 'Overall macro-F1'),
          ('m6A_DRACH', 'f1', 'm6A in DRACH'),
          ('m6A_non_DRACH', 'f1', 'm6A outside DRACH'),
          ('inosine_all', 'f1', 'Inosine'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def primary_table(data):
    """Add a compact exact-value view; all values also remain in the workbook."""
    data = Path(data)
    path = data / 'primary_results.tsv'
    report_path = data / 'run.json'
    report = json.loads(report_path.read_text())
    if path.exists():
        if report.get('primary_results_table', {}).get('sha256') != sha(path):
            raise ValueError('An existing primary table changed outside this helper')
        return path
    with (data / 'effects.tsv').open() as handle:
        indexed = {(row['condition'], row['scope'], row['metric']): row
                   for row in csv.DictReader(handle, delimiter='\t')}
    fields = ['condition', 'feature_group']
    for scope, _, _ in PANELS:
        fields.extend(scope + suffix for suffix in ('_baseline_F1_percent', '_decrease_mean_pp', '_decrease_sd_pp'))
    rows = []
    for condition, label in zip(PRIMARY, LABELS):
        row = dict(condition=condition, feature_group=label)
        for scope, metric, _ in PANELS:
            values = indexed[condition, scope, metric]
            row[scope + '_baseline_F1_percent'] = float(values['baseline'])
            row[scope + '_decrease_mean_pp'] = float(values['delta_mean'])
            row[scope + '_decrease_sd_pp'] = float(values['delta_sd']) if values['delta_sd'] else None
        rows.append(row)
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    report['primary_results_table'] = dict(path=str(path.resolve()), sha256=sha(path),
        rows=4, units='F1 decrease in percentage points, mean and sample SD over donor randomizations',
        numerical_source='Unchanged Effects worksheet and effects.tsv')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return path


def plot(data, figures, refresh=False):
    data, figures = Path(data), Path(figures)
    report_path = data / 'run.json'
    report = json.loads(report_path.read_text())
    if report.get('status') != 'complete':
        raise ValueError('A completed full-test summary is required')
    paths = [figures / ('Feature_reliance_primary_F1' + suffix)
             for suffix in ('.svg', '.pdf', '.png')]
    if any(path.exists() for path in paths):
        if not refresh:
            raise FileExistsError('Preserve the existing primary-group plots unless --refresh is explicit')
        recorded = {entry['path']: entry['sha256'] for entry in report['figures']}
        for path in paths:
            if not path.exists() or recorded.get(str(path.resolve())) != sha(path):
                raise ValueError('Only unchanged plots from this helper can be refreshed')
        archive = figures / 'archive_primary_initial_labels'
        archive.mkdir(exist_ok=False)
        for path in paths:
            path.rename(archive / path.name)
        report['figures'] = [entry for entry in report['figures']
                             if entry['path'] not in {str(path.resolve()) for path in paths}]
    effects = data / 'effects.tsv'
    with effects.open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    lookup = {(row['condition'], row['scope'], row['metric']): row for row in rows}
    selected = [[lookup[condition, scope, metric] for condition in PRIMARY]
                for scope, metric, _ in PANELS]
    for panel in selected:
        if any(row['unit'] != 'percentage points' or
               row['delta_definition'] != 'baseline minus perturbed' for row in panel):
            raise ValueError('Expected F1 decrease in percentage points')

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 6.8))
    for panel_index, (ax, panel, (_, _, title)) in enumerate(zip(axes.flat, selected, PANELS)):
        means = np.array([float(row['delta_mean']) for row in panel])
        deviations = np.array([float(row['delta_sd']) if row['delta_sd'] else 0.0 for row in panel])
        baseline = float(panel[0]['baseline'])
        if any(float(row['baseline']) != baseline for row in panel):
            raise ValueError('Every condition must share the same paired baseline')
        low, high = min(0., float(np.min(means - deviations))), max(0., float(np.max(means + deviations)))
        span = max(high - low, .5)
        padding = max(.35, .23 * span)
        positions = np.arange(len(PRIMARY))
        ax.barh(positions, means, height=.55, color='#277f8e', alpha=.86)
        ax.errorbar(means, positions, xerr=deviations, fmt='none', color='#243641',
                    capsize=3, elinewidth=1)
        for y, mean, sd in zip(positions, means, deviations):
            # Small negative values otherwise put the label across the y-axis
            # feature name. Their minus sign is retained, to the right of zero.
            label_x = mean + sd + .018 * span if mean >= 0 else .018 * span
            ax.text(label_x, y,
                    f'{mean:.2f} ± {sd:.2f}', va='center',
                    ha='left', fontsize=8.5)
        ax.axvline(0, color='#273b48', linewidth=.8)
        ax.set(yticks=positions, yticklabels=LABELS, xlim=(low - padding, high + padding))
        ax.invert_yaxis()
        ax.tick_params(labelsize=9)
        ax.set_title(f'{chr(65 + panel_index)}  {title}\nbaseline {baseline:.2f}%',
                     fontsize=11.5, fontweight='bold', loc='left')
        ax.set_xlabel('F1 decrease (percentage points)', fontsize=9)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color='#e5e8ea', linewidth=.6)
        for edge in ('top', 'right'):
            ax.spines[edge].set_visible(False)
    repeats = len(report['variant_seeds'])
    fig.text(.5, .018,
        f'Panel-specific x scales. Mean ± sample SD across {repeats} donor randomizations; '
        f'fold {report["fold"]}, {report["sites"]:,} held-out observations with operational weak labels.',
        ha='center', fontsize=8.5, color='#45545e')
    fig.subplots_adjust(left=.20, right=.985, top=.93, bottom=.13, hspace=.65, wspace=.99)
    figures.mkdir(parents=True, exist_ok=True)
    for path in paths:
        fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    files = [dict(path=str(path.resolve()), sha256=sha(path)) for path in paths]
    report['figures'].extend(files)
    report['primary_group_view'] = dict(
        source_effects_sha256=sha(effects), conditions=list(PRIMARY),
        panel_x_scales='independent and explicitly labelled',
        unchanged='Original all-condition plots, workbook and numerical TSV files were not rewritten')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    primary_table(data)
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--figures', type=Path, required=True)
    parser.add_argument('--refresh', action='store_true', help='Archive and replace only this helper\'s unchanged prior plots')
    args = parser.parse_args()
    print(json.dumps(plot(args.data, args.figures, args.refresh), indent=2))
