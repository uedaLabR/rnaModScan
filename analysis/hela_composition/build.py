"""Summarize completed HeLa calls; no BAM access or inference is performed."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.io.excel import verify_workbook, write_workbook

M6A = ROOT / 'paper/data/hela_m6a_comparison_20261108'
INO = ROOT / 'paper/data/hela_inosine_comparison_20261108'
DEFAULT_DATA = ROOT / 'paper/data/hela_composition_20261109'
DEFAULT_FIGURES = ROOT / 'paper/figures/hela_composition_20261109'


def read_table(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def write_table(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def m6a_counts():
    sizes = read_table(M6A / 'HeLa_set_sizes.tsv')
    transitions = read_table(M6A / 'HeLa_filter_transitions.tsv')

    def count(method, stage, region):
        selected = [row for row in sizes if row['method'] == method and row['stage'] == stage
                    and row['region'] == region and row['universe'] == 'full_catalog'
                    and row['name'] == region + '__' + stage + '__full_catalog']
        if len(selected) != 1:
            raise ValueError('Ambiguous source count: ' + str((method, stage, region)))
        return int(selected[0]['sites'])

    unknown = next(int(row['dorado']) for row in transitions if row['region'] == 'unknown')
    rows = []
    for group, method, stage, missing in [('GLORI', 'GLORI', 'NN', 0),
            ('Dorado/modkit', 'rnaModScan', 'Dorado', unknown),
            ('rnaModScan NN', 'rnaModScan', 'NN', 0)]:
        drach = count(method, stage, 'DRACH')
        non_drach = count(method, stage, 'non_DRACH')
        known = drach + non_drach
        rows.append(dict(group=group, total_candidates=known + missing,
                         known_context_sites=known, unknown_context_sites=missing,
                         DRACH_sites=drach, non_DRACH_sites=non_drach,
                         DRACH_percent=100 * drach / known,
                         non_DRACH_percent=100 * non_drach / known))
    expected = [(129124, 35637, 0), (82835, 375643, 1), (79699, 26874, 0)]
    if [(r['DRACH_sites'], r['non_DRACH_sites'], r['unknown_context_sites'])
            for r in rows] != expected:
        raise ValueError('Historical m6A source counts changed')
    return rows


def inosine_counts():
    summary = read_table(INO / 'inosine_repeat_summary.tsv')
    names = [('Slic-seq', 'Slic-seq_all', 'Slic_seq'),
             ('Dorado/modkit', 'Dorado_all', 'Dorado'),
             ('rnaModScan NN', 'NN_all', 'NN')]
    audit = {key: dict(total_sites=0, any_repeat_sites=0, any_Alu_sites=0)
             for _, _, key in names}
    with (INO / 'inosine_membership.tsv').open(newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            for _, _, key in names:
                if row[key] == '1':
                    audit[key]['total_sites'] += 1
                    audit[key]['any_repeat_sites'] += int(row['any_repeat'])
                    audit[key]['any_Alu_sites'] += int(row['any_Alu'])
    rows = []
    for label, source_group, key in names:
        selected = next(row for row in summary if row['group'] == source_group)
        for name, value in audit[key].items():
            if int(selected[name]) != value:
                raise ValueError('Repeat membership/summary mismatch: ' + source_group)
        total = audit[key]['total_sites']
        repeat = audit[key]['any_repeat_sites']
        alu = audit[key]['any_Alu_sites']
        rows.append(dict(group=label, total_sites=total, any_repeat_sites=repeat,
                         any_Alu_sites=alu, other_repeat_sites=repeat - alu,
                         non_repeat_sites=total - repeat,
                         unknown_annotation_sites=int(selected['unknown_annotation_sites']),
                         any_repeat_percent=100 * repeat / total,
                         any_Alu_percent=100 * alu / total))
    return rows, audit


def style_axis(ax):
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel('Fraction of sites (%)', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=.16)


def draw_m6a(ax, rows, letter):
    import numpy as np
    y = np.arange(len(rows))
    drach = [r['DRACH_percent'] for r in rows]
    non_drach = [r['non_DRACH_percent'] for r in rows]
    ax.barh(y, drach, color='#478DB3', height=.55, label='DRACH')
    ax.barh(y, non_drach, left=drach, color='#DAB678', height=.55, label='Non-DRACH')
    for i, row in enumerate(rows):
        a, b = row['DRACH_percent'], row['non_DRACH_percent']
        ax.text(a / 2, i, f'{a:.1f}%', ha='center', va='center', color='white', fontsize=10, weight='bold')
        ax.text(a + b / 2, i, f'{b:.1f}%', ha='center', va='center', fontsize=10, color='#172B3A')
    labels = [r['group'] + '\nn = ' + format(r['known_context_sites'], ',') for r in rows]
    ax.set_yticks(y, labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_ylim(2.6, -.65)
    style_axis(ax)
    ax.set_title('m6A sequence context', fontsize=13, pad=18, weight='bold')
    ax.text(-.33, 1.14, letter, transform=ax.transAxes, fontsize=17, weight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(.50, -.23), ncol=2, frameon=False, fontsize=9)


def draw_inosine(ax, rows, letter):
    import numpy as np
    y = np.arange(len(rows))
    for column, offset, color, label in [('any_repeat_percent', -.16, '#478DB3', 'Any repeat'),
                                        ('any_Alu_percent', .16, '#725A90', 'Alu')]:
        values = [row[column] for row in rows]
        ax.barh(y + offset, values, height=.29, color=color, label=label)
        for i, value in enumerate(values):
            ax.text(value - 1.4 if value >= 20 else value + 1.4, i + offset,
                    f'{value:.1f}%', va='center', ha='right' if value >= 20 else 'left',
                    fontsize=9, color='white' if value >= 20 else '#172B3A', weight='bold')
    ax.set_yticks(y, [r['group'] + '\nn = ' + format(r['total_sites'], ',') for r in rows], fontsize=10)
    ax.invert_yaxis()
    ax.set_ylim(2.6, -.65)
    style_axis(ax)
    ax.set_title('Inosine repeat association', fontsize=13, pad=18, weight='bold')
    ax.text(-.33, 1.14, letter, transform=ax.transAxes, fontsize=17, weight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(.50, -.23), ncol=2, frameon=False, fontsize=9)


def save_figure(fig, stem):
    for extension in ['svg', 'pdf', 'png']:
        fig.savefig(str(stem) + '.' + extension, dpi=300, facecolor='white')


def figures(m6a, inosine, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with matplotlib.rc_context({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
        fig.subplots_adjust(left=.13, right=.98, top=.79, bottom=.28, wspace=.61)
        draw_m6a(axes[0], m6a, 'A')
        draw_inosine(axes[1], inosine, 'B')
        fig.text(.035, .035, 'HeLa WT; raw NN decisions before reference rescue. One Dorado m6A site with unknown context is excluded from panel A.\n'
                 'Alu is a subset of any repeat; panel B categories are not additive. Composition is descriptive, not independent accuracy.',
                 fontsize=8, color='#425666', va='bottom', linespacing=1.5)
        save_figure(fig, output / 'Fig_HeLa_candidate_composition')
        plt.close(fig)
        for name, draw, rows, letter in [('m6A', draw_m6a, m6a, 'C'),
                                         ('inosine', draw_inosine, inosine, 'D')]:
            fig, ax = plt.subplots(figsize=(6.4, 4.4))
            fig.subplots_adjust(left=.25, right=.98, top=.81, bottom=.27)
            draw(ax, rows, letter)
            save_figure(fig, output / ('Panel_HeLa_' + name + '_composition'))
            plt.close(fig)


def figure4_composition(m6a, inosine, output):
    """C/D panels for appending below the unchanged Figure4 Venn drawing."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with matplotlib.rc_context({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
        fig.subplots_adjust(left=.13, right=.98, top=.79, bottom=.24, wspace=.61)
        draw_m6a(axes[0], m6a, 'C')
        draw_inosine(axes[1], inosine, 'D')
        save_figure(fig, output / 'Fig4C_D_candidate_composition')
        plt.close(fig)


def add_resource_book(data):
    """Keep the original composition workbook and add a four-sheet companion."""
    original = data / 'archive_initial_three_sheet/HeLa_candidate_composition_20261109.xlsx'
    restored = verify_workbook(original)
    sheets = [dict(name=name, headers=values[0], rows=values[1:])
              for name, values in restored.items()]
    source = ROOT / 'paper/data/runtime_comments_20261109/runtime_table.tsv'
    rows = read_table(source)
    for row in rows:
        for key in ['wall_minutes', 'host_ram_gib', 'gpu_vram_gib']:
            row[key] = float(row[key]) if row[key] else None
    sheets.append(dict(name='ResourceTableS18', headers=list(rows[0]),
                       rows=[list(row.values()) for row in rows]))
    target = data / 'HeLa_composition_and_resources_20261109.xlsx'
    if target.exists():
        raise FileExistsError(target)
    write_workbook(target, sheets, float_format='0.0000')
    values = verify_workbook(target)
    for sheet in sheets:
        if values[sheet['name']] != [sheet['headers']] + sheet['rows']:
            raise ValueError('Companion Excel native-value mismatch')
    return target


def build(data, figure_dir):
    data.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    m6a = m6a_counts()
    inosine, audit = inosine_counts()
    write_table(data / 'm6A_context_composition.tsv', m6a)
    write_table(data / 'inosine_repeat_composition.tsv', inosine)
    notes = [
        ['Scope', 'HeLa WT; original modification-qualified raw NN argmax candidates before DB/short-read/DRACH rescue.'],
        ['Dorado candidates', 'Existing depth/AF-qualified Dorado/modkit candidates before NN filtering; not unthresholded basecaller output.'],
        ['m6A denominator', 'DRACH plus non-DRACH sites with known five-base context; exclude one unknown-context Dorado site from fractions only.'],
        ['Inosine denominator', 'All sites in each named group. Any Alu is included in any repeat; the two fractions are not additive.'],
        ['Repeat assignment', 'Existing UCSC hg38 RepeatMasker center-position half-open interval overlaps; repeat strand ignored.'],
        ['Interpretation', 'Motif/repeat composition is descriptive. NN uses sequence and repeat information; these shifts are not independent accuracy or a background-controlled enrichment analysis.'],
        ['Source m6A counts', str((M6A / 'HeLa_set_sizes.tsv').relative_to(ROOT))],
        ['Source unknown context', str((M6A / 'HeLa_filter_transitions.tsv').relative_to(ROOT))],
        ['Source inosine counts', str((INO / 'inosine_repeat_summary.tsv').relative_to(ROOT))],
        ['Independent repeat reaggregation', str((INO / 'inosine_membership.tsv').relative_to(ROOT))]
    ]
    sheets = [dict(name='m6A_context', headers=list(m6a[0]), rows=[list(r.values()) for r in m6a]),
              dict(name='Inosine_repeat', headers=list(inosine[0]), rows=[list(r.values()) for r in inosine]),
              dict(name='Scope_and_sources', headers=['item', 'description'], rows=notes, widths=[30, 115])]
    archive = data / 'archive_initial_three_sheet'
    archive.mkdir()
    workbook = archive / 'HeLa_candidate_composition_20261109.xlsx'
    write_workbook(workbook, sheets, float_format='0.0000')
    restored = verify_workbook(workbook)
    for sheet in sheets:
        if restored[sheet['name']] != [sheet['headers']] + sheet['rows']:
            raise ValueError('Excel values did not round-trip exactly')
    figures(m6a, inosine, figure_dir)
    figure4_composition(m6a, inosine, figure_dir)
    add_resource_book(data)
    sources = [M6A / 'HeLa_set_sizes.tsv', M6A / 'HeLa_filter_transitions.tsv',
               INO / 'inosine_repeat_summary.tsv', INO / 'inosine_membership.tsv']
    run = dict(status='complete', scope='Existing-result descriptive composition only; no BAM/inference rerun',
               m6A=m6a, inosine=inosine, membership_reaggregation=audit,
               workbook=str((data / 'HeLa_composition_and_resources_20261109.xlsx').relative_to(ROOT)),
               workbook_sheets=4, archived_initial_workbook=str(workbook.relative_to(ROOT)),
               source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (data / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    (data / 'figure_caption_en.txt').write_text(
        'HeLa WT candidate composition before reference rescue. (A) DRACH and non-DRACH fractions in GLORI, '
        'depth/AF-qualified Dorado/modkit candidates before NN filtering, and raw rnaModScan m6A NN calls. '
        'Fractions use known-context denominators of 164,761, 458,478 and 106,573 sites, respectively; '
        'one unknown-context Dorado candidate is omitted from these fractions. (B) Fractions of Slic-seq, '
        'Dorado/modkit and raw rnaModScan inosine NN sites overlapping any RepeatMasker interval or an Alu family. '
        'Alu is a subset of repeat, so the two bars are not additive. All comparisons use unique sites in the '
        'input GRCh38 assembly. Composition shifts are descriptive and do not constitute independently '
        'validated accuracy or background-controlled enrichment.\n')
    print(workbook)
    print(figure_dir / 'Fig_HeLa_candidate_composition.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--figures', type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    build(args.data, args.figures)
