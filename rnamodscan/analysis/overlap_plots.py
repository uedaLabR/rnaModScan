"""Venn diagrams of reference matches within called Dorado/modkit sites."""
import csv
import hashlib
import json
import re
from pathlib import Path

CATEGORIES = ('short_read_only', 'both', 'database_only', 'neither')
IDENTITY = ('sample_id', 'threshold', 'modification', 'motif_group')
SCOPE = 'Dorado/modkit sites passing the specified call thresholds'
FORMATS = ('png', 'svg', 'pdf')


def require_plotting():
    """Import before a long pileup run, so a missing dependency fails early."""
    try:
        import matplotlib
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.patches import Circle, Rectangle
    except ImportError as exc:
        raise RuntimeError(
            'Venn plotting requires matplotlib. Run: python -m pip install -r '
            'analysis/pileup_overlap/requirements-plotting.txt; '
            'or set WRITE_VENN_PLOTS=False to produce tables only.'
        ) from exc
    return matplotlib, Figure, FigureCanvasAgg, Circle, Rectangle


def read_partitions(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        required = set(IDENTITY + CATEGORIES + ('cell', 'reference_cell', 'condition',
                       'call_sites', 'reference_status', 'min_depth', 'min_modified', 'min_af'))
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing columns in reference_overlap_summary.tsv')
        rows, seen = [], set()
        for row in reader:
            key = tuple(row[k] for k in IDENTITY)
            if not all(key) or key in seen:
                raise ValueError(f'Duplicate or empty plot identity: {key}')
            seen.add(key)
            for field in ('call_sites',):
                row[field] = int(row[field])
                if row[field] < 0:
                    raise ValueError('Counts must be nonnegative')
            if row['reference_status'] == 'ready':
                for field in CATEGORIES:
                    row[field] = int(row[field])
                    if row[field] < 0:
                        raise ValueError('Counts must be nonnegative')
                if sum(row[field] for field in CATEGORIES) != row['call_sites']:
                    raise ValueError(f'Partition counts do not sum to call_sites: {key}')
            elif row['reference_status'] != 'incomplete_reference':
                raise ValueError(f"Unknown reference status: {row['reference_status']}")
            rows.append(row)
    if not rows:
        raise ValueError('Empty reference_overlap_summary.tsv')
    return rows


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _figure(row, db_label, plotting):
    _, Figure, FigureCanvasAgg, Circle, Rectangle = plotting
    fig = Figure(figsize=(9.2, 6.8), facecolor='white')
    FigureCanvasAgg(fig)
    ax = fig.add_axes([.06, .19, .88, .65])
    ax.set(xlim=(0, 10), ylim=(0, 5.5), aspect='equal')
    ax.axis('off')
    ax.add_patch(Rectangle((.10, .10), 9.8, 5.15, facecolor='#FAFBFC',
                           edgecolor='#9BA8B5', linewidth=1))
    ax.text(.35, 4.93, f"Dorado/modkit called sites: {row['call_sites']:,}",
            fontsize=12, weight='bold', color='#243647', va='center')
    ax.add_patch(Circle((3.85, 2.95), 1.65, facecolor='#4E9AC7',
                        edgecolor='#2B698F', alpha=.38, linewidth=1.5))
    ax.add_patch(Circle((6.15, 2.95), 1.65, facecolor='#EDA860',
                        edgecolor='#A46121', alpha=.38, linewidth=1.5))
    ax.text(2.6, 4.55, 'Short-read', ha='center', fontsize=12, weight='bold', color='#245B7C')
    ax.text(7.4, 4.55, db_label, ha='center', fontsize=12, weight='bold', color='#86501C')

    def annotation(x, field, title):
        n = row[field]
        fraction = f"{100 * n / row['call_sites']:.1f}%" if row['call_sites'] else 'n/a'
        ax.text(x, 3.35, title, ha='center', va='center', fontsize=9, color='#344452')
        ax.text(x, 2.93, f'{n:,}', ha='center', va='center', fontsize=15, weight='bold', color='#172B3A')
        ax.text(x, 2.52, fraction, ha='center', va='center', fontsize=10, color='#344452')

    annotation(3.05, 'short_read_only', 'Short-read only')
    annotation(5, 'both', 'Both')
    annotation(6.95, 'database_only', 'Database only')
    neither = row['neither']
    fraction = f"{100 * neither / row['call_sites']:.1f}%" if row['call_sites'] else 'n/a'
    ax.text(5, .62, f'Neither reference: {neither:,} ({fraction})',
            ha='center', va='center', fontsize=12, weight='bold', color='#344452')
    sample_title = str(row['sample_id'])
    fig.text(.5, .955, f"{sample_title}  |  {row['modification']}  |  {row['motif_group']}",
             ha='center', va='top', fontsize=15, weight='bold', wrap=True, color='#172B3A')
    fig.text(.5, .903,
             f"{row['condition']}  |  short-read cell: {row['reference_cell']}  |  threshold: {row['threshold']}",
             ha='center', va='top', fontsize=10, color='#455767', wrap=True)
    fig.text(.5, .867,
             f"Depth >= {row['min_depth']}   |   modified reads >= {row['min_modified']}   |   AF >= {row['min_af']}",
             ha='center', fontsize=10, color='#455767')
    notes = [
        'Universe: threshold-passing Dorado/modkit calls; percentages use this total.',
        'Short-read = union of methods for the stated cell. Database = registered reference union.',
        'Schematic Venn diagram; circle areas are not proportional to counts.',
        'No reference match does not establish a false positive.',
    ]
    if row['condition'] == 'IVT':
        notes.append('IVT calls are counted as operational false calls, including reference matches.')
    if not row['call_sites']:
        notes.append('No calls pass this threshold; percentages are undefined.')
    fig.text(.08, .165, '\n'.join(notes), fontsize=8.5, va='top',
             linespacing=1.55, color='#455767')
    return fig


def generate_venn_plots(summary_dir, output_dir=None, *, formats=FORMATS, dpi=250):
    """Read only the completed small summary table and run.json, never pileup/FASTA."""
    summary_dir = Path(summary_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else summary_dir / 'venn'
    if output_dir.exists():
        raise FileExistsError(f'Choose a new Venn output directory: {output_dir}')
    for marker in ('INCOMPLETE.txt', 'FAILED.txt'):
        if (summary_dir / marker).exists():
            raise ValueError(f'Aggregation is incomplete: {summary_dir / marker}')
    run_path = summary_dir / 'run.json'
    report = json.loads(run_path.read_text(encoding='utf-8'))
    if report.get('status') != 'complete':
        raise ValueError('run.json must have status=complete')
    table = summary_dir / 'reference_overlap_summary.tsv'
    rows = read_partitions(table)
    formats = tuple(formats)
    if not formats or len(formats) != len(set(formats)) or not set(formats).issubset(FORMATS):
        raise ValueError('Use unique formats chosen from png, svg, pdf')
    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError('DPI must be a positive integer')
    plotting = require_plotting()
    matplotlib = plotting[0]
    records = []
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'INCOMPLETE.txt').write_text('Plotting is incomplete.', encoding='utf-8')
    try:
        with matplotlib.rc_context({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                                    'pdf.fonttype': 42}):
            for index, row in enumerate(rows, 1):
                record = {key: row[key] for key in IDENTITY}
                if row['reference_status'] != 'ready':
                    records.append(dict(record, status='skipped', reason='incomplete_reference', files=[]))
                    continue
                matching = [s for s in report.get('references', [])
                            if s.get('kind') == 'database' and s.get('modification') == row['modification']
                            and s.get('cell') in ('*', row['reference_cell'])]
                label = ('RMBase 3.0' if matching and all(
                    s.get('database') == 'RMBase' and s.get('version') == '3.0' for s in matching)
                         else 'Database')
                name = '__'.join(re.sub(r'[^A-Za-z0-9._-]+', '_', str(row[key]))[:60] for key in IDENTITY)
                stem = f'{index:03d}__{name}'
                fig = _figure(row, label, plotting)
                paths = []
                try:
                    for fmt in formats:
                        filename = f'{stem}.{fmt}'
                        fig.savefig(output_dir / filename, format=fmt, dpi=dpi, facecolor='white')
                        paths.append(filename)
                finally:
                    fig.clear()
                records.append(dict(record, status='complete', files=paths,
                                    call_sites=row['call_sites'],
                                    counts={k: row[k] for k in CATEGORIES}))
        result = dict(status='complete', scope=SCOPE, area_proportional=False,
                      source_table=str(table), source_table_sha256=_sha256(table),
                      source_run_sha256=_sha256(run_path), matplotlib_version=matplotlib.__version__,
                      formats=list(formats), dpi=dpi, plots=records,
                      generated=sum(p['status'] == 'complete' for p in records),
                      skipped=sum(p['status'] == 'skipped' for p in records))
        (output_dir / 'plots.json').write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        (output_dir / 'INCOMPLETE.txt').unlink()
        print(f"Venn: {result['generated']} figures, {result['skipped']} skipped; {output_dir}", flush=True)
        return result
    except BaseException as exc:
        (output_dir / 'FAILED.txt').write_text(str(exc), encoding='utf-8')
        raise
