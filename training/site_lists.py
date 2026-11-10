"""Disk-backed v2 site lists. Labels are operational evidence, not ground truth."""
import csv
import json
import sqlite3
from collections import Counter
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path

from utils.io.artifacts import file_info
from utils.io.bed import bedmethyl_rows, passes_thresholds as is_candidate
from utils.pileup.provenance import read_pileup_log
from utils.references.catalog import ReferenceCatalog
from training.site_summary import summarize_site_list

TARGETS = ('m6A', 'm5C', 'inosine', 'pseU')
FIELDS = ['row_id', 'sample_id', 'cell', 'reference_cell', 'condition', 'chrom',
          'start', 'end', 'strand', 'modification', 'total_reads', 'modified_reads',
          'af', 'short_read_sources', 'database_sources', 'positive_evidence',
          'paired_ivt_depth', 'paired_ivt_modified_reads', 'paired_ivt_af',
          'ivt_comparison', 'label', 'label_reason', 'label_status', 'group_id']



def ivt_override(wt, ivt, min_modified=3, af_ratio=0.6):
    """Inclusive comparison using integer arithmetic, independent of IVT depth cutoff."""
    if ivt is None or ivt['total_reads'] == 0 or wt['total_reads'] == 0:
        return False
    ratio = Fraction(str(af_ratio))
    return (ivt['modified_reads'] >= min_modified and
            ivt['modified_reads'] * wt['total_reads'] * ratio.denominator >=
            wt['modified_reads'] * ivt['total_reads'] * ratio.numerator)


def wt_label(wt, ivt, ivt_available, min_modified=3, af_ratio=0.6):
    if not ivt_available:
        return '', 'awaiting_ivt', 'pending'
    if ivt_override(wt, ivt, min_modified, af_ratio):
        return 0, 'ivt_mod_reads_and_af_ratio', 'final'
    if wt['short_read_sources'] or wt['database_sources']:
        reason = ('short_read_and_database_overlap' if wt['short_read_sources'] and wt['database_sources']
                  else 'short_read_overlap' if wt['short_read_sources'] else 'database_overlap')
        return 1, reason, 'final'
    return '', 'no_positive_evidence', 'unlabeled'



def _ingest(db, sample, condition, source, log, catalog, thresholds, wt_sample=None,
            progress_every=1000000):
    """Stream complete pileup; only retain candidates and WT comparison positions."""
    stats = Counter()
    kinds = {s['id']: s['kind'] for s in catalog.sources}
    source_before = file_info(source)
    last_chrom = None
    wt_keys, batch = set(), []
    for row in bedmethyl_rows(source):
        chrom, pos = row['chrom'], row['start']
        if chrom != last_chrom:
            last_chrom = chrom
            if condition == 'IVT':
                wt_keys = {(r[0], r[1], r[2]) for r in db.execute(
                    'SELECT start,strand,modification FROM observations WHERE sample_id=? AND chrom=?',
                    (wt_sample, chrom))}
        stats['input_rows'] += 1
        if progress_every and stats['input_rows'] % progress_every == 0:
            print(f"{sample['sample_id']} {condition}: {stats['input_rows']:,} rows", flush=True)
        if row['modification'] not in TARGETS:
            continue
        stats['target_rows'] += 1
        candidate = is_candidate(row, **thresholds)
        if not candidate and (pos, row['strand'], row['modification']) not in wt_keys:
            continue
        sr, database = [], []
        if condition == 'WT':
            matches = catalog.matches(sample['reference_cell'], row['modification'], chrom, pos, row['strand'])
            sr = [s for s in matches if kinds[s] == 'short_read']
            database = [s for s in matches if kinds[s] == 'database']
        batch.append((sample['sample_id'], chrom, pos, row['strand'], row['modification'],
                      row['total_reads'], row['modified_reads'], row['af'], int(candidate),
                      ';'.join(sr), ';'.join(database)))
        stats['retained_rows'] += 1
        if len(batch) >= 10000:
            db.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?)', batch)
            batch.clear()
    if not stats['input_rows'] or not stats['target_rows']:
        raise ValueError(f'No supported target rows in {source}')
    if source_before != file_info(source):
        raise ValueError(f'Input changed during extraction: {source}')
    db.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?)', batch)
    db.commit()
    upstream = read_pileup_log(log)
    return dict(input=source_before, pileup_log=upstream, counts=dict(stats),
                matches_log_row_count=(stats['input_rows'] == upstream['completed_rows'])
                if upstream['completed_rows'] is not None else None)


def build_site_lists(pairs, catalog_path, out, min_depth=10, min_modified=3, min_af=0.1,
                     ivt_min_modified=3, ivt_af_ratio=0.6, flank=9, progress_every=1000000):
    """No FASTA or BAM required. Missing sample paths must be explicit None."""
    if min_depth < 1 or min_modified < 1 or ivt_min_modified < 1 or not 0 <= min_af <= 1:
        raise ValueError('Invalid count/AF thresholds')
    if not 0 <= ivt_af_ratio <= 1 or flank < 0:
        raise ValueError('Invalid IVT ratio or window flank')
    ids = [p[k] for p in pairs for k in ('wt_sample', 'ivt_sample')]
    if not pairs or len(ids) != len(set(ids)) or any(not s or any(c.isspace() for c in s) for s in ids):
        raise ValueError('Use unique nonempty sample IDs without whitespace')
    for p in pairs:
        if not p.get('cell') or not p.get('reference_cell'):
            raise ValueError('Explicit cell and reference_cell required')
        for condition in ('wt', 'ivt'):
            if p.get(f'{condition}_bedmethyl') is not None:
                for key in (f'{condition}_bedmethyl', f'{condition}_pileup_log'):
                    if p.get(key) is None or not Path(p[key]).is_file():
                        raise FileNotFoundError(f'Set existing {key} for {p["cell"]}')
    catalog = ReferenceCatalog(catalog_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE.txt').write_text('Use only after run.json has status=complete.', encoding='utf-8')
    counts, reports = Counter(), []
    thresholds = dict(min_depth=min_depth, min_modified=min_modified, min_af=min_af)
    db = sqlite3.connect(str(out / 'site_index.sqlite'))
    db.row_factory = sqlite3.Row
    try:
        db.execute('''CREATE TABLE observations (sample_id TEXT, chrom TEXT, start INTEGER,
            strand TEXT, modification TEXT, total_reads INTEGER, modified_reads INTEGER,
            af REAL, candidate INTEGER, short_read_sources TEXT, database_sources TEXT,
            PRIMARY KEY(sample_id,chrom,start,strand,modification))''')
        db.execute('CREATE TABLE output (chrom TEXT,start INTEGER,row_id TEXT,payload TEXT)')
        for pair in pairs:
            for condition in ('wt', 'ivt'):
                source = pair.get(f'{condition}_bedmethyl')
                if source is None:
                    reports.append(dict(sample_id=pair[f'{condition}_sample'], status='not_available'))
                    continue
                sample = dict(sample_id=pair[f'{condition}_sample'], reference_cell=pair['reference_cell'])
                report = _ingest(db, sample, condition.upper(), source, pair[f'{condition}_pileup_log'],
                                 catalog, thresholds, pair['wt_sample'], progress_every)
                reports.append(dict(sample_id=sample['sample_id'], status='available', **report))
            for wt in db.execute('SELECT * FROM observations WHERE sample_id=? ORDER BY chrom,start', (pair['wt_sample'],)):
                ivt = db.execute('''SELECT * FROM observations WHERE sample_id=? AND chrom=?
                    AND start=? AND strand=? AND modification=?''',
                    (pair['ivt_sample'], wt['chrom'], wt['start'], wt['strand'], wt['modification'])).fetchone()
                label, reason, status = wt_label(wt, ivt, pair.get('ivt_bedmethyl') is not None,
                                                ivt_min_modified, ivt_af_ratio)
                _store_output(db, pair, wt, 'WT', label, reason, status, ivt)
            for ivt in db.execute('SELECT * FROM observations WHERE sample_id=? ORDER BY chrom,start', (pair['ivt_sample'],)):
                wt = db.execute('''SELECT * FROM observations WHERE sample_id=? AND chrom=?
                    AND start=? AND strand=? AND modification=?''',
                    (pair['wt_sample'], ivt['chrom'], ivt['start'], ivt['strand'], ivt['modification'])).fetchone()
                paired = (wt is not None and ivt['total_reads'] >= min_depth and
                          wt_label(wt, ivt, True, ivt_min_modified, ivt_af_ratio)[0] == 1)
                if ivt['candidate'] or paired:
                    reason = ('ivt_candidate_and_paired' if ivt['candidate'] and paired else
                              'ivt_candidate' if ivt['candidate'] else 'ivt_paired_wt_positive')
                    _store_output(db, pair, ivt, 'IVT', 0, reason, 'final', None)
            db.commit()
        with ExitStack() as stack:
            writers = {}
            for mod in TARGETS:
                folder = out / mod
                folder.mkdir()
                for name in ('all_sites', 'training_sites', 'pending_sites', 'unlabeled_sites'):
                    handle = stack.enter_context((folder / f'{name}.tsv').open('w', newline='', encoding='utf-8'))
                    writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter='\t')
                    writer.writeheader()
                    writers[mod, name] = writer
            all_handle = stack.enter_context((out / 'all_sites.tsv').open('w', newline='', encoding='utf-8'))
            all_writer = csv.DictWriter(all_handle, fieldnames=FIELDS, delimiter='\t')
            all_writer.writeheader()
            chrom, previous, group = None, -1, None
            for result in db.execute('SELECT * FROM output ORDER BY chrom,start,row_id'):
                row = json.loads(result['payload'])
                if row['chrom'] != chrom or row['start'] - previous > 2 * flank:
                    group = f"{row['chrom']}:{row['start']}"
                chrom, previous = row['chrom'], row['start']
                row['group_id'] = group
                all_writer.writerow(row)
                writers[row['modification'], 'all_sites'].writerow(row)
                category = ('training_sites' if row['label'] != '' else
                            'pending_sites' if row['label_status'] == 'pending' else 'unlabeled_sites')
                writers[row['modification'], category].writerow(row)
                counts[row['sample_id'], row['modification'], str(row['label']), row['label_reason']] += 1
        with (out / 'label_counts.tsv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['sample_id', 'modification', 'label', 'label_reason', 'count'])
            for key, value in sorted(counts.items()):
                writer.writerow([*key, value])
        summarize_site_list(out / 'all_sites.tsv', out / 'summary', pairs, catalog.sources)
        report = dict(schema='site_lists_v2', status='complete', assembly='GRCh38', coordinates='BED0', pairs=pairs,
                      inputs=reports, reference_catalog=str(catalog.path), reference_sources=catalog.sources,
                      thresholds=thresholds, ivt_min_modified=ivt_min_modified, ivt_af_ratio=ivt_af_ratio,
                      window_flank=flank, total_rows=sum(counts.values()),
                      policies=dict(positive='cell-matched short-read OR database overlap; no DRACH-only positives',
                                    negative_override='IVT N_mod >= threshold AND IVT AF >= ratio * WT AF',
                                    af='N_mod / (valid_cov + fail + diff + nocall); deletions excluded',
                                    pending='entire IVT sample unavailable: WT labels pending',
                                    missing_locus='available IVT but locus missing/zero-depth: unknown, no veto',
                                    grouping='overlapping genomic windows across samples, conditions, strands, targets',
                                    cv='no folds yet; regenerate union lists and assign grouped folds after all samples arrive',
                                    labels='operational weak supervision, not independently validated truth'))
        (out / 'run.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as exc:
        (out / 'FAILED.txt').write_text(str(exc), encoding='utf-8')
        raise
    finally:
        db.close()


def _store_output(db, pair, observation, condition, label, reason, status, ivt):
    row = {k: observation[k] for k in ('sample_id', 'chrom', 'start', 'strand', 'modification',
                                     'total_reads', 'modified_reads', 'af', 'short_read_sources', 'database_sources')}
    row.update(row_id=f"{row['sample_id']}|{row['chrom']}:{row['start']}:{row['strand']}:{row['modification']}",
               end=row['start'] + 1, cell=pair['cell'], reference_cell=pair['reference_cell'], condition=condition,
               positive_evidence=int(bool(row['short_read_sources'] or row['database_sources'])),
               paired_ivt_depth=ivt['total_reads'] if ivt is not None else '',
               paired_ivt_modified_reads=ivt['modified_reads'] if ivt is not None else '',
               paired_ivt_af=ivt['af'] if ivt is not None and ivt['total_reads'] else '',
               ivt_comparison=('not_applicable' if condition == 'IVT' else
                               'sample_pending' if pair.get('ivt_bedmethyl') is None else
                               'missing_locus' if ivt is None else 'zero_depth' if not ivt['total_reads'] else 'observed'),
               label=label, label_reason=reason, label_status=status)
    db.execute('INSERT INTO output VALUES (?,?,?,?)', (row['chrom'], row['start'], row['row_id'], json.dumps(row)))
