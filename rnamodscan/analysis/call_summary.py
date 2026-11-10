"""Stream bedMethyl once per sample; count sites, reads and reference overlaps."""
import csv
import gzip
import json
import time
from contextlib import ExitStack
from collections import Counter
from pathlib import Path

from utils.pileup.candidates import validate_options
from utils.io.bed import bedmethyl_rows
from utils.references import ReferenceCatalog, IndexedFasta, drach
from utils.references.catalog import MODIFICATIONS, sha256
from utils.pileup.provenance import read_pileup_log



def write_table(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def summarize_calls(samples, thresholds, reference_fasta, catalog_path, out, upstream_min_ml=None,
                    *, write_called_sites=True, compress_called_sites=False, progress_every=1000000):
    """Thresholds are dicts with name/min_depth/min_modified/min_af (inclusive)."""
    if not samples or not thresholds:
        raise ValueError('Provide samples and thresholds')
    names = [s['sample_id'] for s in samples]
    if len(names) != len(set(names)) or any(not x or any(c.isspace() for c in x) for x in names):
        raise ValueError('Unique, nonempty sample IDs without whitespace required')
    if any(s['condition'] not in ('WT', 'IVT') for s in samples):
        raise ValueError('Condition must be WT or IVT')
    if len({t['name'] for t in thresholds}) != len(thresholds) or any(not t['name'] for t in thresholds):
        raise ValueError('Threshold names must be nonempty and unique')
    for t in thresholds:
        validate_options(t['min_depth'], t['min_modified'], t['min_af'],
                         upstream_min_ml if upstream_min_ml is not None else 0)
    for s in samples:
        if not Path(s['bedmethyl']).is_file():
            raise FileNotFoundError(s['bedmethyl'])
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    upstream = {}
    warnings = []
    for sample in samples:
        sid = sample['sample_id']
        metadata = read_pileup_log(sample['pileup_log']) if sample.get('pileup_log') else None
        if metadata and metadata['base_thresholds'] and upstream_min_ml is not None:
            raise ValueError('Sampled per-base thresholds in log; set upstream_min_ml=None, not a fixed ML')
        upstream[sid] = metadata
        if upstream_min_ml is None and (metadata is None or not metadata['base_thresholds']):
            warnings.append(f'{sid}: upstream thresholds are not recorded; resolve before comparing samples')
        if metadata and metadata['filter_mode'] == 'sampled_per_base':
            warnings.append(f'{sid}: upstream filtering is sampled per base, not fixed ML=125')
    catalog = ReferenceCatalog(catalog_path)
    fasta = IndexedFasta(reference_fasta)
    summaries, overlaps, audits, partitions = [], [], [], []
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE.txt').write_text('Use results only after run.json is complete.', encoding='utf-8')
    started = time.monotonic()
    try:
        fields = ['sample_id', 'cell', 'reference_cell', 'condition', 'threshold', 'chrom', 'start',
                  'end', 'strand', 'modification', 'total_reads', 'modified_reads', 'af',
                  'motif5', 'motif_group', 'short_read_sources', 'database_sources', 'ivt_false_call']
        with ExitStack() as stack:
            writer = None
            if write_called_sites:
                if compress_called_sites:
                    handle = stack.enter_context(gzip.open(out / 'called_sites.tsv.gz', 'wt', encoding='utf-8', newline='', compresslevel=1))
                else:
                    handle = stack.enter_context((out / 'called_sites.tsv').open('w', encoding='utf-8', newline=''))
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
                writer.writeheader()
            for sample in samples:
                counts, hits, observed = Counter(), Counter(), Counter()
                sid = sample['sample_id']
                print(f'Summarizing {sid}', flush=True)
                source_lookup = {m: [(s, catalog.sites[s['id']]) for s in catalog.applicable(sample['reference_cell'], m)
                                     if s['status'] == 'ready'] for m in MODIFICATIONS}
                contigs = set()
                stat_before = Path(sample['bedmethyl']).stat()
                sample_started = time.monotonic()
                total_rows = 0
                for total_rows, row in enumerate(bedmethyl_rows(sample['bedmethyl']), 1):
                    contigs.add(row['chrom'])
                    if progress_every and total_rows % progress_every == 0:
                        elapsed = time.monotonic() - sample_started
                        print(f'{sid}: {total_rows:,} rows; {row["chrom"]}:{row["start"]}; {elapsed / 60:.1f} min', flush=True)
                    observed[row['mod_code']] += 1
                    mod = row['modification']
                    if mod not in MODIFICATIONS:
                        continue
                    seq = fasta.motif(row['chrom'], row['start'], row['strand']) if mod == 'm6A' else ''
                    group = ('unknown' if any(b not in 'ACGT' for b in seq) else
                             'DRACH' if drach(seq) else 'non_DRACH') if mod == 'm6A' else 'all'
                    groups = ['all', group] if mod == 'm6A' else ['all']
                    matches, short, db = [], [], []
                    eligible = any(row['total_reads'] >= t['min_depth'] and row['modified_reads'] >= t['min_modified']
                                   and row['af'] >= t['min_af'] for t in thresholds)
                    if eligible:
                        site = row['chrom'], row['start'], row['strand']
                        for source, sites in source_lookup[mod]:
                            if site in sites:
                                matches.append(source['id'])
                                (short if source['kind'] == 'short_read' else db).append(source['id'])
                    for t in thresholds:
                        passed = (row['total_reads'] >= t['min_depth'] and
                                  row['modified_reads'] >= t['min_modified'] and row['af'] >= t['min_af'])
                        for g in groups:
                            k = t['name'], mod, g
                            counts[k + ('input_sites',)] += 1
                            counts[k + ('input_modified_reads',)] += row['modified_reads']
                            counts[k + ('raw_call_sites',)] += row['modified_reads'] > 0
                            counts[k + ('depth_eligible_sites',)] += row['total_reads'] >= t['min_depth']
                            if passed:
                                counts[k + ('call_sites',)] += 1
                                counts[k + ('modified_reads_at_calls',)] += row['modified_reads']
                                category = 'both' if short and db else 'short_read_only' if short else 'database_only' if db else 'neither'
                                hits[k + ('partition_' + category,)] += 1
                                for ref in matches + (['short_read_any'] if short else []) + (['database_any'] if db else []):
                                    hits[k + (ref,)] += 1
                        if passed and writer is not None:
                            record = {k: row[k] for k in fields if k in row}
                            record.update({k: sample[k] for k in ('sample_id', 'cell', 'reference_cell', 'condition')})
                            record.update(threshold=t['name'], motif5=seq, motif_group=group,
                                          short_read_sources=';'.join(short), database_sources=';'.join(db),
                                          ivt_false_call=1 if sample['condition'] == 'IVT' else '')
                            writer.writerow(record)
                if not observed or not any(code in ('a', 'm', '17596', '17802') for code in observed):
                    raise ValueError(f'{sid}: no supported bedMethyl rows')
                stat_after = Path(sample['bedmethyl']).stat()
                if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
                    raise ValueError(f'{sid}: input changed while being read')
                expected = upstream[sid]['completed_rows'] if upstream[sid] else None
                if expected is not None and expected != total_rows:
                    warnings.append(f'{sid}: pileup log records {expected} rows but BED contains {total_rows}')
                audits.append(dict(sample_id=sid, observed_codes=dict(observed), input_rows=total_rows,
                                   source_bytes=stat_before.st_size, source_mtime_ns=stat_before.st_mtime_ns,
                                   contigs=sorted(contigs), expected_rows_from_log=expected,
                                   matches_log_row_count=(expected == total_rows) if expected is not None else None,
                                   missing_modifications=[m for code, m in [('a','m6A'), ('m','m5C'), ('17596','inosine'), ('17802','pseU')] if code not in observed]))
                for t in thresholds:
                    for mod in MODIFICATIONS:
                        applicable = catalog.applicable(sample['reference_cell'], mod)
                        refs = [(s['id'], s['status'], s['kind'], s['cell']) for s in applicable]
                        for kind, union in [('short_read', 'short_read_any'), ('database', 'database_any')]:
                            subset = [s for s in applicable if s['kind'] == kind]
                            ready = sum(s['status'] == 'ready' for s in subset)
                            status = 'ready' if ready and ready == len(subset) else 'partial' if ready else 'unavailable'
                            refs.append((union, status, kind, sample['reference_cell'] if kind == 'short_read' else '*'))
                        for g in (['all', 'DRACH', 'non_DRACH', 'unknown'] if mod == 'm6A' else ['all']):
                            key = t['name'], mod, g
                            base = {k: sample[k] for k in ('sample_id', 'cell', 'reference_cell', 'condition')}
                            base.update(threshold=t['name'], min_depth=t['min_depth'], min_modified=t['min_modified'],
                                        min_af=t['min_af'], modification=mod, motif_group=g)
                            result = dict(base)
                            for metric in ('input_sites', 'input_modified_reads', 'raw_call_sites', 'depth_eligible_sites', 'call_sites', 'modified_reads_at_calls'):
                                result[metric] = counts[key + (metric,)]
                            calls = result['call_sites']
                            result['ivt_false_call_sites'] = calls if sample['condition'] == 'IVT' else ''
                            result['call_fraction_of_depth_eligible_sites'] = calls / result['depth_eligible_sites'] if result['depth_eligible_sites'] else ''
                            summaries.append(result)
                            union_states = {ref: status for ref, status, _, _ in refs if ref in ('short_read_any', 'database_any')}
                            partition_ready = all(value == 'ready' for value in union_states.values())
                            part = dict(base, reference_status='ready' if partition_ready else 'incomplete_reference',
                                        call_sites=calls)
                            for category in ('both', 'short_read_only', 'database_only', 'neither'):
                                part[category] = hits[key + ('partition_' + category,)] if partition_ready else ''
                            partitions.append(part)
                            for ref, status, kind, cell in refs:
                                n = hits[key + (ref,)] if status != 'unavailable' else ''
                                overlaps.append(dict(base, source_id=ref, source_kind=kind, source_cell=cell,
                                                     reference_status=status, call_sites=calls, overlap_sites=n,
                                                     overlap_fraction=n / calls if calls and n != '' else ''))
        write_table(out / 'call_counts.tsv', summaries)
        write_table(out / 'overlap_counts.tsv', overlaps)
        write_table(out / 'reference_overlap_summary.tsv', partitions)
        report = dict(samples=[dict(s, bedmethyl=str(Path(s['bedmethyl']).resolve())) for s in samples],
                      thresholds=thresholds, reference_fasta=str(Path(reference_fasta).resolve()),
                      fasta_index_sha256=sha256(Path(str(reference_fasta)+'.fai')),
                      catalog_path=str(catalog.path), references=catalog.sources, input_audits=audits,
                      upstream_min_ml=upstream_min_ml, upstream_by_sample=upstream, warnings=warnings,
                      upstream_ml_note='See actual per-sample log thresholds; bedMethyl cannot reapply per-read ML',
                      write_called_sites=write_called_sites, compress_called_sites=compress_called_sites,
                      elapsed_seconds=time.monotonic()-started, status='complete',
                      af_definition='N_mod / (N_valid_cov + N_fail + N_diff + N_nocall)',
                      match_definition='Exact chromosome, BED0 start, strand, modification; explicit reference cell',
                      ivt_policy='All IVT calls are operational false calls, including reference overlaps',
                      training_data_created=False)
        (out / 'run.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
        (out / 'INCOMPLETE.txt').unlink()
        for warning in warnings:
            print('WARNING:', warning, flush=True)
        return report
    except BaseException as exc:
        (out / 'FAILED.txt').write_text(str(exc), encoding='utf-8')
        raise
    finally:
        fasta.close()
