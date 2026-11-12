"""Independent group batches, disk-only results, deterministic manifest merging."""
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import csv
import json
import multiprocessing
import os
from pathlib import Path
import time

from utils.io.artifacts import file_info
from utils.references.catalog import ReferenceCatalog

_catalog = None


def initialize_worker(catalog_path, mode='inference'):
    global _catalog
    _catalog = (ReferenceCatalog(catalog_path, for_training=mode == 'training')
                if catalog_path is not None else None)


def sample_groups(site_list, sample_id, gap, max_fetch_span=10000000):
    """Stream metadata in original coordinate order; never load read tensors here."""
    group, last = [], None
    with Path(site_list).open(newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row['sample_id'] != sample_id:
                continue
            key = row['chrom'], int(row['start'])
            if last is not None and key < last:
                raise ValueError('Site TSV must be sorted by chromosome and start')
            if group and (key[0] != last[0] or key[1] - last[1] >= gap or
                          key[1] - int(group[0]['start']) + 1 > max_fetch_span):
                yield group
                group = []
            group.append(row)
            last = key
    if group:
        yield group


def prepare_jobs(site_list, sample, fasta, out, options, batch_size):
    """Batch consecutive groups to amortize BAM/index opens and small-file overhead."""
    jobs, rows = [], []
    total_sites = group_count = batch_groups = 0
    inputs = out / 'group_inputs' / 'batches'
    inputs.mkdir(parents=True)

    def write_batch():
        number = len(jobs)
        path = inputs / f'batch-{number:06d}.tsv'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, list(rows[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
        jobs.append(dict(number=number, site_list=str(path), sample=sample, fasta=str(fasta),
                         out=str(out / 'sites' / f'batch-{number:06d}'), options=options,
                         log=str(out / 'logs' / f'batch-{number:06d}.log')))

    with (out / 'candidate_groups.tsv').open('w', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['chrom', 'group', 'start', 'end', 'candidate_sites'])
        span_limit = options.get('max_fetch_span', 10000000)
        for group in sample_groups(site_list, sample['sample_id'], options['candidate_group_gap'], span_limit):
            if rows and (rows[-1]['chrom'] != group[0]['chrom'] or
                         int(group[-1]['start']) - int(rows[0]['start']) + 1 > span_limit):
                write_batch()
                rows, batch_groups = [], 0
            writer.writerow([group[0]['chrom'], group_count, group[0]['start'], int(group[-1]['start']) + 1, len(group)])
            total_sites += len(group)
            group_count += 1
            batch_groups += 1
            rows.extend(group)
            if batch_groups == batch_size:
                write_batch()
                rows, batch_groups = [], 0
        if rows:
            write_batch()
    if not jobs:
        raise ValueError('No candidate rows for sample')
    (inputs.parent / 'run.json').write_text(json.dumps(dict(status='complete',
        schema='candidate_group_inputs_v1', window_flank=options['flank'],
        input_batches=len(jobs), candidate_groups=group_count, sites=total_sites), indent=2) + '\n')
    return jobs, total_sites, group_count


def extract_group_batch(job):
    from utils.features.site_bag_pipeline import read_sample_metadata, extract_site_bag_rows
    with Path(job['log']).open('w', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        rows = read_sample_metadata(job['site_list'], job['sample']['sample_id'])
        report = extract_site_bag_rows(job['site_list'], rows, job['sample'], job['fasta'],
                                       job['out'], _catalog, **job['options'])
    # No tensors or BAM objects cross process boundaries.
    return dict(number=job['number'], counts=report['counts'])


def accumulate_counts(total, values):
    for key, value in values.items():
        if key.startswith('peak_') or key == 'bam_records_total':
            total[key] = max(total[key], value)
        else:
            total[key] += value


def write_progress(out, sample, counts, sites, groups, completed, total, started):
    data = dict(status='running', sample_id=sample['sample_id'], pid=os.getpid(),
                updated_unix=time.time(), elapsed_seconds=time.monotonic() - started,
                counts=dict(counts, total_sites=sites, candidate_groups_total=groups),
                completed_batches=completed, total_batches=total,
                progress_unit='completed_candidate_groups', completion_order='unordered; final manifest uses original batch order')
    temporary = out / 'progress.json.partial'
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(out / 'progress.json')


def merge_group_reports(jobs, out, expected, options, workers, batch_size, started):
    """Commit file locations and summaries in input order, regardless of finish order."""
    files, counts, channels = [], Counter(), Counter()
    base = None
    logical_offset = 0
    with (out / 'site_summary.tsv').open('w', newline='') as summary:
        writer = None
        for index, job in enumerate(jobs):
            directory = Path(job['out'])
            report = json.loads((directory / 'run.json').read_text())
            if report['status'] != 'complete':
                raise ValueError('Incomplete group batch')
            if base is None:
                base = report
            accumulate_counts(counts, report['counts'])
            channels.update(report['observed_mm_groups'])
            prefix = directory.relative_to(out / 'sites') / 'sites'
            files.extend(dict(item, path=str(prefix / item['path'])) for item in report['parquet_files']['sites'])
            with (directory / 'site_summary.tsv').open(newline='') as handle:
                reader = csv.DictReader(handle, delimiter='\t')
                if index == 0:
                    writer = csv.DictWriter(summary, reader.fieldnames, delimiter='\t', lineterminator='\n')
                    writer.writeheader()
                for row in reader:
                    if 'read_offset' in row:
                        row['read_offset'] = logical_offset
                        logical_offset += int(row['retained_depth'])
                    writer.writerow(row)
    base.update(expected)
    base.update(status='complete', extraction_strategy='independent_candidate_group_batches',
                group_workers=workers, group_batch_size=batch_size,
                counts=dict(counts), parquet_files=dict(sites=files), observed_mm_groups=dict(channels),
                missing_mm_groups=[name for name in base['modification_channels'] if not channels[name]],
                elapsed_seconds=time.monotonic() - started)
    base['policies']['parallelism'] = 'spawn; independent worker-local BAM/FASTA handles; ordered disk manifests; no read tensors in IPC'
    base['policies']['peak_counts'] = 'maximum per worker batch, not the aggregate concurrent memory footprint; use external resource samples for group totals'
    return base


def extract_parallel_groups(site_list, sample, fasta, out, catalog_path, expected, options, workers, batch_size):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE.txt').write_text('Parallel group extraction is incomplete.\n')
    (out / 'logs').mkdir()
    started = time.monotonic()
    inputs = {str(path): file_info(path) for path in (Path(site_list), Path(sample['bam']), Path(fasta))}
    try:
        jobs, total_sites, total_groups = prepare_jobs(site_list, sample, fasta, out, options, batch_size)
        counts = Counter()
        write_progress(out, sample, counts, total_sites, total_groups, 0, len(jobs), started)
        context = multiprocessing.get_context('spawn')
        with context.Pool(min(workers, len(jobs)), initializer=initialize_worker,
                          initargs=(catalog_path, options['mode'])) as pool:
            for completed, result in enumerate(pool.imap_unordered(extract_group_batch, jobs, chunksize=1), 1):
                accumulate_counts(counts, result['counts'])
                write_progress(out, sample, counts, total_sites, total_groups, completed, len(jobs), started)
                print(f"{sample['sample_id']}: group batches {completed}/{len(jobs)}, sites {counts['sites']}/{total_sites}", flush=True)
        for path, identity in inputs.items():
            if file_info(path) != identity:
                raise ValueError('Input changed during parallel extraction: ' + path)
        report = merge_group_reports(jobs, out, expected, options, workers, batch_size, started)
        if report['counts']['sites'] != total_sites or report['counts']['candidate_groups_completed'] != total_groups:
            raise ValueError('Incomplete parallel site/group coverage')
        report['site_list'] = file_info(site_list)
        (out / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
        progress = json.loads((out / 'progress.json').read_text())
        progress.update(status='complete', elapsed_seconds=time.monotonic() - started)
        (out / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as error:
        (out / 'FAILED.txt').write_text(str(error) + '\n')
        progress_path = out / 'progress.json'
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            progress.update(status='failed', error=str(error))
            progress_path.write_text(json.dumps(progress, indent=2) + '\n')
        raise
