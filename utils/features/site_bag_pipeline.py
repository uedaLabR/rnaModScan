"""Common BAM -> ragged per-site Parquet for training and inference."""
import csv
import json
from pathlib import Path

from utils.features.bam import extract_bam_features
from utils.features.grouping import candidate_groups
from utils.features.site_bags import SiteBagWriter, build_site_bag_index, SCHEMA
from utils.features.compact_inputs import STORAGE
from utils.io.artifacts import completed_run, file_info
from utils.references.catalog import ReferenceCatalog, sha256


def read_sample_metadata(path, sample_id):
    with Path(path).open(newline='') as handle:
        return [row for row in csv.DictReader(handle, delimiter='\t') if row['sample_id'] == sample_id]


def extract_site_bag_sample(site_list, sample, fasta, out, catalog_path, *, mode,
                            max_reads=300, sampling_seed=42, flank=9, min_mapq=0,
                            bam_threads=1, progress_seconds=60, candidate_group_gap=10000,
                            read_batch_size=256, group_workers=1, group_batch_size=32,
                            max_fetch_span=10000000):
    """Use the same scanner and storage schema in both modes; only labels differ."""
    out, site_list = Path(out), Path(site_list)
    if candidate_group_gap is not None:
        candidate_groups([], candidate_group_gap, max_fetch_span)
    if type(read_batch_size) is not int or read_batch_size < 1:
        raise ValueError('read_batch_size must be a positive integer')
    if any(type(value) is not int or value < 1 for value in (group_workers, group_batch_size)):
        raise ValueError('Set positive group_workers and group_batch_size')
    if group_workers > 1 and candidate_group_gap is None:
        raise ValueError('Group parallelism requires a candidate_group_gap')
    identity = {key: sample[key] for key in ('sample_id', 'cell', 'reference_cell', 'condition')}
    expected = dict(schema=SCHEMA, mode=mode, max_reads=max_reads,
                    sampling_feature_order='final_reservoir_then_mm_features_v1',
                    sample_identity=identity, catalog=str(Path(catalog_path).resolve()) if catalog_path is not None else None,
                    catalog_sha256=sha256(catalog_path) if catalog_path is not None else None,
                    sampling_seed=sampling_seed, sample_id=sample['sample_id'],
                    site_list_sha256=sha256(site_list), bam=file_info(sample['bam']),
                    reference=file_info(fasta), window_flank=flank, min_mapq=min_mapq,
                    candidate_group_gap=candidate_group_gap, max_fetch_span=max_fetch_span)
    if out.exists():
        run = completed_run(out)
        # Fetch strategy does not change sampled site features. Keep completed
        # data reusable while retaining its actual strategy in the saved audit.
        semantic_inputs = {key: value for key, value in expected.items()
                           if key not in ('candidate_group_gap', 'max_fetch_span')}
        if any(run.get(key) != value for key, value in semantic_inputs.items()):
            raise ValueError('Existing site bags have different inputs/settings; choose a new output')
        for source in run.get('sources', []):
            if source.get('status', 'ready') == 'ready' and sha256(source['path']) != source['sha256']:
                raise ValueError('Reference source changed after extraction')
        if run.get('candidate_group_gap') != candidate_group_gap:
            print(f"{sample['sample_id']}: reusing complete site bags from "
                  f"{run['extraction_strategy']}; requested fetch strategy was not rerun", flush=True)
        return run
    options = dict(mode=mode, max_reads=max_reads, sampling_seed=sampling_seed, flank=flank,
                   min_mapq=min_mapq, bam_threads=bam_threads, progress_seconds=progress_seconds,
                   candidate_group_gap=candidate_group_gap, read_batch_size=read_batch_size,
                   group_batch_size=group_batch_size, max_fetch_span=max_fetch_span)
    if group_workers > 1:
        from utils.features.parallel_groups import extract_parallel_groups
        return extract_parallel_groups(site_list, sample, fasta, out, catalog_path,
                                       expected, options, group_workers, group_batch_size)
    rows = read_sample_metadata(site_list, sample['sample_id'])
    if not rows:
        raise ValueError('No candidate rows for sample')
    catalog = ReferenceCatalog(catalog_path) if catalog_path is not None else None
    return extract_site_bag_rows(site_list, rows, sample, fasta, out, catalog, **options)


def extract_site_bag_rows(site_list, rows, sample, fasta, out, catalog, *, mode,
                          max_reads=300, sampling_seed=42, flank=9, min_mapq=0,
                          bam_threads=1, progress_seconds=60, candidate_group_gap=10000,
                          read_batch_size=256, group_batch_size=32, max_fetch_span=10000000):
    """Worker-local rows and catalog; open and close BAM/FASTA inside this call."""
    out = Path(out)
    sink = SiteBagWriter(out, rows, sample, catalog, mode=mode, flank=flank,
                         max_reads=max_reads, sampling_seed=sampling_seed)
    report = extract_bam_features(site_list, sample['sample_id'], sample['bam'], fasta, out,
        flank=flank, min_mapq=min_mapq, parquet_batch_rows=sink.batch_sites,
        parquet_file_rows=sink.file_sites, partition_by_locus=True, bam_threads=bam_threads,
        progress_seconds=progress_seconds, candidate_group_gap=candidate_group_gap,
        read_batch_size=read_batch_size, group_batch_size=group_batch_size, max_fetch_span=max_fetch_span,
        max_reads=max_reads, sampling_seed=sampling_seed, site_sink=sink)
    report.update(sink.provenance())
    identity = {key: sample[key] for key in ('sample_id', 'cell', 'reference_cell', 'condition')}
    report.update(sample_identity=identity, read_feature_storage=STORAGE)
    report['feature_set'] = 'sequence_reads_repeat18_v1'
    report['read_offset_mode'] = 'one nested read list per site; no flat read offsets'
    report['storage'] = 'one row per site; sampled real reads only; no padding stored'
    (out / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def extract_site_bag_job(job):
    """One worker per BAM; releasing the process releases its reference/read memory."""
    try:
        extract_site_bag_sample(job['site_list'], job['sample'], job['fasta'], job['out'],
                                job['catalog'], **job['options'])
        return dict(sample_id=job['sample']['sample_id'], status='complete')
    except Exception as error:
        import traceback
        traceback.print_exc()
        return dict(sample_id=job['sample']['sample_id'], status='failed', error=str(error))


def extract_site_bags_and_index(site_list, samples, fasta, out, catalog_path, *, mode,
                                max_reads=300, sampling_seed=42, sample_workers=1,
                                flank=9, min_mapq=0, bam_threads=1, progress_seconds=60,
                                candidate_group_gap=10000, read_batch_size=256,
                                group_workers=1, group_batch_size=32, max_fetch_span=10000000):
    """Process independent samples, then index their common per-site records."""
    import multiprocessing
    if mode not in ('training', 'inference') or type(max_reads) is not int or max_reads < 1:
        raise ValueError('Set mode training/inference and a positive max_reads')
    if type(sample_workers) is not int or sample_workers < 1 or not samples:
        raise ValueError('Set positive sample_workers and at least one sample')
    if any(type(value) is not int or value < 1 for value in (group_workers, group_batch_size)):
        raise ValueError('Set positive group_workers and group_batch_size')
    if group_workers > 1 and (sample_workers > 1 or candidate_group_gap is None):
        raise ValueError('Group parallelism requires sample_workers=1 and a candidate_group_gap')
    if len({sample['sample_id'] for sample in samples}) != len(samples):
        raise ValueError('Duplicate sample IDs')
    out = Path(out)
    jobs, directories = [], []
    for sample in samples:
        directory = out / 'samples' / sample['sample_id']
        directories.append(directory)
        jobs.append(dict(site_list=site_list, sample=sample, fasta=fasta, out=directory, catalog=catalog_path,
            options=dict(mode=mode, max_reads=max_reads, sampling_seed=sampling_seed, flank=flank,
                         min_mapq=min_mapq, bam_threads=bam_threads, progress_seconds=progress_seconds,
                         candidate_group_gap=candidate_group_gap, read_batch_size=read_batch_size,
                         group_workers=group_workers, group_batch_size=group_batch_size,
                         max_fetch_span=max_fetch_span)))
    if sample_workers == 1:
        reports = [extract_site_bag_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context('spawn')
        with context.Pool(min(sample_workers, len(jobs)), maxtasksperchild=1) as pool:
            reports = pool.map(extract_site_bag_job, jobs, chunksize=1)
    failed = [report for report in reports if report['status'] != 'complete']
    if failed:
        raise RuntimeError('Site bag extraction failed: ' + json.dumps(failed))
    index_directory = out / 'index' / 'adenine'
    if index_directory.exists():
        run = completed_run(index_directory)
        if run['schema'] != 'site_bag_index_v3' or run['index_sha256'] != sha256(index_directory / 'sites.tsv'):
            raise ValueError('Existing site bag index changed')
        for source in run['feature_sources']:
            if sha256(Path(source['path']) / 'run.json') != source['run_sha256']:
                raise ValueError('Site bag source changed')
        return run
    return build_site_bag_index(directories, index_directory, require_labels=mode == 'training')
