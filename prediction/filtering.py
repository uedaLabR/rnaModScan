"""Join saved site metadata to neural scores, one chromosome/strand at a time.

This CPU stage never loads TensorFlow, BAM, GTF or raw read features. Reference
rescue is applied after independent scoring of each target modification.
"""
import csv
import json
from pathlib import Path

from prediction.predict import write_predictions
from utils.features.parquet import require_arrow
from utils.io.artifacts import completed_run
from utils.references.catalog import ReferenceCatalog, sha256


def partition_file(directory, relative_path, size=None):
    """Resolve a manifest entry inside its completed artifact directory."""
    directory = Path(directory).resolve()
    path = (directory / relative_path).resolve()
    if directory not in path.parents:
        raise ValueError('Partition paths must stay inside their artifact directory')
    if size is not None and path.stat().st_size != size:
        raise ValueError(f'Partition changed after completion: {path}')
    return path


def read_partition_scores(path, expected_rows):
    """Keep only one chromosome/strand's small row_id -> score map in memory."""
    _, pq = require_arrow()
    scores = {}
    for batch in pq.ParquetFile(str(path)).iter_batches(columns=['row_id', 'keep_score']):
        for row in batch.to_pylist():
            if row['row_id'] in scores:
                raise ValueError('Duplicate scored row_id')
            scores[row['row_id']] = row['keep_score']
    if len(scores) != expected_rows:
        raise ValueError('Score partition row count mismatch')
    return scores


def write_partition_sites(path, destination, chrom, strand):
    """Stream the cached site metadata into the existing decision writer."""
    _, pq = require_arrow()
    source = pq.ParquetFile(str(path))
    fields = source.schema_arrow.names + ['regions']
    expected_scores = set()
    with destination.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fields, delimiter='\t')
        writer.writeheader()
        for batch in source.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                if row['chrom'] != chrom or row['strand'] != strand:
                    raise ValueError('Cached site is in the wrong chromosome/strand partition')
                labels = []
                for flag, name in [('region_cds', 'CDS'), ('region_utr', 'UTR'),
                                   ('region_intron', 'intron'), ('region_other_exon', 'other_exon'),
                                   ('region_outside_transcript', 'outside_transcript')]:
                    if row.get(flag):
                        labels.append(name)
                row['regions'] = ';'.join(labels) or 'unknown'
                writer.writerow(row)
                if (row.get('filter_mode') or 'nn') in ('nn', 'nn_drach_keep', 'nn_mettl16_keep') and int(row['retained_depth']) > 0:
                    expected_scores.add(row['row_id'])
    return fields, expected_scores


def merge_count_tables(directories, filename, destination, key_columns):
    """Sum per-partition counts; an unavailable reference remains blank."""
    columns, totals = None, {}
    for directory in directories:
        with (directory / filename).open(encoding='utf-8', newline='') as handle:
            reader = csv.reader(handle, delimiter='\t')
            header = next(reader)
            if columns is None:
                columns = header
            if header != columns:
                raise ValueError('Partition summary columns disagree')
            for row in reader:
                key = tuple(row[:key_columns])
                values = [None if value == '' else int(value) for value in row[key_columns:]]
                if key not in totals:
                    totals[key] = values
                else:
                    for index, value in enumerate(values):
                        if (value is None) != (totals[key][index] is None):
                            raise ValueError('Reference availability differs between partitions')
                        if value is not None:
                            totals[key][index] += value
    if columns is None:
        columns = EMPTY_COUNT_HEADERS[filename]
    with destination.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(columns)
        for key, values in sorted(totals.items()):
            writer.writerow([*key, *values])


EMPTY_COUNT_HEADERS = {
    'summary.tsv': ['sample_id', 'modification', 'input_sites', 'predicted_sites', 'unpredicted_sites',
                    'nn_pass_sites', 'reference_overlap_sites', 'rescued_sites', 'mettl16_keep_sites', 'final_kept_sites', 'final_not_kept_sites'],
    'decision_counts.tsv': ['sample_id', 'modification', 'prediction_status', 'nn_keep', 'rescued',
                            'final_keep', 'final_reason', 'sites'],
    'source_overlap_counts.tsv': ['sample_id', 'modification', 'source_id', 'source_kind', 'source_cell',
                                 'reference_status', 'overlap_sites', 'nn_pass_sites', 'rescued_sites', 'final_kept_sites'],
    'motif_summary.tsv': ['sample_id', 'modification', 'motif_group', 'input_sites', 'predicted_sites',
                          'nn_pass_sites', 'reference_overlap_sites', 'rescued_sites', 'drach_keep_sites', 'mettl16_keep_sites', 'final_kept_sites'],
}


def combine_partition_predictions(directories, out, empty_fields):
    """Produce one final table per modification and small aggregated summaries."""
    for filename in ('all_predictions.tsv', 'kept_sites.tsv'):
        columns = None
        with (out / filename).open('w', encoding='utf-8', newline='') as dest:
            for directory in directories:
                with (directory / filename).open(encoding='utf-8', newline='') as source:
                    header = source.readline()
                    if columns is None:
                        columns = header
                        dest.write(header)
                    elif header != columns:
                        raise ValueError('Site output columns differ between partitions')
                    for line in source:
                        dest.write(line)
            if columns is None:
                csv.writer(dest, delimiter='\t').writerow(empty_fields)
    with (out / 'kept_sites.bed').open('w', encoding='utf-8') as dest:
        for directory in directories:
            with (directory / 'kept_sites.bed').open(encoding='utf-8') as source:
                for line in source:
                    dest.write(line)
    for filename, key_columns in [('summary.tsv', 2), ('decision_counts.tsv', 7),
                                   ('source_overlap_counts.tsv', 6), ('motif_summary.tsv', 3)]:
        merge_count_tables(directories, filename, out / filename, key_columns)


def filter_scored_inputs(input_directory, score_directory, out, reference_catalog,
                         reference_cell, *, condition='WT'):
    """Filter completed scores with the exact sites saved before GPU inference."""
    from prediction.predict import DECISION_FIELDS
    from prediction.input_parquet import SITE_FIELDS
    input_directory, out = Path(input_directory), Path(out)
    inputs = completed_run(input_directory)
    if score_directory is None:
        if inputs.get('model') is not None or any(part['input_count'] for part in inputs['partitions']):
            raise ValueError('Neural inputs require completed neural scores')
        scored = dict(schema='prediction_scores_v1', target=inputs['target'], model=None,
                      threshold=inputs['threshold'], input_run_sha256=sha256(input_directory / 'run.json'),
                      partitions=[dict(chrom=part['chrom'], strand=part['strand'], rows=0)
                                  for part in inputs['partitions']])
    else:
        score_directory = Path(score_directory)
        scored = completed_run(score_directory)
    if inputs.get('schema') != 'prediction_inputs_v9' or scored.get('schema') != 'prediction_scores_v1':
        raise ValueError('Use completed prepared inputs and isolated neural scores')
    if (scored['input_run_sha256'] != sha256(input_directory / 'run.json')
            or scored['target'] != inputs['target'] or scored['model'] != inputs['model']
            or scored['threshold'] != inputs['threshold']):
        raise ValueError('Scores do not belong to this prepared input/model bundle')
    score_parts = {(part['chrom'], part['strand']): part for part in scored['partitions']}
    input_keys = {(part['chrom'], part['strand']) for part in inputs['partitions']}
    if (len(input_keys) != len(inputs['partitions'])
            or len(score_parts) != len(scored['partitions']) or set(score_parts) != input_keys):
        raise ValueError('Input and score chromosome/strand partitions disagree')
    catalog = reference_catalog if isinstance(reference_catalog, ReferenceCatalog) else ReferenceCatalog(reference_catalog)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE.txt').write_text('Chromosome filtering not complete.', encoding='utf-8')
    partitions, directories = [], []
    try:
        for part in inputs['partitions']:
            key = part['chrom'], part['strand']
            score_part = score_parts[key]
            if score_part['rows'] != part['input_count']:
                raise ValueError('Input and score partition counts disagree')
            site_file = partition_file(input_directory, part['sites_path'], part['sites_size_bytes'])
            scores = {}
            if score_directory is not None:
                score_file = partition_file(score_directory, score_part['path'], score_part['size_bytes'])
                scores = read_partition_scores(score_file, part['input_count'])
            relative = Path(part['sites_path']).parent
            directory = out / relative
            directory.parent.mkdir(parents=True, exist_ok=True)
            temporary_index = directory.parent / (directory.name + '.sites.tsv.partial')
            fields, expected_scores = write_partition_sites(site_file, temporary_index, *key)
            if set(scores) != expected_scores:
                raise ValueError('Scored IDs differ from the prepared readable sites')
            write_predictions(temporary_index, scores, directory, catalog, reference_cell, condition,
                              inputs['threshold'], site_fields=fields, source_sites=site_file,
                              model_provenance=dict(model=inputs['model'],
                                  scores=str(score_directory.resolve()) if score_directory is not None else None))
            temporary_index.unlink()
            directories.append(directory)
            partitions.append(dict(chrom=key[0], strand=key[1],
                                   path=str(relative / 'all_predictions.tsv')))
        combine_partition_predictions(directories, out, list(dict.fromkeys(SITE_FIELDS + ['regions'] + DECISION_FIELDS)))
        report = dict(status='complete', schema='partitioned_predictions_v1', target=inputs['target'],
                      input_directory=str(input_directory.resolve()),
                      input_run_sha256=sha256(input_directory / 'run.json'),
                      score_directory=str(score_directory.resolve()) if score_directory is not None else None,
                      score_run_sha256=sha256(score_directory / 'run.json') if score_directory is not None else None,
                      threshold=inputs['threshold'], reference_cell=reference_cell, condition=condition,
                      partitions=partitions, interpretation='NN decisions and reference rescue; not experimental truth')
        (out / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as error:
        (out / 'FAILED.txt').write_text(str(error), encoding='utf-8')
        raise
