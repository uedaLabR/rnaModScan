"""Prepare exact NN tensors on CPU; persist chromosome/strand Parquet partitions."""
import csv
import json
from pathlib import Path
from urllib.parse import quote

from utils.features.position import checked_position_annotations
from utils.features.index import INDEX_FIELDS
from utils.features.parquet import require_arrow
from utils.features.position import POSITION_CHANNEL_NAMES, POSITION_FIELDS
from utils.io.artifacts import completed_run
from utils.features.site_metadata import metadata_fields
from prediction.confidence import AUDIT_FIELDS, BASE_FIELDS
from utils.references.catalog import sha256

INPUT_NAMES = ('sequence_tokens', 'read_features', 'read_mask', 'sequence_repeat_mask')
SITE_FIELDS = list(INDEX_FIELDS) + [name for name in POSITION_FIELDS if name not in INDEX_FIELDS]
SITE_FIELDS += [name for name in AUDIT_FIELDS if name not in SITE_FIELDS]
SITE_FIELDS += [field.name for field in metadata_fields() if field.name not in SITE_FIELDS]
SITE_FIELDS += ['mettl16_motif', 'policy_motif_code', 'policy_motif_sequence', 'auxiliary_motif_eligible']


def partition_directory(chrom, strand):
    """Escape contig names and retain the original name in the manifest."""
    if not chrom or strand not in ('+', '-'):
        raise ValueError('A chromosome and explicit strand are required')
    return Path('chrom=' + quote(chrom, safe='')) / ('strand=plus' if strand == '+' else 'strand=minus')


def metadata_schema():
    pa, _ = require_arrow()
    integers = {'start', 'end', 'read_offset', 'retained_depth', 'eligible_depth',
                'total_reads', 'modified_reads', 'transcript_count', 'gene_count'}
    integers.update(POSITION_CHANNEL_NAMES)
    fields = []
    support_types = {field.name: field.type for field in metadata_fields()}
    for name in SITE_FIELDS:
        count_field = any(name == base or name.startswith(base + '_') for base in BASE_FIELDS if base != 'af_pass')
        if name in support_types:
            dtype = support_types[name]
        elif name in integers or name.startswith('call_count_') or count_field:
            dtype = pa.int64()
        elif name == 'af' or name.startswith('af_'):
            dtype = pa.float64()
        else:
            dtype = pa.string()
        fields.append((name, dtype))
    return pa.schema(fields, metadata={b'schema_version': b'prediction_sites_v1'})


def metadata_record(row, positions, schema):
    """Keep gene/region/count annotations alongside coordinates, outside NN tensors."""
    annotation = positions.get((row['chrom'], int(row['start']), row['strand']), {})
    record = dict(row)
    record.update(annotation)
    for name in POSITION_CHANNEL_NAMES:
        record.setdefault(name, 0)
    result = {}
    for field in schema:
        value = record.get(field.name)
        if value in (None, ''):
            result[field.name] = None
        elif str(field.type) == 'bool':
            if value not in (True, False, 'True', 'False', '1', '0'):
                raise ValueError('Invalid support flag: ' + field.name)
            result[field.name] = value in (True, 'True', '1')
        elif str(field.type).startswith('list<'):
            result[field.name] = value.split(';') if isinstance(value, str) else value
        elif str(field.type) == 'int64':
            result[field.name] = int(value)
        elif str(field.type) == 'double':
            result[field.name] = float(value)
        else:
            result[field.name] = str(value)
    return result


def check_prediction_sources(index_path, model_run):
    index_run = completed_run(index_path.parent)
    if index_run['target'] != model_run['target']:
        raise ValueError('Prediction index target does not match final model target')
    for source in index_run['feature_sources']:
        directory = Path(source['path'])
        if sha256(directory / 'run.json') != source['run_sha256']:
            raise ValueError('Feature run changed after the prediction index was built')
        feature_run = completed_run(directory)
        if 2 * feature_run['window_flank'] + 1 != model_run['input_schema']['window']:
            raise ValueError('Prediction feature window differs from the trained model')


def prepare_prediction_inputs(index_path, model_directory, out, *, position_sites=None,
                              batch_sites=4, policy_sites=None):
    """Snapshot the four saved NN tensors; no BAM/FASTA access or re-encoding."""
    from prediction.site_bag_inputs import prepare_site_bag_prediction_inputs
    return prepare_site_bag_prediction_inputs(index_path, model_directory, out,
        position_sites=position_sites, batch_sites=batch_sites, policy_sites=policy_sites)


def add_policy_sites(partitions, path, seen, target):
    """Policy candidates have metadata and an explicitly empty NN input."""
    if path is None:
        return
    with Path(path).open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if (row['row_id'] in seen or row['modification'] != target
                    or row['filter_mode'] != 'reference_only'):
                raise ValueError('Duplicate or invalid policy candidate')
            seen.add(row['row_id'])
            row.update(retained_depth=0, eligible_depth=0, read_offset=0)
            key = row['chrom'], row['strand']
            partitions.setdefault(key, dict(metadata=[], rows=[]))['metadata'].append(row)


def prepare_policy_inputs(site_path, target, out, *, position_sites=None):
    """Persist known-site/motif policy metadata without BAM or neural tensors."""
    pa, pq = require_arrow()
    positions, position_source = checked_position_annotations(position_sites)
    partitions = {}
    add_policy_sites(partitions, site_path, set(), target)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE.txt').write_text('Policy inputs not complete.\n')
    manifest = []
    try:
        for (chrom, strand), part in sorted(partitions.items()):
            relative = partition_directory(chrom, strand)
            directory = out / relative
            directory.mkdir(parents=True)
            sites_path = directory / 'sites.parquet'
            schema = metadata_schema()
            records = [metadata_record(row, positions, schema) for row in part['metadata']]
            pq.write_table(pa.Table.from_pylist(records, schema=schema), sites_path, compression='zstd')
            manifest.append(dict(chrom=chrom, strand=strand, sites_path=str(relative / sites_path.name),
                                 site_count=len(records), input_count=0,
                                 sites_size_bytes=sites_path.stat().st_size))
        report = dict(status='complete', schema='prediction_inputs_v9', target=target,
                      model=None, input_schema=None, threshold=.5, partitions=manifest,
                      policy_sites_sha256=sha256(site_path), position_annotation=position_source,
                      policies=dict(filtering='known-site overlap or enabled sequence motif; no neural score'))
        (out / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as error:
        (out / 'FAILED.txt').write_text(str(error) + '\n')
        raise


def checked_partition_path(directory, relative, size_bytes):
    """Manifest paths must stay inside their bundle and retain their file size."""
    root = Path(directory).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path.stat().st_size != size_bytes:
        raise ValueError('Prepared Parquet path or size mismatch')
    return path


