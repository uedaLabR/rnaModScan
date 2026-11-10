"""Snapshot common site Parquet for inference without re-encoding neural tensors."""
import csv
import json
import os
from pathlib import Path
import shutil

from prediction.input_parquet import (INPUT_NAMES, add_policy_sites, check_prediction_sources,
                                     metadata_record, metadata_schema, partition_directory)
from prediction.model_bundle import load_model_bundle, model_fingerprint
from utils.features.position import checked_position_annotations
from utils.features.parquet import require_arrow
from utils.features.site_dataset import LABEL_COLUMNS, SiteBagBatches
from utils.io.artifacts import completed_run
from utils.references.catalog import sha256
from utils.features.compact_inputs import tensor_column, SITE_SCHEMAS


SNAPSHOT_COLUMNS = ('row_id', 'chrom', 'start', 'strand', 'retained_depth') + LABEL_COLUMNS


def checked_source_shard(path, metadata_by_id, chrom, strand, pq):
    """Inspect metadata only; a whole-file snapshot must not introduce extra sites."""
    parquet = pq.ParquetFile(path)
    tensor_column(parquet.schema_arrow)
    positive_depth = 0
    for batch in parquet.iter_batches(batch_size=1024, columns=list(SNAPSHOT_COLUMNS)):
        for row in batch.to_pylist():  # Small reporting metadata only, never read tensors.
            saved = metadata_by_id.get(row['row_id'])
            if saved is None:
                raise ValueError('A source shard contains sites absent from the prediction index')
            if ((row['chrom'], row['strand']) != (chrom, strand)
                    or int(saved['start']) != row['start']
                    or int(saved['retained_depth']) != row['retained_depth']):
                raise ValueError('Site shard coordinates or read count disagree with the prediction index')
            if any(row[name] is not None for name in LABEL_COLUMNS):
                raise ValueError('Inference site bags must have null training labels')
            positive_depth += row['retained_depth'] > 0
    return positive_depth


def indexed_shards(rows, index_path, allowed_files):
    """Resolve each distinct metadata location once, including zero-read sites."""
    paths = set()
    for source_name, relative in {(row.get('features_dir'), row.get('site_parquet')) for row in rows
                                  if row.get('site_parquet') and row.get('filter_mode') != 'reference_only'}:
        source = (index_path.parent / source_name).resolve()
        path = Path(relative)
        path = path.resolve() if path.is_absolute() else (source / path).resolve()
        if not path.is_relative_to(source) or path not in allowed_files:
            raise ValueError('Site snapshot path is outside the completed feature sources')
        if path.stat().st_size != allowed_files[path]:
            raise ValueError('Site snapshot source changed after extraction')
        paths.add(path)
    return sorted(paths)


def snapshot_shard(source, destination):
    """Prefer a hard link; copying across filesystems keeps inference self-contained."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return 'hardlink'
    except OSError:
        shutil.copyfile(source, destination)
        return 'copy'


def prepare_site_bag_prediction_inputs(index_path, model_directory, out, *, position_sites=None,
                                       batch_sites=4, policy_sites=None):
    """Write reporting metadata and link/copy existing numeric site-bag shards.

    This CPU phase never loads TensorFlow, BAM, raw read tensors, or learned
    weights. The prepared bundle remains usable after source paths are removed.
    """
    if type(batch_sites) is not int or batch_sites < 1:
        raise ValueError('batch_sites must be positive')
    pa, pq = require_arrow()
    index_path, model_directory, out = Path(index_path).resolve(), Path(model_directory).resolve(), Path(out)
    _, model_run = load_model_bundle(model_directory)
    index_run = completed_run(index_path.parent)
    if index_run.get('schema') != 'site_bag_index_v3':
        raise ValueError('Use the common site-bag feature index')
    if index_run.get('index_sha256') != sha256(index_path):
        raise ValueError('Prediction site-bag index changed after completion')
    check_prediction_sources(index_path, model_run)
    allowed_files = {}
    source_layouts = set()
    for item in index_run['feature_sources']:
        source = Path(item['path']).resolve()
        run = completed_run(source)
        if run.get('schema') not in SITE_SCHEMAS:
            raise ValueError('Expected common site-bag Parquet v3 or v4')
        source_layouts.add(run['schema'])
        if (run.get('max_reads') != model_run['input_schema'].get('max_reads')
                or run.get('sampling_seed') != model_run['input_schema'].get('sampling_seed', 42)):
            raise ValueError('Site snapshot sampling policy differs from model settings')
        for file in run['parquet_files']['sites']:
            allowed_files[(source / 'sites' / file['path']).resolve()] = file['size_bytes']
    layout = 'site_bags_parquet_v4' if 'site_bags_parquet_v4' in source_layouts else 'site_bags_parquet_v3'
    positions, position_source = checked_position_annotations(position_sites)
    dataset = SiteBagBatches(index_path, None, batch_sites=batch_sites, require_labels=False,
                            max_reads=model_run['input_schema'].get('max_reads'),
                            sampling_seed=model_run['input_schema'].get('sampling_seed', 42))
    partitions, metadata_by_id = {}, {}
    with index_path.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row['row_id'] in metadata_by_id or row['modification'] != model_run['target']:
                raise ValueError('Duplicate row ID or mixed target in prediction index')
            if any(row.get(name) not in (None, '') for name in LABEL_COLUMNS) or row.get('label') not in (None, ''):
                raise ValueError('Inference index must have null training labels')
            metadata_by_id[row['row_id']] = row
            key = row['chrom'], row['strand']
            partitions.setdefault(key, dict(metadata=[], rows=[]))['metadata'].append(row)
    for row in dataset.rows:
        partitions[row['chrom'], row['strand']]['rows'].append(row)
    add_policy_sites(partitions, policy_sites, set(metadata_by_id), model_run['target'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE.txt').write_text('Site-bag snapshot is incomplete.\n')
    manifest = []
    try:
        site_schema = metadata_schema()
        for (chrom, strand), partition in sorted(partitions.items()):
            relative = partition_directory(chrom, strand)
            directory = out / relative
            directory.mkdir(parents=True)
            sites_path = directory / 'sites.parquet'
            metadata = [metadata_record(row, positions, site_schema) for row in partition['metadata']]
            pq.write_table(pa.Table.from_pylist(metadata, schema=site_schema), sites_path,
                           compression='zstd', row_group_size=1024)
            sources = indexed_shards(partition['metadata'], index_path, allowed_files)
            shards, input_count = [], 0
            for source in sources:
                count = checked_source_shard(source, metadata_by_id, chrom, strand, pq)
                if count == 0:
                    continue
                destination = directory / 'inputs' / f'part-{len(shards):06d}.parquet'
                method = snapshot_shard(source, destination)
                shards.append(dict(path=str(destination.relative_to(out)), size_bytes=destination.stat().st_size,
                                   source_path=str(source), snapshot_method=method))
                input_count += count
            if input_count != len(partition['rows']):
                raise ValueError('Snapshot input count differs from the candidate index')
            manifest.append(dict(chrom=chrom, strand=strand, layout=layout,
                                 sites_path=str(sites_path.relative_to(out)), input_shards=shards,
                                 site_count=len(metadata), input_count=input_count,
                                 sites_size_bytes=sites_path.stat().st_size))
        report = dict(status='complete', schema='prediction_inputs_v9', target=model_run['target'],
                      layout=layout, input_schema=model_run['input_schema'],
                      threshold=model_run['threshold'], model=model_fingerprint(model_directory, model_run),
                      index_path=str(index_path), index_sha256=sha256(index_path),
                      policy_sites_sha256=sha256(policy_sites) if policy_sites is not None else None,
                      position_annotation=position_source, partitions=manifest,
                      policies=dict(reads='reuse sampled site bags; no additional downsampling or padding on disk',
                                    model_inputs=list(INPUT_NAMES), snapshot='hardlink or copy existing Parquet; no tensor rewrite',
                                    annotations='reporting metadata only', partition='target/chromosome/strand'))
        (out / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as error:
        (out / 'FAILED.txt').write_text(str(error) + '\n')
        raise
