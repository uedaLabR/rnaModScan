"""Reuse saved cell splits and raw feature offsets for a cell-only model.

Only small TSV/JSON metadata are read. Read-feature Parquet is neither opened
nor copied, and the original five-fold and production-model inputs stay intact.
"""
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from training.cell_datasets import SITE_FIELDS
from training.cross_validation import position_groups
from utils.io.artifacts import completed_run
from utils.nn.inputs import label_values
from utils.references.catalog import sha256

SAMPLE_FIELDS = ('sample_id', 'cell', 'reference_cell', 'condition')
IDENTITY_FIELDS = tuple(name for name in SITE_FIELDS if name != 'group_id')


def sample_metadata(samples):
    """Require explicit, unique sample identities instead of guessing from IDs."""
    result = {}
    for sample in samples:
        if any(not sample.get(name) for name in SAMPLE_FIELDS):
            raise ValueError('Cell datasets require explicit sample/cell/reference_cell/condition metadata')
        record = {name: sample[name] for name in SAMPLE_FIELDS}
        if record['sample_id'] in result or record['condition'] not in ('WT', 'IVT'):
            raise ValueError('Duplicate sample ID or invalid sample condition')
        result[record['sample_id']] = record
    return result


def checked_cell_sources(index_path, datasets_root, samples, training_cell, separate_cell, flank):
    """Bind the saved split to the exact current index and explicit metadata."""
    index_run = completed_run(index_path.parent)
    if index_run.get('schema') != 'site_bag_index_v3':
        raise ValueError('Re-extract site Parquet with saved sequence repeat masks')
    target = index_run['target']
    if (type(index_run.get('max_reads')) is not int or index_run['max_reads'] != 300
            or type(index_run.get('sampling_seed')) is not int
            or index_run['sampling_seed'] < 0):
        raise ValueError('Site-bag index must declare max_reads=300 and a nonnegative sampling seed')
    if not index_run.get('require_labels', True) or index_run['counts'].get('not_extracted_sites', 0):
        raise ValueError('Use a completed labeled feature index with no unextracted sites')
    datasets_run = completed_run(datasets_root)
    if (datasets_run.get('schema') != 'cell_datasets_v1'
            or datasets_run.get('training_cell') != training_cell
            or datasets_run.get('separate_cell') != separate_cell
            or datasets_run.get('flank') != flank):
        raise ValueError('Cell dataset scope or grouping window differs from the requested model')
    current_samples = sample_metadata(samples)
    if sample_metadata(datasets_run['samples']) != current_samples:
        raise ValueError('Current sample metadata differ from the saved cell datasets')
    if {sample['cell'] for sample in current_samples.values()} != {training_cell, separate_cell}:
        raise ValueError('Sample cells must match the explicit training and separate cells')
    sources = [source for source in datasets_run['inputs'] if source['target'] == target]
    source_index = dict(path=str(index_path), sha256=sha256(index_path),
                        run_sha256=sha256(index_path.parent / 'run.json'))
    if index_run.get('index_sha256') != source_index['sha256']:
        raise ValueError('Site-bag index checksum differs from its completed manifest')
    if (len(sources) != 1 or Path(sources[0]['path']).resolve() != index_path
            or any(sources[0].get(name) != source_index[name] for name in ('sha256', 'run_sha256'))):
        raise ValueError('Cell datasets do not belong to this exact feature index/run')
    return index_run, datasets_run, current_samples, source_index


def read_cell_manifests(datasets_root, run, samples, target, training_cell, separate_cell, flank):
    """Validate all target IDs, including the separately analyzed cell."""
    records, paths, hashes, sizes = {}, {}, {}, Counter()
    for split, cell in (('train', training_cell), ('validation', training_cell), ('separate', separate_cell)):
        relative = Path(cell) / target / ('sites.tsv' if split == 'separate' else split + '/sites.tsv')
        path = datasets_root / relative
        expected_hash = run['files_sha256'].get(str(relative))
        if expected_hash is None or sha256(path) != expected_hash:
            raise ValueError(f'Cell dataset manifest changed: {relative}')
        paths[split], hashes[str(relative)] = path, expected_hash
        with path.open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            if not set(SITE_FIELDS) <= set(reader.fieldnames or []):
                raise ValueError('Cell dataset manifest lacks site identity/group fields')
            for row in reader:
                sample = samples.get(row['sample_id'])
                start = int(row['start'])
                if (sample is None or sample['cell'] != cell or row['modification'] != target
                        or start < 0 or int(row['end']) != start + 1 or row['strand'] not in ('+', '-')
                        or not row['group_id'] or not row['row_id'] or row['row_id'] in records):
                    raise ValueError('Invalid, duplicate or wrong-cell ID in cell dataset manifest')
                for name in ('cell', 'condition'):
                    if name in row and row[name] != sample[name]:
                        raise ValueError('Manifest sample metadata differ from explicit sample metadata')
                if 'label' in row and row['label'] not in label_values(target):
                    raise ValueError('Invalid separate-cell label')
                row['split'] = split
                records[row['row_id']] = row
                sizes[split] += 1
    if not sizes['train'] or not sizes['validation']:
        raise ValueError('Cell model needs nonempty train and validation manifests')
    training = [row for row in records.values() if row['split'] != 'separate']
    for row in training:
        row['saved_group_id'] = row['group_id']
    position_groups(training, flank)
    splits_by_group = {}
    for row in training:
        previous = splits_by_group.setdefault(row['group_id'], row['split'])
        if previous != row['split']:
            raise ValueError('Overlapping training/validation windows cross position groups')
    # Group labels are checked after recomputing coordinates, so forged group
    # strings cannot hide overlapping train and validation windows.
    for row in training:
        if row['group_id'] != row['saved_group_id']:
            raise ValueError('Saved cell dataset group differs from the actual genomic window group')
        row.pop('saved_group_id')
    return records, paths, hashes, sizes


def checked_feature_source(directory, source, sample_id, flank):
    """Check extraction provenance without opening any read-feature Parquet."""
    run = completed_run(directory)
    if sha256(directory / 'run.json') != source['run_sha256']:
        raise ValueError('Feature extraction changed after building the original index')
    if (not isinstance(run.get('window_flank'), int) or isinstance(run['window_flank'], bool)
            or not 0 <= run['window_flank'] <= flank):
        raise ValueError('Use features within the grouped window')
    if run.get('schema') in ('site_bags_parquet_v3', 'site_bags_parquet_v4'):
        if (type(run.get('max_reads')) is not int or run['max_reads'] != 300
                or type(run.get('sampling_seed')) is not int
                or run['sampling_seed'] < 0):
            raise ValueError('Site bags must declare max_reads=300 and a nonnegative sampling seed')
    else:
        raise ValueError('Use site-bag Parquet v3/v4 with saved sequence repeat masks')
    if run.get('sample_id', sample_id) != sample_id:
        raise ValueError('Feature source sample differs from the indexed sample')
    return run


def write_cell_index(index_path, index_run, records, out, target, flank):
    """Stream the source index and preserve only the training-cell rows."""
    declared = {}
    for source in index_run['feature_sources']:
        directory = (index_path.parent / source['path']).resolve()
        if directory in declared:
            raise ValueError('Duplicate feature source in original index')
        declared[directory] = dict(source, path=str(directory))
    used, feature_runs, indexed, input_count = {}, {}, 0, 0
    resolved_sources, source_paths, site_locations = {}, {}, {}
    required = set(IDENTITY_FIELDS) | {'label', 'features_dir', 'read_offset', 'retained_depth',
                                      'eligible_depth', 'reference_sequence'}
    required.update(('site_parquet', 'site_row_group', 'site_row_offset'))
    with index_path.open(encoding='utf-8', newline='') as handle, out.open('w', encoding='utf-8', newline='') as dest:
        reader = csv.DictReader(handle, delimiter='\t')
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Feature index lacks labeled site or read-offset metadata')
        writer = csv.DictWriter(dest, reader.fieldnames, delimiter='\t')
        writer.writeheader()
        for row in reader:
            input_count += 1
            expected = records.pop(row['row_id'], None)
            if (expected is None or any(row[name] != expected[name] for name in IDENTITY_FIELDS)
                    or row['modification'] != target or row['label'] not in label_values(target)
                    or ('label' in expected and row['label'] != expected['label'])
                    or int(row['retained_depth']) < 1
                    or int(row['eligible_depth']) < int(row['retained_depth']) or int(row['read_offset']) < 0):
                raise ValueError('Source index and saved cell manifests disagree on site IDs, labels or read offsets')
            eligible, retained = int(row['eligible_depth']), int(row['retained_depth'])
            expected_retained = min(eligible, index_run['max_reads'])
            if retained != expected_retained:
                raise ValueError('Retained depth differs from the declared read sampling policy')
            if (not row['site_parquet'] or int(row['site_row_group']) < 0
                              or int(row['site_row_offset']) < 0):
                raise ValueError('Invalid per-site Parquet row location')
            if expected['split'] == 'separate':
                continue
            source_key = row['features_dir']
            if source_key not in resolved_sources:
                resolved_sources[source_key] = (index_path.parent / source_key).resolve()
            directory = resolved_sources[source_key]
            if directory not in declared:
                raise ValueError('Selected feature source is absent from original index provenance')
            if directory not in feature_runs:
                feature_runs[directory] = checked_feature_source(directory, declared[directory], row['sample_id'], flank)
                used[directory] = declared[directory]
                source_run = feature_runs[directory]
                for name in ('max_reads', 'sampling_seed'):
                    if source_run[name] != index_run[name]:
                        raise ValueError('Feature source sampling policy differs from parent index')
                source_paths[directory] = {(directory / 'sites' / item['path']).resolve()
                                           for item in source_run['parquet_files']['sites']}
            source_run = feature_runs[directory]
            if source_run.get('sample_id', row['sample_id']) != row['sample_id']:
                raise ValueError('Selected feature source contains a different sample')
            location_key = directory, row['site_parquet']
            if location_key not in site_locations:
                location = Path(row['site_parquet'])
                location = location.resolve() if location.is_absolute() else (directory / location).resolve()
                if not location.is_relative_to(directory) or location not in source_paths[directory]:
                    raise ValueError('Site Parquet location is outside the declared feature source')
                site_locations[location_key] = location
            if len(row['reference_sequence']) != 2 * source_run['window_flank'] + 1:
                raise ValueError('Indexed reference window differs from raw extraction provenance')
            row['features_dir'] = str(directory)
            writer.writerow(row)
            indexed += 1
    if records:
        raise ValueError('Cell dataset manifest IDs are missing from the source index')
    if index_run['counts'].get('indexed_sites', input_count) != input_count:
        raise ValueError('Source index row count differs from its completed run')
    return indexed, [used[directory] for directory in sorted(used)]


def prepare_cell_model_data(index_path, datasets_root, samples, out, *,
                            training_cell='HEK293T', separate_cell='HeLa', flank=9):
    """Prepare an isolated index and saved train/validation IDs for one cell."""
    index_path, datasets_root, out = (Path(path).resolve() for path in (index_path, datasets_root, out))
    if out.exists():
        raise FileExistsError(out)
    if (not isinstance(flank, int) or isinstance(flank, bool) or flank < 0
            or training_cell == separate_cell or any(not cell or Path(cell).name != cell or cell in ('.', '..')
                                                     for cell in (training_cell, separate_cell))):
        raise ValueError('Use distinct explicit cells and a nonnegative grouping flank')
    index_run, datasets_run, metadata, source_index = checked_cell_sources(
        index_path, datasets_root, samples, training_cell, separate_cell, flank)
    target = index_run['target']
    records, paths, hashes, sizes = read_cell_manifests(
        datasets_root, datasets_run, metadata, target, training_cell, separate_cell, flank)
    sample_ids = sorted({row['sample_id'] for row in records.values() if row['split'] != 'separate'})
    scope = dict(cell=training_cell, sample_ids=sample_ids,
                 conditions=sorted({metadata[sample_id]['condition'] for sample_id in sample_ids}))
    source_datasets = dict(path=str(datasets_root), run_sha256=sha256(datasets_root / 'run.json'), files_sha256=hashes)
    out.mkdir(parents=True)
    (out / 'INCOMPLETE.txt').write_text('Cell model data preparation not complete.\n', encoding='utf-8')
    try:
        index_directory = out / 'index'
        index_directory.mkdir()
        selected_index = index_directory / 'sites.tsv'
        count, sources = write_cell_index(index_path, index_run, records, selected_index, target, flank)
        if count != sizes['train'] + sizes['validation'] or sha256(index_path) != source_index['sha256']:
            raise ValueError('Cell subset size or original feature index changed during preparation')
        if sha256(index_path.parent / 'run.json') != source_index['run_sha256']:
            raise ValueError('Original feature-index run changed during preparation')
        if sha256(datasets_root / 'run.json') != source_datasets['run_sha256']:
            raise ValueError('Cell dataset run changed during preparation')
        for relative, expected_hash in hashes.items():
            if sha256(datasets_root / relative) != expected_hash:
                raise ValueError('Cell dataset manifest changed during preparation')
        for source in sources:
            if sha256(Path(source['path']) / 'run.json') != source['run_sha256']:
                raise ValueError('Selected extraction provenance changed during preparation')
        split_paths = {}
        for split in ('train', 'validation'):
            destination = out / split / 'sites.tsv'
            destination.parent.mkdir()
            shutil.copyfile(paths[split], destination)
            relative = str(paths[split].relative_to(datasets_root))
            if sha256(destination) != hashes[relative]:
                raise ValueError('Saved cell manifest changed while copying IDs')
            split_paths[split] = str(destination)
        subset_run = dict(status='complete', target=target, require_labels=True,
                          counts=dict(input_sites=count, labeled_sites=count, indexed_sites=count, not_extracted_sites=0),
                          feature_sources=sources, source_index=source_index, training_scope=scope)
        subset_run.update(schema='site_bag_index_v3', max_reads=index_run['max_reads'],
                          sampling_seed=index_run['sampling_seed'], index_sha256=sha256(selected_index))
        (index_directory / 'run.json').write_text(json.dumps(subset_run, indent=2) + '\n', encoding='utf-8')
        report = dict(status='complete', schema='cell_model_data_v1', target=target,
                      index_path=str(selected_index), split_paths=split_paths, training_scope=scope,
                      sizes=dict(train=sizes['train'], validation=sizes['validation'], final=count),
                      source_index=source_index, source_cell_datasets=source_datasets,
                      index_sha256=sha256(selected_index),
                      policies=dict(parquet='reuse existing feature sources and row locations; no read-feature copy',
                                    separate_cell='excluded from fitting, validation and final refit',
                                    splits='saved cell manifests; actual genomic windows checked for leakage'))
        (out / 'run.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as error:
        (out / 'FAILED.txt').write_text(str(error) + '\n', encoding='utf-8')
        raise
