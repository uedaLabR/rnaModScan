"""HEK293T train/validation manifests and a separate HeLa analysis list.

Read feature-index metadata only. Labels, BAM and read features do not determine
the split; the existing grouped five-fold datasets are left intact.
"""
import csv
import hashlib
import json
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

from training.cross_validation import position_groups
from utils.io.artifacts import completed_run
from utils.references.catalog import sha256
from utils.nn.inputs import label_values

SITE_FIELDS = ['row_id', 'sample_id', 'chrom', 'start', 'end', 'strand', 'modification', 'group_id']
SEPARATE_FIELDS = SITE_FIELDS + ['cell', 'condition', 'label', 'overlap_training_window', 'overlap_validation_window']


def read_index_rows(index_paths, samples, flank):
    """Use explicit sample metadata, never infer HEK293T from a folder name."""
    sample_by_id = {sample['sample_id']: sample for sample in samples}
    if len(sample_by_id) != len(samples):
        raise ValueError('Duplicate sample_id in configuration')
    rows, inputs, seen = [], [], set()
    for target, path in index_paths.items():
        path = Path(path).resolve()
        run = completed_run(path.parent)
        if (run['target'] != target or not run.get('require_labels', True)
                or run['counts'].get('not_extracted_sites', 0)):
            raise ValueError('Use completed labeled feature indices for all targets')
        for source in run['feature_sources']:
            feature_run = completed_run(source['path'])
            if feature_run['window_flank'] > flank:
                raise ValueError('Grouping radius must cover the feature window')
            if sha256(Path(source['path']) / 'run.json') != source['run_sha256']:
                raise ValueError('Feature extraction changed after indexing')
        with path.open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                sample = sample_by_id.get(row['sample_id'])
                if sample is None:
                    raise ValueError(f'Unknown sample_id: {row["sample_id"]}')
                start = int(row['start'])
                if (row['row_id'] in seen or row['modification'] != target or row['label'] not in label_values(target)
                        or start < 0 or int(row['end']) != start + 1 or row['strand'] not in ('+', '-')
                        or int(row.get('retained_depth', 1)) < 1):
                    raise ValueError('Invalid, duplicate or zero-read labeled site')
                seen.add(row['row_id'])
                record = {name: row[name] for name in SITE_FIELDS if name != 'group_id'}
                record.update(cell=sample['cell'], condition=sample['condition'], label=row['label'])
                rows.append(record)
        inputs.append(dict(target=target, path=str(path), sha256=sha256(path),
                           run_sha256=sha256(path.parent / 'run.json')))
    return rows, inputs


def assign_training_groups(rows, flank, validation_fraction, seed):
    """Only training-cell positions decide the train/validation assignment."""
    position_groups(rows, flank)
    groups = sorted({row['group_id'] for row in rows},
                    key=lambda group: hashlib.sha256(f'{seed}|cell_validation|{group}'.encode()).hexdigest())
    if len(groups) < 2:
        raise ValueError('Need at least two nonoverlapping training-cell groups')
    count = max(1, min(len(groups) - 1, round(len(groups) * validation_fraction)))
    validation = set(groups[:count])
    for row in rows:
        row['split'] = 'validation' if row['group_id'] in validation else 'train'


def nearby_training_positions(rows):
    positions = defaultdict(list)
    for row in rows:
        positions[row['chrom'], row['split']].append(int(row['start']))
    return {key: sorted(set(values)) for key, values in positions.items()}


def overlaps_window(positions, chrom, start, split, flank):
    values = positions.get((chrom, split), [])
    index = bisect_left(values, start - 2 * flank)
    return int(index < len(values) and values[index] <= start + 2 * flank)


def write_sites(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row['chrom'], int(row['start']), row['row_id'])))


def build_cell_datasets(index_paths, samples, out, *, training_cell='HEK293T', separate_cell='HeLa',
                        flank=9, validation_fraction=.1, seed=42):
    """Reuse shared Parquet via row IDs; HeLa is never a training-time test set."""
    if (not 0 < validation_fraction < 1 or flank < 0 or training_cell == separate_cell
            or not training_cell or not separate_cell):
        raise ValueError('Use distinct cells, a nonnegative flank and validation fraction in (0,1)')
    for cell in (training_cell, separate_cell):
        if Path(cell).name != cell or cell in ('.', '..'):
            raise ValueError('Cell names must be simple directory names')
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    rows, inputs = read_index_rows(index_paths, samples, flank)
    training = [row for row in rows if row['cell'] == training_cell]
    separate = [row for row in rows if row['cell'] == separate_cell]
    if len(training) + len(separate) != len(rows):
        raise ValueError('Index contains cells outside the selected training/separate cells')
    assign_training_groups(training, flank, validation_fraction, seed)
    # Separate-cell coordinates cannot move a training row between train and validation.
    position_groups(separate, flank)
    positions = nearby_training_positions(training)
    for row in separate:
        row['split'] = 'separate_analysis'
        row['overlap_training_window'] = overlaps_window(positions, row['chrom'], int(row['start']), 'train', flank)
        row['overlap_validation_window'] = overlaps_window(positions, row['chrom'], int(row['start']), 'validation', flank)
    out.mkdir(parents=True)
    (out / 'INCOMPLETE.txt').write_text('Cell dataset creation not complete.', encoding='utf-8')
    counts, hashes, warnings = Counter(), {}, []
    try:
        for target in index_paths:
            for split in ('train', 'validation'):
                selected = [row for row in training if row['modification'] == target and row['split'] == split]
                path = out / training_cell / target / split / 'sites.tsv'
                write_sites(path, selected, SITE_FIELDS)
                hashes[str(path.relative_to(out))] = sha256(path)
                if {row['label'] for row in selected} != label_values(target):
                    warnings.append(f'{training_cell}/{target}/{split}: missing one or both labels')
            selected = [row for row in separate if row['modification'] == target]
            path = out / separate_cell / target / 'sites.tsv'
            write_sites(path, selected, SEPARATE_FIELDS)
            hashes[str(path.relative_to(out))] = sha256(path)
        for row in training + separate:
            counts[row['modification'], row['cell'], row['split'], row['sample_id'], row['condition'], row['label']] += 1
        with (out / 'counts.tsv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['modification', 'cell', 'split', 'sample_id', 'condition', 'label', 'sites'])
            writer.writerows([*key, value] for key, value in sorted(counts.items()))
        write_sites(out / 'training_assignments.tsv', training, SITE_FIELDS + ['split'])
        sample_metadata = [{name: sample[name] for name in ('sample_id', 'cell', 'reference_cell', 'condition')}
                           for sample in samples]
        report = dict(status='complete', schema='cell_datasets_v1', inputs=inputs, samples=sample_metadata,
                      training_cell=training_cell, separate_cell=separate_cell, seed=seed, flank=flank,
                      validation_fraction_of_training_groups=validation_fraction, files_sha256=hashes,
                      sizes=dict(train=sum(row['split'] == 'train' for row in training),
                                 validation=sum(row['split'] == 'validation' for row in training),
                                 separate_analysis=len(separate)), warnings=warnings,
                      policies=dict(assignment='training-cell positions only; no labels, AF, features or separate-cell rows',
                                    grouping='overlapping genomic windows across targets, strands and WT/IVT stay together',
                                    separate='not used for training, early stopping, threshold selection or training-time test',
                                    shared_loci='retained; per-row training/validation window overlap flags for later analysis',
                                    labels='existing operational weak labels; not independently validated truth',
                                    parquet='reuse original feature indices and shared Parquet; no read-feature copies'))
        (out / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as error:
        (out / 'FAILED.txt').write_text(str(error), encoding='utf-8')
        raise
