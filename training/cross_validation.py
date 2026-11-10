"""Five held-out test folds, with a small validation subset inside each training pool."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from utils.references.catalog import sha256
from utils.nn.inputs import label_values


def position_groups(rows, flank):
    """Combine overlapping windows across cells, conditions, targets AND strands."""
    previous_chrom, previous_start, group = None, -1, None
    for row in sorted(rows, key=lambda r: (r['chrom'], int(r['start']), r['row_id'])):
        start = int(row['start'])
        if row['chrom'] != previous_chrom or start - previous_start > 2 * flank:
            group = f"{row['chrom']}:{start}"
        row['group_id'] = group
        previous_chrom, previous_start = row['chrom'], start


def assign_position_folds(rows, flank=9, seed=42):
    """Balance row counts, never looking at labels or feature values."""
    position_groups(rows, flank)
    groups = defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(row)
    if len(groups) < 5:
        raise ValueError('At least five nonoverlapping locus groups are required')
    stable_hash = lambda text: hashlib.sha256(f'{seed}|{text}'.encode()).hexdigest()
    sizes = [0] * 5
    for group in sorted(groups, key=lambda g: (-len(groups[g]), stable_hash(g))):
        fold = min(range(5), key=lambda f: (sizes[f], stable_hash(f'{group}|{f}')))
        for row in groups[group]:
            row['fold'] = fold
        sizes[fold] += len(groups[group])


def split_groups(rows, fold, validation_fraction=0.1, seed=42):
    """Outer test ~20%; reserve 10% of the remaining groups for validation.

    Defaults approximate train/validation/test = 72/8/20. Entire groups stay
    together, so exact row fractions depend on group sizes. No labels are used.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must be between zero and one')
    remaining = sorted({r['group_id'] for r in rows if r['fold'] != fold},
                       key=lambda g: hashlib.sha256(f'{seed}|validation|{fold}|{g}'.encode()).hexdigest())
    if len(remaining) < 2:
        raise ValueError('Need at least two non-test groups for train and validation')
    number = max(1, min(len(remaining)-1, round(len(remaining) * validation_fraction)))
    validation = set(remaining[:number])
    return {r['row_id']: 'test' if r['fold'] == fold else
            'validation' if r['group_id'] in validation else 'train' for r in rows}


def build_cross_validation(index_paths, out, flank=9, seed=42, validation_fraction=0.1, target_outputs=None):
    """Read small metadata TSVs, write ID manifests; no BAM/read Parquet access.

    Make one global position assignment across the four targets. A target keeps
    both cell lines' observations and receives five independent split manifests.
    """
    from utils.features.index import completed_run
    out = Path(out)
    rows, ids, inputs = [], set(), []
    for target, path in index_paths.items():
        path = Path(path).resolve()
        run = completed_run(path.parent)
        if run['target'] != target or run['counts'].get('not_extracted_sites', 0):
            raise ValueError('Finish extraction of all labeled sites before making CV splits')
        for source in run['feature_sources']:
            feature_run = completed_run(source['path'])
            if feature_run['window_flank'] > flank:
                raise ValueError('Grouping radius must cover the extracted feature window')
        with path.open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                if row['row_id'] in ids or row['modification'] != target or row['label'] not in label_values(target):
                    raise ValueError('Invalid or duplicate feature-index row')
                ids.add(row['row_id'])
                rows.append({key: row[key] for key in ('row_id', 'sample_id', 'chrom', 'start', 'end',
                                                       'strand', 'modification', 'label')})
        inputs.append(dict(target=target, path=str(path), sha256=sha256(path)))
    assign_position_folds(rows, flank, seed)
    target_outputs = {target: Path(target_outputs[target]) if target_outputs else out / target
                      for target in index_paths}
    for directory in [out, *target_outputs.values()]:
        if directory.exists():
            raise FileExistsError(directory)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE.txt').write_text('Splits not complete.', encoding='utf-8')
    counts, warnings, sizes = Counter(), [], []
    try:
        assignments = [split_groups(rows, fold, validation_fraction, seed) for fold in range(5)]
        for target in index_paths:
            target_rows = [row for row in rows if row['modification'] == target]
            for fold in range(5):
                directory = target_outputs[target] / f'fold_{fold}'
                directory.mkdir(parents=True)
                for split in ('train', 'validation', 'test'):
                    selected = [row for row in target_rows if assignments[fold][row['row_id']] == split]
                    fields = ['row_id', 'sample_id', 'chrom', 'start', 'end', 'strand', 'modification', 'group_id']
                    split_directory = directory / split
                    split_directory.mkdir()
                    with (split_directory / 'sites.tsv').open('w', newline='', encoding='utf-8') as handle:
                        writer = csv.DictWriter(handle, fields, delimiter='\t', extrasaction='ignore')
                        writer.writeheader()
                        writer.writerows(selected)
                    labels = {row['label'] for row in selected}
                    sizes.append([target, fold, split, len(selected), len({r['group_id'] for r in selected}),
                                  sum(r['label'] in ('0', '1') if target == 'adenine' else r['label'] == '1' for r in selected),
                                  sum(r['label'] == ('2' if target == 'adenine' else '0') for r in selected)])
                    if labels != label_values(target):
                        warnings.append(f'{target}/fold_{fold}/{split}: labels={sorted(labels)}')
                    for row in selected:
                        counts[target, fold, split, row['sample_id'], row['label']] += 1
        with (out / 'split_counts.tsv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['modification', 'fold', 'split', 'sample_id', 'label', 'sites'])
            writer.writerows([(*key, value) for key, value in sorted(counts.items())])
        with (out / 'position_folds.tsv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['row_id', 'sample_id', 'chrom', 'start', 'strand', 'modification', 'group_id', 'fold'])
            writer.writerows([[row[k] for k in ('row_id', 'sample_id', 'chrom', 'start', 'strand',
                                                'modification', 'group_id', 'fold')] for row in rows])
        with (out / 'split_sizes.tsv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['modification', 'fold', 'split', 'sites', 'locus_groups', 'positive', 'negative'])
            writer.writerows(sizes)
        report = dict(status='complete', inputs=inputs, seed=seed, flank=flank, folds=5,
                      assignment='positions + observation counts only; no label/feature stratification',
                      rotation='test=i; validation=subset of non-test groups; train=remaining groups',
                      validation_fraction_of_non_test=validation_fraction,
                      approximate_fractions=dict(train=.8*(1-validation_fraction), validation=.8*validation_fraction, test=.2),
                      warnings=warnings, evaluation='internal grouped CV, not independent external testing')
        (out / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        for target, directory in target_outputs.items():
            hashes = {f'fold_{fold}/{split}/sites.tsv': sha256(directory / f'fold_{fold}' / split / 'sites.tsv')
                      for fold in range(5) for split in ('train', 'validation', 'test')}
            (directory / 'run.json').write_text(json.dumps(dict(report, target=target, split_sha256=hashes), indent=2), encoding='utf-8')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as exc:
        (out / 'FAILED.txt').write_text(str(exc), encoding='utf-8')
        raise
