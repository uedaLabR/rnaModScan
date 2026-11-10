"""Evaluate saved FP-labelled observations by origin and modification."""
from collections import Counter
import csv
import json
from pathlib import Path

from utils.references.catalog import sha256

ORIGINS = [('paired_IVT', 'm6A'), ('paired_IVT', 'inosine'),
           ('additional_IVT', 'm6A'), ('additional_IVT', 'inosine'),
           ('additional_IVT', 'm6A+inosine'), ('WT_IVT_override', 'm6A+inosine')]


def site_key(row):
    return tuple(row[name] for name in ('cell', 'chrom', 'start', 'strand'))


def threshold_pass(row, modification):
    depth, modified, af = [row.get(name + '_' + modification, '')
                           for name in ('total_reads', 'modified_reads', 'af')]
    return bool(depth and modified and af and int(depth) >= 10 and int(modified) >= 3 and float(af) >= .1)


def negative_origins(index_path):
    """Paired IVT takes priority, including when its own calls pass thresholds."""
    positives, paired, origins = {}, {}, {}
    with Path(index_path).open(newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row['label'] in ('0', '1'):
                assert row['condition'] == 'WT'
                positives[site_key(row)] = 'm6A' if row['label'] == '0' else 'inosine'
            elif row['selection_reason'] == 'paired_ivt':
                assert row['condition'] == 'IVT'
                paired[row['row_id']] = site_key(row)
            elif row['condition'] == 'WT':
                assert row['label_reason'] == 'both_targets_negative'
                origins[row['row_id']] = ('WT_IVT_override', 'm6A+inosine')
            else:
                assert row['condition'] == 'IVT' and row['label_reason'] == 'ivt_negative'
                called = [mod for mod in ('m6A', 'inosine') if threshold_pass(row, mod)]
                assert called, row['row_id']
                origins[row['row_id']] = ('additional_IVT', '+'.join(called))
    for row_id, key in paired.items():
        origins[row_id] = ('paired_IVT', positives[key])
    return origins


def negative_origin_sheets(run_dir):
    run_dir = Path(run_dir)
    run = json.loads((run_dir / 'adenine/fold_0/run.json').read_text())
    index = Path(run['index_path'])
    assert sha256(index) == run['index_sha256']
    origins = negative_origins(index)
    totals = Counter(origins.values())
    full_rows = [[origin, mod, totals[origin, mod], totals[origin, mod] / len(origins)]
                 for origin, mod in ORIGINS]
    rows, sources = [], [(str(index), run['index_sha256'])]
    for fold in range(5):
        for split in ('validation', 'test'):
            directory = run_dir / 'adenine' / f'fold_{fold}' / split
            path = directory / 'predictions.tsv'
            counts = Counter()
            with path.open(newline='') as handle:
                for row in csv.DictReader(handle, delimiter='\t'):
                    if row['label'] == 'false_positive':
                        origin, mod = origins[row['row_id']]
                        counts[origin, mod, row['predicted_class']] += 1
            metrics = json.loads((directory / 'metrics.json').read_text())
            expected = metrics['per_class']['false_positive']
            assert sum(counts.values()) == expected['support']
            assert sum(count for (origin, mod, predicted), count in counts.items()
                       if predicted == 'false_positive') == expected['tp']
            for origin, mod in ORIGINS:
                predicted = [counts[origin, mod, label] for label in ('m6A', 'inosine', 'false_positive')]
                total = sum(predicted)
                correct = predicted[2] / total if total else None
                rows.append([fold, split, origin, mod, total, *predicted, correct, correct])
            sources.append((str(path), sha256(path)))
    return [dict(name='FP_source_counts', headers=['origin', 'modification', 'all_sites', 'fraction_of_all_FP'], rows=full_rows),
            dict(name='FP_best_by_source', headers=['fold', 'split', 'origin', 'modification', 'FP_labelled_sites',
                'predicted_m6A', 'predicted_inosine', 'predicted_FP', 'FP_recall', 'conditional_accuracy'], rows=rows)], sources
