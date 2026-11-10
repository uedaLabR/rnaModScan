"""Explore neighboring Inosine scores using saved HeLa site summaries only.

No BAM, FASTA, model or prediction is opened. Sites, rather than reads, are
the observations. Genomic-block checks acknowledge local dependence; these
post-hoc comparisons describe selection by a model using the same inputs.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.io.excel import verify_workbook, write_workbook

ENDPOINT = 'neighbor_canonical_ml_inosine'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False) + '\n')


def write_tsv(path, headers, rows):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(headers)
        writer.writerows(rows)


def read_candidates(path):
    """Read only original Inosine candidates and the already summarized score."""
    rows = []
    total = 0
    seen = set()
    with gzip.open(path, 'rt', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            total += 1
            if row['candidate_inosine'] != '1':
                continue
            assert row['row_id'] not in seen, 'Duplicate candidate row_id'
            seen.add(row['row_id'])
            value = float(row[ENDPOINT]) if row[ENDPOINT] else math.nan
            rows.append((row['row_id'], row['chrom'], row['strand'],
                         int(row['position_bed0']), row['predicted_class'],
                         row['predicted_class'] == 'inosine', value,
                         int(float(row['sampled_read_count']))))
    return rows, total


def read_annotations(path, rows):
    """Join an existing external-reference/repeat table by exact assembly key."""
    annotations = {}
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row['Dorado'] != '1':
                continue
            key = (row['chrom'], row['strand'], int(row['start']))
            assert key not in annotations, 'Duplicate original Dorado membership key'
            annotations[key] = {'slic_supported': row['Slic_seq'] == '1',
                                'any_Alu': row['any_Alu'] == '1',
                                'any_repeat': row['any_repeat'] == '1',
                                'full_total_reads': int(row['total_reads']),
                                'nn_retained': row['NN'] == '1'}
    assert len(annotations) == len(rows)
    for row in rows:
        annotation = annotations[(row[1], row[2], row[3])]
        assert annotation['nn_retained'] == row[5], 'Historical NN decisions disagree'
    return annotations


def normal_log_survival(z):
    """Stable log normal upper tail; asymptotic Mills expansion beyond 26 SD."""
    z = abs(float(z))
    if z <= 26:
        return math.log(math.erfc(z / math.sqrt(2)) / 2)
    term, correction = 1.0, 1.0
    for index in range(1, 20):
        term *= -(2 * index - 1) / (z * z)
        correction += term
        if abs(term) < 1e-16:
            break
    return -z * z / 2 - math.log(z) - math.log(2 * math.pi) / 2 + math.log(correction)


def two_sided_normal_p(z):
    logp = min(0.0, math.log(2) + normal_log_survival(z))
    numeric = math.exp(logp)
    log10p = logp / math.log(10)
    if not numeric:
        numeric = None
    display = 'p < 1e-300' if log10p < -300 else f'p = {numeric:.6g}'
    return {'p_two_sided': numeric, 'log10_p_two_sided': log10p,
            'p_display': display}


def describe(values):
    q25, median, q75 = np.quantile(values, [.25, .5, .75])
    return {'n_valid_sites': len(values), 'mean_ml_div255': float(values.mean()),
            'sd_ml_div255': float(values.std(ddof=1)),
            'q25_ml_div255': float(q25), 'median_ml_div255': float(median),
            'q75_ml_div255': float(q75)}


def mann_whitney(x, y):
    """Tie-corrected asymptotic U with continuity correction, no SciPy needed."""
    values = np.concatenate((x, y))
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - (counts - 1) / 2
    nx, ny, n = len(x), len(y), len(values)
    u = float(ranks[inverse[:nx]].sum() - nx * (nx + 1) / 2)
    tie_sum = np.sum(counts.astype(np.longdouble) ** 3 - counts)
    variance = float(nx * ny / 12 * (n + 1 - tie_sum / (n * (n - 1))))
    centered = u - nx * ny / 2
    z = (centered - math.copysign(.5, centered)) / math.sqrt(variance)
    delta = 2 * u / (nx * ny) - 1
    return {'test': 'Mann-Whitney U, asymptotic, tie/continuity corrected',
            'n_retained': nx, 'n_rejected': ny, 'u_retained': u,
            'z': z, 'cliff_delta_retained_minus_rejected': delta,
            'probability_retained_gt_rejected_plus_half_ties': (delta + 1) / 2,
            **two_sided_normal_p(z),
            'interpretation': 'Nominal site-level p; correlated sites and post-hoc NN selection.'}


def block_checks(rows, width):
    """Cluster-robust two-group OLS mean difference, retaining all valid sites."""
    valid = [row for row in rows if math.isfinite(row[6])]
    y = np.asarray([row[6] for row in valid])
    keep = np.asarray([row[5] for row in valid])
    means = np.asarray([y[~keep].mean(), y[keep].mean()])
    keys = [(row[1], row[2], row[3] // width) for row in valid]
    groups, inverse = np.unique(np.asarray(keys, dtype='U40'), axis=0, return_inverse=True)
    n_blocks, n = len(groups), len(y)
    counts = np.zeros((n_blocks, 2), dtype=np.int64)
    sums = np.zeros((n_blocks, 2), dtype=float)
    for group in (0, 1):
        chosen = keep == bool(group)
        counts[:, group] = np.bincount(inverse[chosen], minlength=n_blocks)
        sums[:, group] = np.bincount(inverse[chosen], weights=y[chosen], minlength=n_blocks)
    residual_sums = sums - counts * means
    influence = residual_sums[:, 1] / keep.sum() - residual_sums[:, 0] / (~keep).sum()
    correction = n_blocks / (n_blocks - 1) * (n - 1) / (n - 2)
    se = math.sqrt(float(np.sum(influence ** 2) * correction))
    difference = float(means[1] - means[0])
    z = difference / se
    mixed = np.all(counts > 0, axis=1)
    within = sums[mixed, 1] / counts[mixed, 1] - sums[mixed, 0] / counts[mixed, 0]
    block_rows = []
    for index, key in enumerate(groups):
        mean0 = float(sums[index, 0] / counts[index, 0]) if counts[index, 0] else None
        mean1 = float(sums[index, 1] / counts[index, 1]) if counts[index, 1] else None
        block_rows.append([width, str(key[0]), str(key[1]), int(key[2]) * width,
                           int(counts[index, 1]), int(counts[index, 0]), mean1, mean0,
                           mean1 - mean0 if mixed[index] else None])
    positive, negative = int((within > 0).sum()), int((within < 0).sum())
    summary = {'test': 'Two-group OLS, CR1 cluster-robust SE, asymptotic normal Wald',
               'block_width_bp': width, 'n_sites': n, 'n_blocks': n_blocks,
               'n_mixed_blocks': int(mixed.sum()), 'mean_difference_ml_div255': difference,
               'cluster_robust_se': se, 'ci95_lower': difference - 1.959963984540054 * se,
               'ci95_upper': difference + 1.959963984540054 * se, 'z': z,
               **two_sided_normal_p(z), 'cr1_finite_sample_correction': correction,
               'mixed_block_positive_count': positive, 'mixed_block_negative_count': negative,
               'mixed_block_tie_count': int((within == 0).sum()),
               'mixed_block_median_difference_ml_div255': float(np.median(within)),
               'mixed_block_retained_sites': int(counts[mixed, 1].sum()),
               'mixed_block_rejected_sites': int(counts[mixed, 0].sum())}
    return summary, block_rows


def depth_summaries(rows):
    result = []
    for label, lower, upper in [('1-9', 1, 10), ('10-29', 10, 30), ('30-99', 30, 100),
                                ('100-299', 100, 300), ('300', 300, 301)]:
        for keep, decision in [(True, 'NN_retained'), (False, 'NN_rejected')]:
            chosen = [row for row in rows if row[5] == keep and lower <= row[7] < upper]
            values = np.asarray([row[6] for row in chosen if math.isfinite(row[6])])
            if not len(values):
                continue
            result.append({'sampled_read_bin': label, 'decision': decision,
                           'n_candidate_sites': len(chosen), **describe(values)})
    return result


def group_summaries(rows, group_names):
    descriptions, values = [], []
    for keep, decision in zip((True, False), group_names):
        selected = [row for row in rows if row[5] == keep]
        numeric = np.asarray([row[6] for row in selected if math.isfinite(row[6])])
        values.append(numeric)
        descriptions.append({'decision': decision, 'n_candidate_sites': len(selected),
                             'n_excluded_undefined_endpoint': len(selected) - len(numeric),
                             **describe(numeric),
                             'median_sampled_reads_all_candidates': float(np.median([row[7] for row in selected])),
                             'n_at_300_read_cap_all_candidates': sum(row[7] == 300 for row in selected)})
    return descriptions, values


def rename_slic_fields(result):
    return {key.replace('retained', 'slic_supported').replace('rejected', 'without_slic_support'): value
            for key, value in result.items()}


def table_from_dicts(name, rows):
    headers = list(rows[0])
    return {'name': name, 'headers': headers,
            'rows': [[row.get(key) for key in headers] for row in rows]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'paper/data/hela_feature_patterns_20261109/site_feature_metrics.tsv.gz')
    parser.add_argument('--membership', type=Path, default=ROOT / 'paper/data/hela_inosine_comparison_20261108/inosine_membership.tsv')
    parser.add_argument('--output', type=Path, default=ROOT / 'paper/data/inosine_neighbor_test_20261109')
    args = parser.parse_args()
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    before = sha256(args.source)
    membership_before = sha256(args.membership)
    rows, total = read_candidates(args.source)
    assert total == 507501 and len(rows) == 50062, (total, len(rows))
    annotations = read_annotations(args.membership, rows)
    descriptions, values = group_summaries(rows, ('NN_retained', 'NN_rejected'))
    assert [len(value) for value in values] == [8169, 40596]
    nominal = mann_whitney(*values)
    cluster_summaries, blocks = [], []
    for width in (1000, 10000):
        summary, block_rows = block_checks(rows, width)
        cluster_summaries.append(summary)
        blocks.extend(block_rows)
    depth = depth_summaries(rows)
    slic_rows = []
    for row in rows:
        support = annotations[(row[1], row[2], row[3])]['slic_supported']
        slic_rows.append((*row[:5], support, *row[6:]))
    slic_descriptions, slic_values = group_summaries(slic_rows, ('Slic_supported', 'Without_Slic_support'))
    assert [item['n_candidate_sites'] for item in slic_descriptions] == [1427, 48635]
    slic_nominal = rename_slic_fields(mann_whitney(*slic_values))
    slic_nominal['interpretation'] = 'Secondary post-hoc external-support association; not chemical ground truth; correlated sites.'
    slic_clusters, slic_blocks = [], []
    for width in (1000, 10000):
        summary, block_rows = block_checks(slic_rows, width)
        slic_clusters.append(rename_slic_fields(summary))
        slic_blocks.extend(block_rows)
    alu = []
    for is_alu, label in ((True, 'Alu'), (False, 'Not_Alu')):
        subset = [row for row in rows if annotations[(row[1], row[2], row[3])]['any_Alu'] == is_alu]
        summary, _ = group_summaries(subset, ('NN_retained', 'NN_rejected'))
        alu.extend([{'Alu_stratum': label, **item} for item in summary])
    definitions = [
        ['scope', 'Exploratory post-hoc association with historical raw NN selection among original HeLa WT Inosine candidates; final DB/short-read/DRACH rescue excluded.'],
        ['endpoint', 'Per-site mean Inosine ML/255 over real sampled reads and reference-A positions at offsets -9..-1,+1..+9; center excluded. Missing ML=0 as in NN inputs. No AF or biochemical occupancy interpretation.'],
        ['unit', 'One equally weighted site; no individual-read significance tests. All score/difference values are ML/255, range0..1, not percent.'],
        ['group', 'NN_retained: original Inosine candidate and raw softmax argmax inosine. NN_rejected includes argmax false_positive or m6A. No chemical positive/negative status asserted.'],
        ['undefined', 'No neighboring reference-A opportunity and/or no real reads yields undefined endpoint; excluded only from endpoint analyses, retained in candidate counts.'],
        ['sampling', 'Existing uniform random reservoir of at most300 real reads/site; no new resampling. Depth bins refer to sampled read count, not original full depth.'],
        ['mann_whitney', 'Two-sided asymptotic U comparing retained with rejected; tie-adjusted variance and continuity correction. Cliff delta=2U/(n_retained*n_rejected)-1. Nominal p assumes independent sites and is not the main robustness evidence.'],
        ['cluster', 'Mean difference retained minus rejected from OLS with intercept/group, CR1 sandwich SE by fixed genomic chromosome/strand floor(BED0 position/blockwidth) bins. All valid sites retained, including single-group blocks. Two-sided asymptotic normal Wald p and95% CI.'],
        ['cluster_correction', 'CR1 factor G/(G-1)*(N-1)/(N-2), with G observed blocks and N valid sites; residual sums formed separately by group within each block.'],
        ['mixed_blocks', 'Descriptive within-bin retained mean minus rejected mean only for bins containing both groups; this sensitivity subset differs from all candidates. No additional test or multiple-test claim.'],
        ['limits', 'Blocks reduce local pseudoreplication but do not ensure independence of distant sites/transcripts/shared long reads. No biological replicates, ground truth, permutation of NN decisions or causal inference. Width1kb primary robustness,10kb sensitivity, both chosen after observing descriptive association.'],
        ['multiplicity', 'All p-values are exploratory and unadjusted for multiple comparisons. Neither the hypothesis nor block widths were prespecified before observing these descriptive associations.'],
        ['depth', 'Descriptive score comparisons within sampled-read bins show whether direction persists across depth ranges; no covariate-adjusted causal claim.'],
        ['Slic_support', 'Secondary comparison independent of NN decision grouping: original Dorado candidates overlapping existing same-cell HeLa Slic-seq sites versus candidates without Slic-seq support. Membership is exact chromosome/strand/BED0 coordinate. Slic-seq absence is not a false-positive label and the catalogue is incomplete.'],
        ['Alu', 'Existing any_Alu annotation defines descriptive Alu/not-Alu strata of the NN comparison; no new RepeatMasker scan. Repeat association can confound overall score differences; no causal adjustment asserted.'],
        ['p_underflow', 'Two-sided p never stored as zero. Extreme tails retain finite log10 p and text bound p<1e-300; normal log survival uses asymptotic Mills expansion beyond26SD.'],
        ['interpretation', 'The NN uses neighboring modification scores itself, so higher scores in its retained candidates are a model-selection association, not independent evidence of Inosine enrichment or sensitivity/specificity.'],
        ['source', str(args.source.resolve())],
        ['source_sha256', before],
        ['membership_source', str(args.membership.resolve())],
        ['membership_sha256', membership_before],
    ]
    sheets = [{'name': 'Definitions', 'headers': ['item', 'definition'], 'rows': definitions,
               'widths': [32, 115]}, table_from_dicts('Group_summary', descriptions),
              table_from_dicts('Nominal_Mann_Whitney', [nominal]),
              table_from_dicts('Cluster_robust', cluster_summaries),
              table_from_dicts('Depth_sensitivity', depth),
              table_from_dicts('Slic_group_summary', slic_descriptions),
              table_from_dicts('Slic_nominal_MW', [slic_nominal]),
              table_from_dicts('Slic_cluster_robust', slic_clusters),
              table_from_dicts('Alu_sensitivity', alu),
              {'name': 'Block_means', 'headers': ['block_width_bp', 'chrom', 'strand', 'start_bed0',
                 'n_retained', 'n_rejected', 'retained_mean_ml_div255', 'rejected_mean_ml_div255',
                 'within_block_mean_difference_ml_div255'], 'rows': blocks}]
    sheets.append({'name': 'Slic_block_means', 'headers': ['block_width_bp', 'chrom', 'strand', 'start_bed0',
                   'n_slic_supported', 'n_without_slic_support', 'slic_supported_mean_ml_div255',
                   'without_slic_support_mean_ml_div255', 'within_block_mean_difference_ml_div255'],
                   'rows': slic_blocks})
    for sheet in sheets:
        write_tsv(args.output / (sheet['name'] + '.tsv'), sheet['headers'], sheet['rows'])
    site_headers = ['row_id', 'chrom', 'strand', 'position_bed0', 'predicted_class',
                    'nn_retained', ENDPOINT, 'sampled_read_count', 'slic_supported', 'any_Alu',
                    'any_repeat', 'full_total_reads']
    clean_rows = []
    for row in rows:
        a = annotations[(row[1], row[2], row[3])]
        clean_rows.append([*row[:6], float(row[6]) if math.isfinite(row[6]) else None, row[7],
                           a['slic_supported'], a['any_Alu'], a['any_repeat'], a['full_total_reads']])
    write_tsv(args.output / 'inosine_candidate_endpoints.tsv.gz', site_headers, clean_rows)
    book = args.output / 'HeLa_Inosine_neighbor_score_tests_20261109.xlsx'
    write_workbook(book, sheets, float_format='General')
    restored = verify_workbook(book)
    assert set(restored) == {sheet['name'] for sheet in sheets}
    for sheet in sheets:
        assert restored[sheet['name']] == [sheet['headers'], *sheet['rows']], sheet['name']
    assert sha256(args.source) == before, 'Source changed during analysis'
    assert sha256(args.membership) == membership_before, 'Membership changed during analysis'
    report = {'schema': 'inosine_neighbor_posthoc_association_v1', 'status': 'complete',
              'source': str(args.source.resolve()), 'source_size_bytes': args.source.stat().st_size,
              'source_sha256_before': before, 'source_sha256_after': sha256(args.source),
              'source_total_sites': total, 'candidate_sites': len(rows),
              'endpoint': ENDPOINT, 'endpoint_center_excluded': True,
              'raw_NN_decision_before_rescue': True, 'groups': descriptions,
              'nominal_mann_whitney': nominal, 'cluster_checks': cluster_summaries,
              'depth_sensitivity': depth, 'source_script_sha256': sha256(__file__),
              'slic_group_summary': slic_descriptions, 'slic_nominal_mann_whitney': slic_nominal,
              'slic_cluster_checks': slic_clusters, 'Alu_sensitivity': alu,
              'membership_source': str(args.membership.resolve()),
              'membership_sha256_before': membership_before, 'membership_sha256_after': sha256(args.membership),
              'python': platform.python_version(), 'numpy': np.__version__,
              'elapsed_seconds': time.monotonic() - start,
              'workbook_native_cells_roundtrip_exact': True,
              'p_values_exploratory_unadjusted': True,
              'hypothesis_and_block_widths_prespecified': False,
              'original_data_models_predictions_unchanged': True,
              'interpretation': dict(definitions)['interpretation']}
    write_json(args.output / 'report.json', report)
    readme = ('# HeLa neighboring Inosine score association\n\n'
              'Exploratory post-hoc analysis of previously saved site summaries. '
              'No BAM extraction, model inference, retraining or original-output modification.\n\n'
              'Each site contributes one mean over sampled reads and neighboring reference-A positions, '
              'excluding the center. Mean ML/255 is not AF or biochemical occupancy. '
              'See Definitions.tsv for exact group, missing-data, sampling and test definitions.\n\n'
              'The workbook stores all summary/test/block values as native numeric cells; '
              'inosine_candidate_endpoints.tsv.gz stores every original-target candidate endpoint. '
              'The all-site Mann–Whitney p is nominal because nearby sites can share reads. '
              'Cluster-robust1kb and10kb checks retain all valid sites and allow within-bin correlation, '
              'without guaranteeing biological independence across bins. '
              'Blocks are chromosome/strand/floor(BED0 position/width). '
              'OLS has only intercept and group; CR1 uses '
              'G/(G−1)×(N−1)/(N−2) and asymptotic normal two-sided Wald p/95% CI. '
              'There is no depth/repeat covariate adjustment and no bootstrap. '
              'Groups are defined by an NN using these same scores: this is selection-associated, '
              'not an independent biochemical validation or feature-causality experiment. '
              'The secondary Slic-seq-supported/unsupported comparison uses external grouping; '
              'catalogue absence is never a false-positive label. Alu and depth strata are descriptive.\n')
    readme += ('\nAll p-values are exploratory and unadjusted for multiple comparisons. '
               'The hypothesis and block widths were not prespecified before observing the associations.\n')
    (args.output / 'README.md').write_text(readme)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
