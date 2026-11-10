"""Publication audit tables from site TSVs; no BAM or repeated BED scan needed."""
import csv
from collections import Counter, defaultdict
from pathlib import Path


def write_counts(path, columns, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(columns)
        writer.writerows(rows)


def summarize_site_list(site_list, out, pairs, sources):
    """Counts are sites/read-site observations, never independent experimental truth."""
    site_list, out = Path(site_list), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    totals = defaultdict(Counter)
    reasons, overlaps, source_hits, comparisons, histograms, groups = (Counter() for _ in range(6))
    last_group = {}
    with site_list.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            sid, mod, label, status = row['sample_id'], row['modification'], row['label'], row['label_status']
            total = totals[sid, mod]
            total['sites'] += 1
            total['positive' if label == '1' else 'negative' if label == '0' else
                  'pending' if status == 'pending' else 'unlabeled'] += 1
            sr, db = bool(row['short_read_sources']), bool(row['database_sources'])
            evidence = bool(int(row['positive_evidence']))
            total['positive_evidence'] += evidence
            total['overridden_overlap'] += evidence and label == '0'
            group = row['group_id']
            strata = (sid, mod, label, status, row['label_reason'])
            reasons[strata] += 1
            if last_group.get(strata) != group:
                groups[strata] += 1
                last_group[strata] = group
            comparisons[sid, mod, row['ivt_comparison'], label, status] += 1
            if row['condition'] == 'WT':
                partition = 'both' if sr and db else 'short_read_only' if sr else 'database_only' if db else 'neither'
                overlaps[sid, mod, label, status, partition] += 1
                for field in ('short_read_sources', 'database_sources'):
                    for source in filter(None, row[field].split(';')):
                        source_hits[sid, mod, source, label, status] += 1
            depth, af = int(row['total_reads']), float(row['af'])
            depth_bin = next((name for bound, name in [(10, '<10'), (20, '10-19'), (50, '20-49'),
                            (100, '50-99'), (200, '100-199')] if depth < bound), '>=200')
            af_bin = next((name for bound, name in [(.1, '[0,0.1)'), (.2, '[0.1,0.2)'),
                         (.4, '[0.2,0.4)'), (.6, '[0.4,0.6)'), (.8, '[0.6,0.8)')] if af < bound), '[0.8,1]')
            histograms[sid, mod, label, status, 'depth', depth_bin] += 1
            histograms[sid, mod, label, status, 'af', af_bin] += 1
    sample_rows = []
    for pair in pairs:
        for condition in ('wt', 'ivt'):
            sid = pair[f'{condition}_sample']
            available = pair.get(f'{condition}_bedmethyl') is not None
            for mod in ('m6A', 'm5C', 'inosine', 'pseU'):
                c = totals[sid, mod]
                sample_rows.append([sid, pair['cell'], pair['reference_cell'], condition.upper(), mod,
                                    'available' if available else 'not_available',
                                    *[c[k] if available else '' for k in ('sites', 'positive', 'negative',
                                       'pending', 'unlabeled', 'positive_evidence', 'overridden_overlap')]])
    write_counts(out / 'sample_modification_counts.tsv',
                 ['sample_id', 'cell', 'reference_cell', 'condition', 'modification', 'input_status',
                  'total_sites', 'positive', 'negative', 'pending', 'unlabeled',
                  'positive_evidence_sites', 'overridden_overlap_sites'], sample_rows)
    write_counts(out / 'label_reason_counts.tsv',
                 ['sample_id', 'modification', 'label', 'label_status', 'label_reason', 'sites', 'locus_groups'],
                 [(*key, count, groups[key]) for key, count in sorted(reasons.items())])
    write_counts(out / 'reference_overlap_counts.tsv',
                 ['sample_id', 'modification', 'label', 'label_status', 'partition', 'sites'],
                 [(*key, count) for key, count in sorted(overlaps.items())])
    lookup = {s['id']: s for s in sources}
    source_rows = []
    for (sid, mod, source, label, status), count in sorted(source_hits.items()):
        s = lookup[source]
        source_rows.append([sid, mod, source, s['kind'], s['cell'], label, status, count])
    write_counts(out / 'source_overlap_counts.tsv',
                 ['sample_id', 'modification', 'source_id', 'source_kind', 'source_cell', 'label', 'label_status', 'sites'],
                 source_rows)
    write_counts(out / 'ivt_comparison_counts.tsv',
                 ['sample_id', 'modification', 'ivt_comparison', 'label', 'label_status', 'sites'],
                 [(*key, count) for key, count in sorted(comparisons.items())])
    write_counts(out / 'depth_af_histograms.tsv',
                 ['sample_id', 'modification', 'label', 'label_status', 'variable', 'bin', 'sites'],
                 [(*key, count) for key, count in sorted(histograms.items())])
    # Zero matches and unavailable references are different; enumerate every applicable source.
    applicability = []
    source_totals = []
    for pair in pairs:
        for source in sources:
            if source['cell'] in ('*', pair['reference_cell']):
                applicability.append([pair['wt_sample'], source['modification'], source['id'], source['kind'],
                                      source['cell'], source['status'], source.get('unique_sites', '')])
                sid, mod = pair['wt_sample'], source['modification']
                ready = source['status'] == 'ready' and pair.get('wt_bedmethyl') is not None
                hits = Counter()
                for (hit_sid, hit_mod, source_id, label, status), n in source_hits.items():
                    if (hit_sid, hit_mod, source_id) == (sid, mod, source['id']):
                        hits['all'] += n
                        hits['positive' if label == '1' else 'negative' if label == '0' else
                             'pending' if status == 'pending' else 'unlabeled'] += n
                source_totals.append([sid, mod, source['id'], source['kind'], source['cell'], source['status'],
                                      *[hits[k] if ready else '' for k in ('all', 'positive', 'negative', 'pending', 'unlabeled')]])
    write_counts(out / 'reference_sources.tsv',
                 ['sample_id', 'modification', 'source_id', 'source_kind', 'source_cell', 'reference_status', 'reference_sites'],
                 applicability)
    write_counts(out / 'source_overlap_summary.tsv',
                 ['sample_id', 'modification', 'source_id', 'source_kind', 'source_cell', 'reference_status',
                  'overlap_sites', 'positive', 'negative', 'pending', 'unlabeled'], source_totals)
