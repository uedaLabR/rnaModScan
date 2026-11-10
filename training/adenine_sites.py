"""Build one three-class training example per sample/coordinate/strand."""
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from utils.adenine import CLASSES, FIELDS, MODIFICATIONS, combine_site, merged_sites
from utils.io.artifacts import completed_run
from utils.references.catalog import sha256

PAIRED_REASONS = {'ivt_paired_wt_positive', 'ivt_candidate_and_paired'}


def labeled_sites(paths):
    for rows in merged_sites(paths):
        row = combine_site(rows)
        positives = [target for target, source in rows.items() if source['label'] == '1']
        paired = any(source['label_reason'] in PAIRED_REASONS for source in rows.values())
        if len(positives) == 2:
            label, reason = '', 'both_positive_excluded'
        elif positives:
            label, reason = str(CLASSES.index(positives[0])), 'positive_' + positives[0]
        elif row['condition'] == 'IVT':
            label, reason = '2', 'paired_ivt' if paired else 'ivt_negative'
        elif len(rows) == 2 and all(source['label'] == '0' for source in rows.values()):
            label, reason = '2', 'both_targets_negative'
        else:
            label, reason = '', 'other_target_unknown'
        row.update(label=label, label_reason=reason, label_status='final' if label else 'excluded')
        yield row


def select_adenine_sites(source, out, *, seed=42):
    """Keep positives/paired IVT, then sample one extra FP per positive per cell."""
    source, out = Path(source), Path(out)
    upstream = completed_run(source)
    if any(entry['status'] != 'available' for entry in upstream.get('inputs', [])):
        raise ValueError('Wait for all paired WT/IVT bedMethyl inputs before joint training selection')
    # Include unlabeled candidate rows when joining metadata. Otherwise the
    # presence of the other AF would itself reveal whether a label was assigned.
    paths = [source / target / 'all_sites.tsv' for target in MODIFICATIONS]
    digests = [sha256(path) for path in paths]
    if out.exists():
        run = completed_run(out)
        if run.get('source_sha256') != digests or run['seed'] != seed:
            raise ValueError('Joint training source/seed changed; use a new output directory')
        if sha256(out / 'adenine/training_sites.tsv') != run['selected_sha256']:
            raise ValueError('Selected joint training list changed')
        return run
    out.mkdir(parents=True)
    (out / 'INCOMPLETE.txt').write_text('Joint training selection not complete.\n')
    try:
        positives, paired, available, excluded = Counter(), Counter(), Counter(), Counter()
        classes = Counter()
        print('Joint sites: counting positive, control and eligible FP sites', flush=True)
        for row in labeled_sites(paths):
            cell = row['cell']
            classes[cell, row['label_reason']] += 1
            if row['label'] in ('0', '1'):
                positives[cell] += 1
            elif row['label_reason'] == 'paired_ivt':
                paired[cell] += 1
            elif row['label'] == '2':
                available[cell] += 1
            else:
                excluded[row['label_reason']] += 1
        cells = sorted(set(positives) | set(available) | set(paired))
        reservoirs = {cell: [] for cell in cells}
        rng = {cell: random.Random(int.from_bytes(hashlib.sha256(f'{seed}:adenine:{cell}'.encode()).digest(), 'big'))
               for cell in cells}
        seen = Counter()
        print('Joint sites: sampling additional FP sites', flush=True)
        for row in labeled_sites(paths):
            if row['label'] != '2' or row['label_reason'] == 'paired_ivt':
                continue
            cell = row['cell']
            seen[cell] += 1
            bucket, quota = reservoirs[cell], positives[cell]
            if len(bucket) < quota:
                bucket.append(row['row_id'])
            elif quota:
                index = rng[cell].randrange(seen[cell])
                if index < quota:
                    bucket[index] = row['row_id']
        selected = {name for bucket in reservoirs.values() for name in bucket}
        folder = out / 'adenine'
        folder.mkdir()
        with paths[0].open(encoding='utf-8') as handle:
            fields = list(dict.fromkeys(next(csv.reader(handle, delimiter='\t')) + FIELDS + ['selection_reason']))
        selected_counts = Counter()
        print('Joint sites: writing selected sites and exclusion audit', flush=True)
        with (folder / 'training_sites.tsv').open('w', encoding='utf-8', newline='') as handle, \
                (out / 'excluded_sites.tsv').open('w', encoding='utf-8', newline='') as audit:
            writer = csv.DictWriter(handle, fields, delimiter='\t', extrasaction='ignore')
            writer.writeheader()
            audit_fields = ['row_id', 'sample_id', 'chrom', 'start', 'strand', 'label_reason',
                            'source_row_id_m6A', 'source_row_id_inosine']
            audit_writer = csv.DictWriter(audit, audit_fields, delimiter='\t', extrasaction='ignore')
            audit_writer.writeheader()
            for row in labeled_sites(paths):
                if row['label'] == '':
                    audit_writer.writerow(row)
                elif row['label'] in ('0', '1') or row['label_reason'] == 'paired_ivt' or row['row_id'] in selected:
                    row['selection_reason'] = ('positive' if row['label'] in ('0', '1') else
                                               'paired_ivt' if row['label_reason'] == 'paired_ivt' else 'random_negative')
                    writer.writerow(row)
                    selected_counts[row['cell'], CLASSES[int(row['label'])], row['selection_reason']] += 1
        with (out / 'summary.tsv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['cell', 'class', 'selection_reason', 'sites'])
            writer.writerows([*key, count] for key, count in sorted(selected_counts.items()))
        if [sha256(path) for path in paths] != digests:
            raise ValueError('Source lists changed during joint selection')
        excluded.setdefault('both_positive_excluded', 0)
        run = dict(status='complete', schema='adenine_training_sites_v1', source=str(source.resolve()),
                   source_sha256=digests, selected_sha256=sha256(folder / 'training_sites.tsv'),
                   seed=seed, class_names=list(CLASSES), window_flank=upstream['window_flank'],
                   selected_sites=sum(selected_counts.values()), excluded=dict(excluded),
                   counts=[dict(cell=cell, class_name=label, selection_reason=reason, sites=count)
                           for (cell, label, reason), count in sorted(selected_counts.items())],
                   policies=dict(ambiguity='both positive excluded and audited',
                                 fp='IVT or WT with explicit negative labels for BOTH targets; absent label is unknown',
                                 positives='original weak positive labels, not independent ground truth',
                                 sampling='all positives and paired IVT; other FP sampled 1:1 per cell',
                                 split='group sample/coordinate/strand before train/validation/test'))
        (out / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
        (out / 'INCOMPLETE.txt').unlink()
        return run
    except BaseException as error:
        (out / 'FAILED.txt').write_text(str(error) + '\n')
        raise
