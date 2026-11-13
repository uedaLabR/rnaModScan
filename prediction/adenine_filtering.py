"""Apply independent evidence flags to original A-site candidates."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

from prediction.filtering import (combine_partition_predictions, partition_file, write_partition_sites)
from prediction.predict import DECISION_FIELDS, write_predictions
from utils.adenine import CLASSES, MODIFICATIONS
from utils.features.parquet import require_arrow
from utils.io.artifacts import completed_run
from utils.references.catalog import sha256
from prediction.confidence import BASE_FIELDS
from prediction.motifs import mettl16_site

CLASSIFICATION_FIELDS = ['row_id', 'sample_id', 'chrom', 'start', 'strand', 'score_m6A', 'score_inosine',
                         'score_false_positive', 'predicted_class', 'known_m6A', 'known_inosine',
                         'drach_keep', 'mettl16_keep', 'final_class', 'decision_reason']


def retained_adenine_classes(scores, known, drach_keep, candidates, mettl16_keep=False):
    """OR policy for original candidates; the NN still has one softmax class."""
    retained = []
    for target in MODIFICATIONS:
        nn_pass = scores is not None and scores['predicted_class'] == target
        if target in candidates and (target in known or nn_pass or target == 'm6A' and (drach_keep or mettl16_keep)):
            retained.append(target)
    return retained


def read_joint_scores(path, expected):
    _, pq = require_arrow()
    result = {}
    for batch in pq.ParquetFile(path).iter_batches():
        for row in batch.to_pylist():
            if row['row_id'] in result:
                raise ValueError('Duplicate joint score ID')
            values = [row['score_' + name] for name in CLASSES]
            if (any(not math.isfinite(value) or not 0 <= value <= 1 for value in values)
                    or not math.isclose(sum(values), 1, abs_tol=1e-5)
                    or row['predicted_class'] != CLASSES[max(range(3), key=lambda index: values[index])]):
                raise ValueError('Invalid joint class scores or predicted class')
            result[row['row_id']] = row
    if len(result) != expected:
        raise ValueError('Joint score count differs from prepared inputs')
    return result


def project_site(site, target):
    row = dict(site)
    row.update(modification=target, row_id=f"{site['sample_id']}|{site['chrom']}:{site['start']}:{site['strand']}:{target}",
               filter_mode='nn', label='', label_reason='joint_prediction')
    for name in ('af', 'total_reads', 'modified_reads', *BASE_FIELDS):
        row[name] = site.get(name + '_' + target, '')
    return row


def filter_joint_inputs(input_directory, score_directory, prediction_root, catalog, reference_cell,
                        *, condition='WT', m6a_drach_keep=True, m6a_mettl16_keep=False):
    _, pq = require_arrow()
    input_directory, score_directory, prediction_root = map(Path, (input_directory, score_directory, prediction_root))
    from utils.references.catalog import ReferenceCatalog
    if not isinstance(catalog, ReferenceCatalog):
        catalog = ReferenceCatalog(catalog)
    inputs, scored = completed_run(input_directory), completed_run(score_directory)
    if (inputs.get('schema') != 'prediction_inputs_v9' or scored.get('schema') != 'prediction_scores_v1' or inputs['target'] != 'adenine'
            or scored['target'] != 'adenine' or scored['model'] != inputs['model']
            or scored['threshold'] != inputs['threshold']
            or scored['input_run_sha256'] != sha256(input_directory / 'run.json')):
        raise ValueError('Joint scores and prepared input/model bindings disagree')
    score_parts = {(part['chrom'], part['strand']): part for part in scored['partitions']}
    input_keys = {(part['chrom'], part['strand']) for part in inputs['partitions']}
    if len(score_parts) != len(scored['partitions']) or len(input_keys) != len(inputs['partitions']) or set(score_parts) != input_keys:
        raise ValueError('Joint score partitions disagree with prepared inputs')
    roots = {target: prediction_root / target for target in MODIFICATIONS}
    joint_root = prediction_root / 'adenine'
    for directory in [joint_root, *roots.values()]:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'INCOMPLETE.txt').write_text('Joint filtering not complete.\n')
    directories = {target: [] for target in MODIFICATIONS}
    partitions = {target: [] for target in MODIFICATIONS}
    classification_files = []
    class_counts = Counter()
    fields = None
    try:
        for part in inputs['partitions']:
            key = part['chrom'], part['strand']
            score_part = score_parts[key]
            if score_part['rows'] != part['input_count']:
                raise ValueError('Joint partition row counts disagree')
            score_path = partition_file(score_directory, score_part['path'], score_part['size_bytes'])
            scores = read_joint_scores(score_path, part['input_count'])
            sites_path = partition_file(input_directory, part['sites_path'], part['sites_size_bytes'])
            relative = Path(part['sites_path']).parent
            work = joint_root / relative
            work.mkdir(parents=True)
            temporary = work / 'sites.tsv.partial'
            fields, expected = write_partition_sites(sites_path, temporary, *key)
            if set(scores) != expected:
                raise ValueError('Joint scored IDs disagree with prepared readable sites')
            contexts = {}
            classification = work / 'site_classifications.tsv'
            with temporary.open(encoding='utf-8', newline='') as handle, classification.open('w', encoding='utf-8', newline='') as dest:
                reader = csv.DictReader(handle, delimiter='\t')
                writer = csv.DictWriter(dest, CLASSIFICATION_FIELDS, delimiter='\t', extrasaction='ignore')
                writer.writeheader()
                for site in reader:
                    known = [target for target in MODIFICATIONS if condition == 'WT' and catalog.matches(
                        reference_cell, target, site['chrom'], int(site['start']), site['strand'])]
                    raw = scores.get(site['row_id'])
                    keep_drach = (m6a_drach_keep and site.get('motif_group') == 'DRACH'
                                  and str(site.get('candidate_m6A')) == '1')
                    keep_mettl16 = (m6a_mettl16_keep and str(site.get('candidate_m6A')) == '1'
                                    and mettl16_site(site))
                    candidates = [target for target in MODIFICATIONS if str(site.get('candidate_' + target)) == '1']
                    selected = ';'.join(retained_adenine_classes(raw, known, keep_drach, candidates, keep_mettl16))
                    reason = 'independent_evidence_or' if selected else 'filter_fail'
                    context = dict(raw) if raw is not None else dict(predicted_class='', **{'score_' + name: None for name in CLASSES})
                    context.update(final_class=selected, decision_reason=reason, drach_keep=keep_drach,
                                   mettl16_keep=keep_mettl16)
                    contexts[site['row_id']] = context
                    class_counts[site['sample_id'], context['predicted_class'], selected, reason] += 1
                    record = dict(site)
                    record.update({name: context[name] for name in context if name != 'row_id'},
                                  known_m6A=int('m6A' in known), known_inosine=int('inosine' in known),
                                  drach_keep=int(keep_drach), mettl16_keep=int(keep_mettl16))
                    writer.writerow(record)
            classification_files.append(classification)
            for target in MODIFICATIONS:
                projected = work / (target + '.tsv.partial')
                probabilities, target_contexts = {}, {}
                with temporary.open(encoding='utf-8', newline='') as handle, projected.open('w', encoding='utf-8', newline='') as dest:
                    writer = csv.DictWriter(dest, fields, delimiter='\t', extrasaction='ignore')
                    writer.writeheader()
                    for site in csv.DictReader(handle, delimiter='\t'):
                        if str(site.get('candidate_' + target)) != '1':
                            continue  # Never manufacture another modification's candidate.
                        row = project_site(site, target)
                        writer.writerow(row)
                        context = contexts[site['row_id']]
                        target_contexts[row['row_id']] = context
                        if context['score_' + target] is not None:
                            probabilities[row['row_id']] = context['score_' + target]
                destination = roots[target] / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                write_predictions(projected, probabilities, destination, catalog, reference_cell, condition,
                                  inputs['threshold'], site_fields=fields, source_sites=sites_path,
                                  class_scores=target_contexts,
                                  model_provenance=dict(model=inputs['model'], scores=str(score_directory.resolve())))
                projected.unlink()
                directories[target].append(destination)
                partitions[target].append(dict(chrom=key[0], strand=key[1], path=str(relative / 'all_predictions.tsv')))
            temporary.unlink()
        with (joint_root / 'all_classifications.tsv').open('w', encoding='utf-8', newline='') as dest:
            csv.writer(dest, delimiter='\t').writerow(CLASSIFICATION_FIELDS)
            for path in classification_files:
                with path.open(encoding='utf-8') as source:
                    source.readline()
                    for line in source:
                        dest.write(line)
        from prediction.input_parquet import SITE_FIELDS
        output_fields = fields or SITE_FIELDS + ['regions']
        for target in MODIFICATIONS:
            combine_partition_predictions(directories[target], roots[target], list(dict.fromkeys(output_fields + DECISION_FIELDS)))
            run = dict(status='complete', schema='joint_target_predictions_v1', target=target,
                       class_names=list(CLASSES), partitions=partitions[target],
                       input_run_sha256=sha256(input_directory / 'run.json'),
                       score_run_sha256=sha256(score_directory / 'run.json'),
                       filtering=dict(m6a_drach_keep=m6a_drach_keep, m6a_mettl16_keep=m6a_mettl16_keep),
                       interpretation='weakly supervised NN with independent evidence OR and original-candidate DRACH/METTL16 motif policies')
            (roots[target] / 'run.json').write_text(json.dumps(run, indent=2))
            (roots[target] / 'INCOMPLETE.txt').unlink()
        with (joint_root / 'class_counts.tsv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['sample_id', 'nn_class', 'final_class', 'decision_reason', 'sites'])
            writer.writerows([*key, count] for key, count in sorted(class_counts.items()))
        report = dict(status='complete', schema='adenine_classifications_v1', class_names=list(CLASSES),
                      sites=sum(class_counts.values()),
                      input_run_sha256=sha256(input_directory / 'run.json'),
                      score_run_sha256=sha256(score_directory / 'run.json'),
                      filtering=dict(m6a_drach_keep=m6a_drach_keep, m6a_mettl16_keep=m6a_mettl16_keep),
                      policies=dict(known='independent DB and short-read flags; WT only',
                                    drach='preserve original DRACH m6A independently of NN',
                                    mettl16='when enabled, preserve original canonical UAC[m6A]GAGAA m6A candidates independently of NN',
                                    output='original candidate rows only; OR of DB, short-read, NN, DRACH, METTL16-motif'))
        (joint_root / 'run.json').write_text(json.dumps(report, indent=2))
        (joint_root / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as error:
        for directory in [joint_root, *roots.values()]:
            (directory / 'FAILED.txt').write_text(str(error) + '\n')
        raise
