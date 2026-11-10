"""Predict existing candidates, keep model and reference-rescue decisions separate."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

from utils.nn.inputs import CHANNEL_NAMES, MODIFICATIONS
from utils.references.catalog import ReferenceCatalog, sha256
from utils.features.index import INDEX_FIELDS, completed_run
from prediction.model_bundle import load_model_bundle, validate_input_schema

DECISION_FIELDS = ['reference_cell', 'condition', 'prediction_status', 'keep_score', 'nn_keep',
                   'short_read_sources', 'database_sources', 'reference_overlap', 'rescued',
                   'drach_keep', 'final_keep', 'final_reason', 'score_m6A', 'score_inosine',
                   'score_false_positive', 'nn_class', 'final_class',
                   'flag_db', 'flag_short_read', 'flag_nn', 'flag_drach', 'filter_pass']


def site_decision(score, threshold, has_reference, condition='WT'):
    """Reference rescue is a policy decision, never an independent truth label."""
    if condition not in ('WT', 'IVT'):
        raise ValueError('condition must be WT or IVT')
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be between zero and one')
    if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError('Model score must be finite and between zero and one')
    nn_keep = None if score is None else bool(score >= threshold)
    rescued = condition == 'WT' and has_reference and nn_keep is not True
    final_keep = nn_keep is True or rescued
    reason = ('nn_and_reference' if nn_keep and has_reference and condition == 'WT' else
              'nn_pass' if nn_keep else 'reference_rescue' if rescued else
              'no_prediction' if score is None else 'nn_below_threshold')
    return dict(nn_keep='' if nn_keep is None else int(nn_keep), rescued=int(rescued),
                final_keep=int(final_keep), final_reason=reason)


def policy_decision(mode, has_reference, condition):
    if mode == 'reference_only':
        keep = has_reference and condition == 'WT'
        reason = 'known_reference' if keep else 'reference_not_supported'
    else:
        raise ValueError(f'Unknown prediction filter mode: {mode}')
    return dict(nn_keep='', rescued=0, final_keep=int(keep), final_reason=reason)


def _load_model(directory):
    """Load a checked final model with sequence, read, read-mask and repeat-mask inputs."""
    import tensorflow as tf
    from utils.nn import keras_site  # Register custom layers before load_model.
    directory = Path(directory).resolve()
    _, run = load_model_bundle(directory)
    schema = run['input_schema']
    model_path = directory / run['model_file']
    model = tf.keras.models.load_model(model_path, compile=False)
    inputs = {tensor.name.split(':')[0]: tuple(tensor.shape) for tensor in model.inputs}
    expected_shapes = dict(sequence_tokens=(None, schema['window'] - 2),
                           read_features=(None, schema.get('max_reads'), schema['window'], len(CHANNEL_NAMES)),
                           read_mask=(None, schema.get('max_reads')),
                           sequence_repeat_mask=(None, schema['window']))
    output_size = 3 if run['target'] == 'adenine' else 1
    if inputs != expected_shapes or tuple(model.output_shape) != (None, output_size):
        raise ValueError('Saved Keras input/output shapes do not match the final model schema')
    encoder = model.get_layer('shared_read_encoder')
    if encoder.window != schema['window'] or encoder.channels != len(CHANNEL_NAMES):
        raise ValueError('Saved read encoder disagrees with model schema')
    return model, {}, run


def write_predictions(index_path, scores, out, catalog, reference_cell, condition, threshold,
                      model_provenance=None, site_fields=None, source_sites=None, class_scores=None):
    """Write all candidates, including zero-read sites, with auditable decisions.

    scores maps row_id to a model score. Coordinates and source evidence are
    joined only after prediction and never enter the trained neural network.
    """
    if not reference_cell or any(c.isspace() for c in reference_cell):
        raise ValueError('Set an explicit reference_cell')
    site_decision(None, threshold, False, condition)  # Validate policies before output.
    index_path, out = Path(index_path), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE.txt').write_text('Prediction not complete.', encoding='utf-8')
    counts, details, source_counts, motif_counts = Counter(), Counter(), Counter(), Counter()
    source_by_id = {s['id']: s for s in catalog.sources}
    seen = set()
    try:
        with index_path.open(encoding='utf-8', newline='') as handle, \
                (out / 'all_predictions.tsv').open('w', encoding='utf-8', newline='') as result, \
                (out / 'kept_sites.tsv').open('w', encoding='utf-8', newline='') as kept, \
                (out / 'kept_sites.bed').open('w', encoding='utf-8') as bed:
            fields = list(site_fields or INDEX_FIELDS)
            writer = csv.DictWriter(result, fieldnames=fields + DECISION_FIELDS, delimiter='\t', extrasaction='ignore')
            writer.writeheader()
            kept_writer = csv.DictWriter(kept, fieldnames=fields + DECISION_FIELDS, delimiter='\t', extrasaction='ignore')
            kept_writer.writeheader()
            for row in csv.DictReader(handle, delimiter='\t'):
                if row['row_id'] in seen:
                    raise ValueError('Duplicate candidate row_id')
                seen.add(row['row_id'])
                score = scores.get(row['row_id'])
                joint_score = (class_scores or {}).get(row['row_id'])
                matches = catalog.matches(reference_cell, row['modification'], row['chrom'], int(row['start']), row['strand'])
                short = [s for s in matches if source_by_id[s]['kind'] == 'short_read']
                database = [s for s in matches if source_by_id[s]['kind'] == 'database']
                mode = row.get('filter_mode') or 'nn'
                drach_keep = mode == 'nn_drach_keep'
                if mode in ('nn', 'nn_drach_keep'):
                    decision = site_decision(score, threshold, bool(matches), condition)
                    if joint_score is not None:
                        raw_keep = joint_score['predicted_class'] == row['modification']
                        decision['nn_keep'] = int(raw_keep) if score is not None else ''
                        drach_keep = bool(joint_score.get('drach_keep')) and row['modification'] == 'm6A'
                    status = 'predicted' if score is not None else 'no_reads' if int(row['retained_depth']) == 0 else 'no_prediction'
                    if drach_keep and joint_score is None:
                        if row['modification'] != 'm6A' or row.get('motif_group') != 'DRACH':
                            raise ValueError('DRACH retention requires an annotated m6A DRACH candidate')
                        decision.update(final_keep=1, final_reason='nn_and_drach' if decision['nn_keep'] == 1 else 'drach_keep',
                                        rescued=0)
                else:
                    if score is not None:
                        raise ValueError('Policy-only sites must not have a neural score')
                    decision = policy_decision(mode, bool(matches), condition)
                    status = mode
                flags = dict(flag_db=int(condition == 'WT' and bool(database)),
                             flag_short_read=int(condition == 'WT' and bool(short)),
                             flag_nn=int(decision['nn_keep'] == 1), flag_drach=int(drach_keep))
                final_keep = any(flags.values())
                # Separate evidence columns are independent; any one retains the original candidate.
                decision.update(final_keep=int(final_keep),
                                rescued=int(bool(flags['flag_db'] or flags['flag_short_read'])
                                            and not flags['flag_nn']) if mode != 'reference_only' else 0)
                if joint_score is not None:
                    reasons = [name.removeprefix('flag_') for name, value in flags.items() if value]
                    decision['final_reason'] = '+'.join(reasons) if reasons else 'filter_fail'
                record = dict(row, reference_cell=reference_cell, condition=condition, prediction_status=status,
                              keep_score='' if score is None else score, drach_keep=int(drach_keep), short_read_sources=';'.join(short),
                              database_sources=';'.join(database), reference_overlap=int(bool(matches)), **decision, **flags, filter_pass=int(final_keep))
                record.update(score_m6A=joint_score['score_m6A'] if joint_score else '',
                              score_inosine=joint_score['score_inosine'] if joint_score else '',
                              score_false_positive=joint_score['score_false_positive'] if joint_score else '',
                              nn_class=joint_score['predicted_class'] if joint_score else '',
                              final_class=row['modification'] if joint_score and final_keep else '')
                writer.writerow(record)
                if decision['final_keep']:
                    kept_writer.writerow(record)
                    bed.write(f"{row['chrom']}\t{row['start']}\t{row['end']}\t{row['row_id']}\t0\t{row['strand']}\n")
                key = row['sample_id'], row['modification']
                for metric, value in dict(input_sites=1, predicted_sites=score is not None,
                                          unpredicted_sites=score is None, nn_pass_sites=decision['nn_keep'] == 1,
                                          reference_overlap_sites=bool(matches), rescued_sites=decision['rescued'],
                                          final_kept_sites=decision['final_keep'], final_not_kept_sites=not decision['final_keep']).items():
                    counts[key + (metric,)] += int(value)
                if row['modification'] == 'm6A':
                    group = row.get('motif_group') or 'unknown'
                    for metric, value in dict(input_sites=1, predicted_sites=score is not None,
                                              nn_pass_sites=decision['nn_keep'] == 1,
                                              reference_overlap_sites=bool(matches), rescued_sites=decision['rescued'],
                                              drach_keep_sites=drach_keep, final_kept_sites=decision['final_keep']).items():
                        motif_counts[key + (group, metric)] += int(value)
                details[key + (status, str(decision['nn_keep']), decision['rescued'],
                               decision['final_keep'], decision['final_reason'])] += 1
                for source in matches:
                    for metric, value in dict(overlap_sites=1, nn_pass_sites=decision['nn_keep'] == 1,
                                              rescued_sites=decision['rescued'], final_kept_sites=decision['final_keep']).items():
                        source_counts[key + (source, metric)] += int(value)
        if set(scores) - seen:
            raise ValueError('Model scores include unknown candidate IDs')
        keys = sorted({(sample, mod) for sample, mod, metric in counts})
        metrics = ['input_sites', 'predicted_sites', 'unpredicted_sites', 'nn_pass_sites',
                   'reference_overlap_sites', 'rescued_sites', 'final_kept_sites', 'final_not_kept_sites']
        with (out / 'summary.tsv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['sample_id', 'modification', *metrics])
            writer.writerows([[*key, *[counts[key + (metric,)] for metric in metrics]] for key in keys])
        with (out / 'decision_counts.tsv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['sample_id', 'modification', 'prediction_status', 'nn_keep', 'rescued', 'final_keep', 'final_reason', 'sites'])
            writer.writerows([(*key, value) for key, value in sorted(details.items())])
        with (out / 'motif_summary.tsv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            metrics = ['input_sites', 'predicted_sites', 'nn_pass_sites', 'reference_overlap_sites',
                       'rescued_sites', 'drach_keep_sites', 'final_kept_sites']
            writer.writerow(['sample_id', 'modification', 'motif_group', *metrics])
            for key in sorted({key[:3] for key in motif_counts}):
                writer.writerow([*key, *[motif_counts[key + (metric,)] for metric in metrics]])
        with (out / 'source_overlap_counts.tsv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            source_metrics = ['overlap_sites', 'nn_pass_sites', 'rescued_sites', 'final_kept_sites']
            writer.writerow(['sample_id', 'modification', 'source_id', 'source_kind', 'source_cell', 'reference_status', *source_metrics])
            for key in keys:
                for source in catalog.applicable(reference_cell, key[1]):
                    writer.writerow([*key, source['id'], source['kind'], source['cell'], source['status'],
                                     *[source_counts[key + (source['id'], metric)] if source['status'] == 'ready' else ''
                                       for metric in source_metrics]])
        source_path = Path(source_sites) if source_sites is not None else index_path
        report = dict(status='complete', index_path=str(source_path.resolve()), index_sha256=sha256(source_path),
                      model=model_provenance, reference_cell=reference_cell, condition=condition, threshold=threshold,
                      reference_catalog=str(catalog.path) if catalog.path is not None else None,
                      reference_sources=catalog.sources,
                      policies=dict(reference_only='m5C/psi: keep known WT candidates only; blank NN score',
                                    drach='when enabled, score all m6A candidates and retain DRACH independently of NN score',
                                    rescue='WT candidates only; exact chromosome/BED0 start/strand/target modification',
                                    output='original candidates only; PASS is OR of independent DB, short-read, NN and DRACH flags',
                                    ivt='no reference rescue; neural score retained for background-call auditing',
                                    missing_prediction='keep blank score/status; reference rescue may retain WT candidates',
                                    interpretation='neural filtering and reference-based rescue decisions, not independent experimental truth',
                                    reference_evaluation='rescue references must not be reused as independent accuracy evidence'))
        (out / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as exc:
        (out / 'FAILED.txt').write_text(str(exc), encoding='utf-8')
        raise
