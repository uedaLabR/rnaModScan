"""Predictions and metrics for validation and held-out test splits."""
import csv
from pathlib import Path

import numpy as np
from utils.references.genome import drach
from utils.adenine import CLASSES

METRIC_INTERPRETATION = 'metrics against operational weak labels; internal evaluation, not independent validation'


def motif_group(row):
    if row.get('modification') not in ('m6A', 'adenine'):
        return ''
    sequence = row.get('reference_sequence', '').upper()
    center = len(sequence) // 2
    motif = sequence[center - 2:center + 3] if len(sequence) >= 5 else ''
    if len(motif) != 5 or any(base not in 'ACGT' for base in motif):
        return 'unknown'
    return 'DRACH' if drach(motif) else 'non_DRACH'


def label_metrics(labels, scores, threshold):
    """Compare site scores with operational binary labels at a fixed threshold."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    predictions = scores >= threshold
    tp = int(np.sum((labels == 1) & predictions))
    fp = int(np.sum((labels == 0) & predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    tn = int(np.sum((labels == 0) & ~predictions))

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return dict(sites=len(labels), tp=tp, fp=fp, fn=fn, tn=tn, threshold=threshold,
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                fpr=ratio(fp, fp + tn), accuracy=ratio(tp + tn, len(labels)),
                f1=ratio(2 * tp, 2 * tp + fp + fn))


def predict_split(model, dataset, out, threshold):
    """Save each site's score and return this split's label-based metrics."""
    if dataset.target == 'adenine':
        return predict_three_class_split(model, dataset, out)
    scores = []
    for inputs, _ in dataset.batches():
        scores.extend(model(inputs, training=False).numpy().ravel().tolist())
    with Path(out).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['row_id', 'sample_id', 'label', 'keep_score', 'predicted_label', 'motif_group'])
        for row, score in zip(dataset.rows, scores):
            writer.writerow([row['row_id'], row['sample_id'], row['label'], score, int(score >= threshold), motif_group(row)])
    metrics = label_metrics([int(row['label']) for row in dataset.rows], scores, threshold)
    if any(row.get('modification') == 'm6A' for row in dataset.rows):
        metrics['by_motif'] = {}
        for group in ('DRACH', 'non_DRACH', 'unknown'):
            selected = [(row, score) for row, score in zip(dataset.rows, scores) if motif_group(row) == group]
            metrics['by_motif'][group] = label_metrics([int(row['label']) for row, _ in selected],
                                                      [score for _, score in selected], threshold)
    return metrics


def class_metrics(matrix, class_names):
    """Count one class against all other classes; undefined ratios stay null."""
    sites = int(matrix.sum())
    per_class = {}
    for index, name in enumerate(class_names):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum()) - tp
        fn = int(matrix[index, :].sum()) - tp
        tn = sites - tp - fp - fn
        per_class[name] = dict(support=int(matrix[index, :].sum()), tp=tp, fp=fp, fn=fn, tn=tn,
            one_vs_rest_accuracy=(tp + tn) / sites if sites else None,
            precision=tp / (tp + fp) if tp + fp else None,
            recall=tp / (tp + fn) if tp + fn else None,
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
    return per_class


def three_class_metrics(labels, scores):
    """Keep overall argmax accuracy separate from per-class one-vs-rest accuracy.

    An absent class may have perfect one-vs-rest accuracy from true negatives;
    support and undefined precision/recall therefore remain explicit. Preserve
    the existing macro-F1 convention of averaging only defined class F1 values.
    """
    if len(labels) != len(scores):
        raise ValueError('Labels and scores must contain the same number of sites')
    matrix = np.zeros((3, 3), dtype=int)
    for label, score in zip(labels, scores):
        values = np.asarray(score)
        if int(label) not in range(3) or values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError('Use three finite scores and a valid class label per site')
        matrix[int(label), int(np.argmax(values))] += 1
    per_class = class_metrics(matrix, CLASSES)
    f1s = [value['f1'] for value in per_class.values() if value['f1'] is not None]
    return dict(sites=len(labels), class_names=list(CLASSES), confusion_matrix=matrix.tolist(),
                accuracy=float(np.trace(matrix) / len(labels)) if len(labels) else None,
                macro_f1=float(np.mean(f1s)) if f1s else None, per_class=per_class,
                accuracy_definition='correct three-class argmax predictions / sites',
                one_vs_rest_accuracy_definition='(TP + TN) / sites for one class versus all others',
                macro_f1_policy='mean of class F1 values that are defined',
                interpretation=METRIC_INTERPRETATION)


def write_metrics_summary(metrics, out):
    """Write metrics_summary.tsv and a labelled confusion_matrix.tsv in out.

    Overall accuracy is the multiclass correct fraction. Class rows instead
    populate one_vs_rest_accuracy, precision, recall and F1. The binary fallback
    keeps the same writer usable by older single-modification callers.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    class_names = metrics.get('class_names', ['unmodified', 'modified'])
    matrix = metrics.get('confusion_matrix')
    if matrix is None:
        matrix = [[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]]
    matrix = np.asarray(matrix, dtype=int)
    if matrix.shape != (len(class_names), len(class_names)) or int(matrix.sum()) != metrics['sites']:
        raise ValueError('Confusion matrix does not match the reported sites/classes')
    per_class = metrics.get('per_class') or class_metrics(matrix, class_names)
    interpretation = metrics.get('interpretation', METRIC_INTERPRETATION)
    summary = out / 'metrics_summary.tsv'
    fields = ['scope', 'class', 'sites', 'support', 'predicted_sites', 'tp', 'fp', 'fn', 'tn',
              'accuracy', 'one_vs_rest_accuracy', 'precision', 'recall', 'f1', 'macro_f1', 'interpretation']
    with summary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fields, delimiter='\t')
        writer.writeheader()
        writer.writerow(dict(scope='overall', **{'class': 'all'}, sites=metrics['sites'],
                             support=metrics['sites'], accuracy=metrics.get('accuracy'),
                             macro_f1=metrics.get('macro_f1'), interpretation=interpretation))
        for name in class_names:
            values = per_class[name]
            writer.writerow(dict(scope='one_vs_rest', **{'class': name}, sites=metrics['sites'],
                                 predicted_sites=values['tp'] + values['fp'],
                                 interpretation=interpretation, **values))
    confusion = out / 'confusion_matrix.tsv'
    with confusion.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['true_class / predicted_class', *class_names])
        for name, values in zip(class_names, matrix.tolist()):
            writer.writerow([name, *values])
    return dict(metrics_summary=str(summary), confusion_matrix=str(confusion))


def predict_three_class_split(model, dataset, out):
    scores = []
    for inputs, _ in dataset.batches():
        scores.extend(model(inputs, training=False).numpy().tolist())
    with Path(out).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['row_id', 'sample_id', 'label', 'score_m6A', 'score_inosine',
                         'score_false_positive', 'predicted_class', 'motif_group'])
        for row, score in zip(dataset.rows, scores):
            writer.writerow([row['row_id'], row['sample_id'], CLASSES[int(row['label'])], *score,
                             CLASSES[int(np.argmax(score))], motif_group(row)])
    labels = [int(row['label']) for row in dataset.rows]
    report = three_class_metrics(labels, scores)
    report['by_motif'] = {}
    for group in ('DRACH', 'non_DRACH', 'unknown'):
        selected = [(row, score) for row, score in zip(dataset.rows, scores) if motif_group(row) == group]
        report['by_motif'][group] = three_class_metrics([int(row['label']) for row, _ in selected],
                                                       [score for _, score in selected])
    report['by_depth'] = {}
    for name, lower, upper in (('1-9', 1, 10), ('10-29', 10, 30), ('30-99', 30, 100),
                               ('100-299', 100, 300), ('300+', 300, float('inf'))):
        selected = [(row, score) for row, score in zip(dataset.rows, scores)
                    if lower <= int(row['eligible_depth']) < upper]
        report['by_depth'][name] = three_class_metrics([int(row['label']) for row, _ in selected],
                                                       [score for _, score in selected])
    return report
