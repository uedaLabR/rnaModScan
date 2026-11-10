"""Paired metrics for a frozen model against the existing operational labels."""
import numpy as np

from utils.adenine import CLASSES
from training.validation import motif_group


def classification_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(labels), 3) or not np.isfinite(scores).all():
        raise ValueError('Expected three finite probabilities for every site')
    if len(labels) and (np.any(labels < 0) or np.any(labels > 2)):
        raise ValueError('Unknown operational class')
    if scores.size and (np.any(scores < 0) or np.any(scores > 1)
                        or not np.allclose(scores.sum(axis=1), 1, atol=1e-5)):
        raise ValueError('Class probabilities must be normalized')
    matrix = np.zeros((3, 3), dtype=np.int64)
    if len(labels):
        np.add.at(matrix, (labels, scores.argmax(axis=1)), 1)
    n = len(labels)
    per_class = {}
    for i, name in enumerate(CLASSES):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum()) - tp
        fn = int(matrix[i, :].sum()) - tp
        tn = n - tp - fp - fn
        per_class[name] = dict(support=int(matrix[i].sum()), tp=tp, fp=fp, fn=fn, tn=tn,
            precision=tp / (tp + fp) if tp + fp else None,
            recall=tp / (tp + fn) if tp + fn else None,
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            one_vs_rest_accuracy=(tp + tn) / n if n else None)
    f1s = [row['f1'] for row in per_class.values() if row['f1'] is not None]
    loss = float(-np.log(np.clip(scores[np.arange(n), labels], 1e-7, 1)).mean()) if n else None
    return dict(sites=n, accuracy=float(np.trace(matrix) / n) if n else None,
        macro_f1=float(np.mean(f1s)) if f1s else None, log_loss=loss,
        confusion_matrix=matrix.tolist(), per_class=per_class)


def evaluate_variants(rows, variants, scores):
    """Keep the baseline and every intervention on the identical site cohort."""
    labels = np.asarray([int(row['label']) for row in rows], dtype=np.int32)
    motifs = np.asarray([motif_group(row) for row in rows])
    baseline = np.asarray(scores[0])
    baseline_classes = baseline.argmax(axis=1)
    evaluations = []
    for i, variant in enumerate(variants):
        current = np.asarray(scores[i])
        groups = {name: classification_metrics(labels[motifs == name], current[motifs == name])
                  for name in ('DRACH', 'non_DRACH', 'unknown')}
        evaluations.append(dict(variant, score_index=i,
            metrics=dict(overall=classification_metrics(labels, current), by_motif=groups),
            class_flip_rate=float(np.mean(current.argmax(axis=1) != baseline_classes)),
            mean_true_class_probability_change=float(np.mean(
                current[np.arange(len(rows)), labels] - baseline[np.arange(len(rows)), labels])),
            mean_absolute_probability_change=float(np.mean(np.abs(current - baseline)))))
    return evaluations
