"""Additional requirements for m5C/psi motif retention; known-site rules stay independent."""
import math

AUXILIARY_TARGETS = ('m5C', 'pseU')
DEFAULT_MIN_DEPTH = 20
DEFAULT_MIN_AF = .40
MIN_MODIFIED = 3


def validate_auxiliary_motif_thresholds(min_depth, min_af):
    if isinstance(min_depth, bool) or not isinstance(min_depth, int) or min_depth < 1:
        raise ValueError('filtering.auxiliary_motif_min_depth must be a positive integer')
    if isinstance(min_af, bool) or not isinstance(min_af, (int,float)) or not math.isfinite(min_af) or not 0 <= min_af <= 1:
        raise ValueError('filtering.auxiliary_motif_min_af must be finite and between zero and one')


def auxiliary_motif_eligible(row, min_depth=DEFAULT_MIN_DEPTH, min_af=DEFAULT_MIN_AF):
    """Recompute AF from original counts, including for reused routed metadata."""
    validate_auxiliary_motif_thresholds(min_depth, min_af)
    if row['modification'] not in AUXILIARY_TARGETS:
        return True
    try:
        depth, modified = int(row['total_reads']), int(row['modified_reads'])
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('m5C/psi output requires original total_reads and modified_reads') from exc
    if depth < 0 or modified < 0 or modified > depth:
        raise ValueError('Invalid original m5C/psi candidate counts')
    # Source bedMethyl AF columns may use a different denominator. These counts
    # are the original canonical pipeline D=valid+fail+diff+nocall values.
    fraction = modified/depth if depth else 0.
    return depth >= min_depth and modified >= MIN_MODIFIED and fraction >= min_af
