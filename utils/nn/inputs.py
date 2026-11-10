"""Model input conversion; raw Parquet values remain unchanged. No TensorFlow needed."""
import math
from functools import lru_cache

MODIFICATIONS = ('m6A', 'inosine', 'm5C', 'pseU', '2OmeA', '2OmeC', '2OmeG', '2OmeU')


def label_values(target):
    return {'0', '1', '2'} if target == 'adenine' else {'0', '1'}


CHANNEL_NAMES = (
    tuple('ml_' + m for m in MODIFICATIONS)
    + ('inverse_quality',)
    + ('log1p_insertion',)
    + ('base_match',) + tuple('base_' + b for b in 'ACGTN')
    + ('splice_before_log_distance', 'splice_after_log_distance')
)


def trimer_tokens(sequence):
    """ACGT base-4 coding; 64=unknown. Token i centers reference position i+1."""
    code = {base: i for i, base in enumerate('ACGT')}
    sequence = sequence.upper()
    tokens = []
    for start in range(len(sequence) - 2):
        trimer = sequence[start:start + 3]
        if any(base not in code for base in trimer):
            tokens.append(64)
            continue
        token = 0
        for base in trimer:
            token = token * 4 + code[base]
        tokens.append(token)
    return tokens


def encode_splice_distances(row):
    """Encode two same-read junction distances; missing and >=1000 both map to 1."""
    values = []
    maximum_distance = 1000
    normalization = math.log1p(maximum_distance)
    for key in ('splice_distance_before', 'splice_distance_after'):
        distance = row[key]  # Missing columns are an incompatible pre-v3 schema.
        if distance is None:
            distance = maximum_distance
        if isinstance(distance, bool) or not isinstance(distance, int) or distance < 0:
            raise ValueError('Splice distance must be a nonnegative query-base count or null')
        normalized_distance = math.log1p(min(distance, maximum_distance)) / normalization
        values.append(normalized_distance)
    return values


def read_base_one_hot(base, mismatch):
    """MATCH/A/C/G/T/N; only known ACGT matches use MATCH, absent bases use zeros.

    BAM extraction already compares RNA-oriented read and reference bases.
    Unknown reference bases have mismatch=None, so the observed read base is
    retained. N always has its own channel, even when the reference is also N.
    """
    category = 'match' if base in ('A', 'C', 'G', 'T') and mismatch == 0 else base
    return [float(category == value) for value in ('match', 'A', 'C', 'G', 'T', 'N')]


@lru_cache(maxsize=8192)
def numeric_base_features(scores, quality, insertion, base, mismatch):
    """Bounded value cache for overlapping windows; no coordinates or reads retained."""
    if len(scores) != len(MODIFICATIONS):
        raise ValueError('Expected eight modification channels')
    if any(score is not None and not 0 <= score <= 255 for score in scores):
        raise ValueError('ML byte outside 0..255')
    values = [score / 255 if score is not None else 0.0 for score in scores]
    values.append(1 / (quality + 1) if quality is not None else 0.0)
    values.append(math.log1p(insertion))
    values.extend(read_base_one_hot(base, mismatch))
    return tuple(values)


def encode_read_row(row):
    """Convert raw windows to 18 channels; splice distances remain site-specific."""
    splice = encode_splice_distances(row)
    result = []
    for i, base in enumerate(row['base']):
        values = numeric_base_features(tuple(row['ml_byte'][i]), row['base_quality'][i],
            row['insertion_length_after'][i], base, row['mismatch'][i])
        result.append(list(values) + splice)
    return result
