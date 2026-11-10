"""Lossless integer storage for the existing 18-channel float32 NN inputs."""
import math

import numpy as np

SITE_SCHEMAS = ('site_bags_parquet_v3', 'site_bags_parquet_v4')
STORAGE = 'lossless_integer_inputs_v1'
ML_VALUES = np.asarray([value / 255 for value in range(256)], dtype=np.float32)
QUALITY_VALUES = np.asarray([0.0] + [1 / (quality + 1) for quality in range(256)], dtype=np.float32)
SPLICE_VALUES = np.asarray([math.log1p(distance) / math.log1p(1000) for distance in range(1001)], dtype=np.float32)
BASE_VALUES = np.vstack([np.zeros((1, 6), np.float32), np.eye(6, dtype=np.float32)])


def integer_read_type(window):
    from utils.features.parquet import require_arrow
    pa, _ = require_arrow()
    return pa.struct([
        ('ml_byte', pa.list_(pa.uint8(), window * 8)),
        ('quality_code', pa.list_(pa.uint16(), window)),
        ('insertion_length', pa.list_(pa.uint32(), window)),
        ('base_category', pa.list_(pa.uint8(), window)),
        ('splice_before', pa.uint16()), ('splice_after', pa.uint16())])


def clipped_splice(row, name):
    value = row[name]
    if value is None:
        return 1000
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('Splice distance must be a nonnegative query-base count or null')
    return min(value, 1000)


def pack_read_inputs(rows):
    """Pack only real reads; null versus zero remains in the unchanged raw audit."""
    if not rows:
        return []
    scores = np.asarray([row['ml_byte'] for row in rows], dtype=np.float64)
    if scores.ndim != 3 or scores.shape[2] != 8 or np.any((scores < 0) | (scores > 255)):
        raise ValueError('Expected eight ML bytes in 0..255 at every position')
    ml = np.nan_to_num(scores, nan=0).astype(np.uint8).reshape(len(rows), -1)
    qualities = np.asarray([row['base_quality'] for row in rows], dtype=np.float64)
    if np.any((qualities < 0) | (qualities > 255)):
        raise ValueError('Base quality outside 0..255')
    # Zero means missing. Q=255 needs code256, hence uint16 rather than uint8.
    quality = np.nan_to_num(qualities + 1, nan=0).astype(np.uint16)
    insertions = np.asarray([row['insertion_length_after'] for row in rows], dtype=np.int64)
    if np.any((insertions < 0) | (insertions > np.iinfo(np.uint32).max)):
        raise ValueError('Insertion length outside uint32 range')
    bases = np.asarray([row['base'] for row in rows], dtype='U4')
    mismatch = np.asarray([row['mismatch'] for row in rows], dtype=np.float64)
    category = np.zeros(bases.shape, dtype=np.uint8)
    for code, base in enumerate('ACGTN', 2):
        category[bases == base] = code
    category[(category >= 2) & (category <= 5) & (mismatch == 0)] = 1
    return [dict(ml_byte=ml[index].tolist(), quality_code=quality[index].tolist(),
                 insertion_length=insertions[index].tolist(), base_category=category[index].tolist(),
                 splice_before=clipped_splice(row, 'splice_distance_before'),
                 splice_after=clipped_splice(row, 'splice_distance_after'))
            for index, row in enumerate(rows)]


def tensor_column(schema):
    version = (schema.metadata or {}).get(b'schema_version', b'').decode()
    if version not in SITE_SCHEMAS:
        raise ValueError('Expected site-bag Parquet v3 or v4')
    return 'read_inputs' if version == 'site_bags_parquet_v4' else 'read_features'


def tensor_columns(schema):
    return ['row_id', 'sequence_tokens', 'sequence_repeat_mask', tensor_column(schema)]


def restore_read_inputs(value, window):
    """Arrow integer arrays -> NumPy lookup tables, with no per-read Python decoding."""
    from utils.features.parquet import require_arrow
    pa, _ = require_arrow()
    if not value.is_valid:
        raise ValueError('Site has null neural read inputs')
    reads = value.values
    depth = len(reads)
    if reads.type == pa.list_(pa.list_(pa.float32(), 18), window):
        return reads.flatten().flatten().to_numpy(zero_copy_only=False).reshape(depth, window, 18)
    if reads.type != integer_read_type(window) or reads.null_count:
        raise ValueError('Site-bag neural storage dtype/window mismatch')

    def values(name, width=None):
        array = reads.field(name)
        if array.null_count:
            raise ValueError('Integer neural inputs cannot contain null codes')
        if width is not None:
            array = array.flatten()
        if array.null_count:
            raise ValueError('Integer neural inputs cannot contain null codes')
        result = array.to_numpy(zero_copy_only=False)
        return result.reshape(depth, width) if width is not None else result

    ml = values('ml_byte', window * 8).reshape(depth, window, 8)
    quality = values('quality_code', window)
    category = values('base_category', window)
    before, after = values('splice_before'), values('splice_after')
    if np.any(quality > 256) or np.any(category > 6) or np.any(before > 1000) or np.any(after > 1000):
        raise ValueError('Invalid integer neural input code')
    insertions = values('insertion_length', window)
    unique, inverse = np.unique(insertions, return_inverse=True)
    # math.log1p matches the previous Python transform exactly before float32 rounding.
    insertion_values = np.asarray([math.log1p(int(number)) for number in unique], dtype=np.float32)
    result = np.empty((depth, window, 18), dtype=np.float32)
    result[:, :, :8] = ML_VALUES[ml]
    result[:, :, 8] = QUALITY_VALUES[quality]
    result[:, :, 9] = insertion_values[inverse].reshape(depth, window)
    result[:, :, 10:16] = BASE_VALUES[category]
    result[:, :, 16] = SPLICE_VALUES[before, None]
    result[:, :, 17] = SPLICE_VALUES[after, None]
    return result
