"""Restore a ragged Arrow site batch to fixed NN slots without Python float decoding."""
import numpy as np

from utils.nn.inputs import CHANNEL_NAMES
from utils.features.compact_inputs import tensor_column, restore_read_inputs


def site_batch_inputs(batch, window, max_reads):
    inputs = dict(sequence_tokens=np.zeros((batch.num_rows, window - 2), np.int32),
                  sequence_repeat_mask=np.zeros((batch.num_rows, window), bool),
                  read_features=np.zeros((batch.num_rows, max_reads, window, len(CHANNEL_NAMES)), np.float32),
                  read_mask=np.zeros((batch.num_rows, max_reads), bool))
    for index in range(batch.num_rows):
        tokens = batch.column(batch.schema.get_field_index('sequence_tokens'))[index].values
        repeat_mask = batch.column(batch.schema.get_field_index('sequence_repeat_mask'))[index]
        if not repeat_mask.is_valid or repeat_mask.values.null_count:
            raise ValueError('A complete sequence repeat mask is required')
        repeat_mask = repeat_mask.values.to_numpy(zero_copy_only=False)
        if repeat_mask.shape != (window,) or repeat_mask.dtype != bool:
            raise ValueError('Sequence repeat-mask shape or dtype differs from the NN input')
        values = batch.column(batch.schema.get_field_index(tensor_column(batch.schema)))[index]
        features = restore_read_inputs(values, window)
        depth = len(features)
        if not 1 <= depth <= max_reads:
            raise ValueError('Neural input needs at least one and at most max_reads reads')
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite neural input')
        inputs['sequence_tokens'][index] = tokens.to_numpy(zero_copy_only=False)
        inputs['sequence_repeat_mask'][index] = repeat_mask
        inputs['read_features'][index, :depth] = features
        inputs['read_mask'][index, :depth] = True
    return inputs
