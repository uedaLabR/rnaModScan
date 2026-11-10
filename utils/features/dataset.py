"""Common site-index validation and TensorFlow batching for per-site Parquet."""
import csv
from pathlib import Path

import numpy as np

from utils.nn.inputs import CHANNEL_NAMES, label_values


class SiteBatches:
    """Share split metadata and tensor signatures with the site-Parquet reader."""
    def __init__(self, index_path, split_path, *, batch_sites=16, require_labels=True,
                 max_reads=300, sampling_seed=42):
        if batch_sites < 1:
            raise ValueError('batch_sites must be positive')
        index_path = Path(index_path)
        selected = None
        if split_path is not None:
            with Path(split_path).open(encoding='utf-8', newline='') as handle:
                selected = {row['row_id']: row for row in csv.DictReader(handle, delimiter='\t')}
        self.rows = []
        resolved_sources = {}
        with index_path.open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                if selected is not None and row['row_id'] not in selected:
                    continue
                if require_labels and row['label'] not in label_values(row['modification']):
                    raise ValueError('Training dataset contains an unlabeled row')
                if int(row['retained_depth']) == 0:
                    if require_labels:
                        raise ValueError('Training index contains a zero-read site')
                    continue
                if selected is not None:
                    for key in ('sample_id', 'chrom', 'start', 'strand', 'modification'):
                        if row[key] != selected[row['row_id']][key]:
                            raise ValueError('CV manifest and index disagree')
                source = row['features_dir']
                if source not in resolved_sources:
                    resolved_sources[source] = str((index_path.parent / source).resolve())
                row['features_dir'] = resolved_sources[source]
                row['group_id'] = selected[row['row_id']]['group_id'] if selected is not None else ''
                self.rows.append(row)
        if (selected is not None and len(self.rows) != len(selected)) or (require_labels and not self.rows):
            raise ValueError('Empty split or split IDs missing from feature index')
        self.max_reads, self.sampling_seed = max_reads, sampling_seed
        self.batch_sites = batch_sites
        self.require_labels = require_labels
        self.target = self.rows[0]['modification'] if self.rows else None

    def batches(self, shuffle=False, rng=None):
        """Read one batch at a time; a concrete reader supplies _batch()."""
        order = np.arange(len(self.rows))
        if shuffle:
            (rng or np.random.default_rng(42)).shuffle(order)
        for start in range(0, len(order), self.batch_sites):
            rows = [self.rows[index] for index in order[start:start + self.batch_sites]]
            yield self._batch(rows)

    def tensorflow_dataset(self, shuffle=False, seed=42):
        """Expose the four exact model inputs, padding only the current batch."""
        import tensorflow as tf
        if not self.require_labels:
            raise ValueError('Prediction-only datasets cannot be used for supervised training')
        window = len(self.rows[0]['reference_sequence'])
        signature = dict(sequence_tokens=tf.TensorSpec((None, window - 2), tf.int32),
                         sequence_repeat_mask=tf.TensorSpec((None, window), tf.bool),
                         read_features=tf.TensorSpec((None, self.max_reads, window, len(CHANNEL_NAMES)), tf.float32),
                         read_mask=tf.TensorSpec((None, self.max_reads), tf.bool))
        rng = np.random.default_rng(seed)
        label_spec = tf.TensorSpec((None,), tf.int32)
        dataset = tf.data.Dataset.from_generator(lambda: self.batches(shuffle, rng),
                      output_signature=(signature, label_spec))
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(
            (len(self.rows) + self.batch_sites - 1) // self.batch_sites))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        return dataset.with_options(options).prefetch(1)
