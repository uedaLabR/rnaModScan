"""Read per-site Parquet bags directly into NN batches; pad only in memory."""
from collections import OrderedDict
from pathlib import Path

import numpy as np

from utils.adenine import CLASSES
from utils.features.dataset import SiteBatches
from utils.features.parquet import require_arrow
from utils.io.artifacts import completed_run
from utils.nn.inputs import CHANNEL_NAMES
from utils.features.compact_inputs import SITE_SCHEMAS, tensor_columns, tensor_column, restore_read_inputs


LABEL_COLUMNS = ('training_class', 'training_label_m6A', 'training_label_inosine',
                 'training_label_false_positive')


class SiteBagBatches(SiteBatches):
    """Use indexed ragged site bags without revisiting BAM or raw read rows.

    The lightweight TSV uses the usual site/split fields plus site_parquet,
    site_row_group and site_row_offset. Relative Parquet paths are resolved
    against features_dir, including the sites/ prefix. Only tensors and small
    label/ID columns are decompressed. No per-read Python conversion is needed.
    """
    def __init__(self, index_path, split_path, *, batch_sites=16, max_reads=300,
                 sampling_seed=42, require_labels=True, cache_groups=4):
        if type(max_reads) is not int or max_reads < 1:
            raise ValueError('Site bag batches need a positive max_reads')
        if type(cache_groups) is not int or cache_groups < 1:
            raise ValueError('cache_groups must be positive')
        super().__init__(index_path, split_path, batch_sites=batch_sites, max_reads=max_reads,
                         sampling_seed=sampling_seed, require_labels=require_labels)
        self.pa, self.pq = require_arrow()
        self.cache_groups = cache_groups
        self.cache = OrderedDict()
        self.files = {}
        self.sources = {}
        self._check_sources()
        self.row_groups = {}
        for index, row in enumerate(self.rows):
            key = row['site_parquet'], int(row['site_row_group'])
            self.row_groups.setdefault(key, []).append(index)

    def _check_sources(self):
        """Bind each indexed file to the completed extraction and sampling policy."""
        for directory in {row['features_dir'] for row in self.rows}:
            source = Path(directory)
            run = completed_run(source)
            if run.get('schema') not in SITE_SCHEMAS:
                raise ValueError('Use site-bag Parquet v3/v4 with saved repeat masks; regenerate old site bags')
            if (run.get('max_reads') != self.max_reads
                    or run.get('sampling_seed') != self.sampling_seed):
                raise ValueError('Site bag sampling policy differs from model settings')
            self.sources[directory] = run
            for item in run['parquet_files']['sites']:
                path = (source / 'sites' / item['path']).resolve()
                if not path.is_relative_to(source):
                    raise ValueError('Site Parquet manifest escapes its source directory')
                self.files[path] = item
        seen = set()
        resolved_paths = {}
        for row in self.rows:
            if row['row_id'] in seen:
                raise ValueError('Duplicate site ID in feature index')
            seen.add(row['row_id'])
            if self.sources[row['features_dir']].get('sample_id') != row['sample_id']:
                raise ValueError('Site sample ID disagrees with its feature source')
            key = row['features_dir'], row['site_parquet']
            if key not in resolved_paths:
                path = Path(row['site_parquet'])
                source = Path(row['features_dir'])
                path = path.resolve() if path.is_absolute() else (source / path).resolve()
                if not path.is_relative_to(source) or path not in self.files:
                    raise ValueError('Site Parquet is outside the completed feature source')
                resolved_paths[key] = str(path)
            row['site_parquet'] = resolved_paths[key]
            if not 1 <= int(row['retained_depth']) <= self.max_reads:
                raise ValueError('Retained depth must be between one and max_reads')
            if int(row['eligible_depth']) < int(row['retained_depth']):
                raise ValueError('Eligible depth cannot be below retained depth')
            if int(row['site_row_group']) < 0 or int(row['site_row_offset']) < 0:
                raise ValueError('Site row-group offsets must be nonnegative')

    def _shuffled_row_group_order(self, rng):
        """Visit each selected row group once, using no labels to choose order."""
        groups = list(self.row_groups)
        rng.shuffle(groups)
        order = []
        for key in groups:
            selected = list(self.row_groups[key])
            rng.shuffle(selected)
            order.extend(selected)
        return order

    def batches(self, shuffle=False, rng=None):
        """Shuffle groups then their sites for training; preserve audit order otherwise."""
        order = (self._shuffled_row_group_order(rng or np.random.default_rng(42))
                 if shuffle else range(len(self.rows)))
        for start in range(0, len(order), self.batch_sites):
            rows = [self.rows[index] for index in order[start:start + self.batch_sites]]
            yield self._batch(rows)

    def _row_group(self, row):
        path = Path(row['site_parquet'])
        group = int(row['site_row_group'])
        key = path, group
        if key not in self.cache:
            if path.stat().st_size != self.files[path]['size_bytes']:
                raise ValueError('Site Parquet file changed after indexing')
            parquet = self.pq.ParquetFile(path)
            tensor_column(parquet.schema_arrow)
            if group >= parquet.metadata.num_row_groups:
                raise ValueError('Site row group exceeds the Parquet file')
            names = tensor_columns(parquet.schema_arrow) + (list(LABEL_COLUMNS) if self.require_labels else [])
            table = parquet.read_row_group(group, columns=names)
            self.cache[key] = table
            if len(self.cache) > self.cache_groups:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def _site_tensors(self, row, window):
        table = self._row_group(row)
        offset = int(row['site_row_offset'])
        if offset >= table.num_rows or table['row_id'][offset].as_py() != row['row_id']:
            raise ValueError('Site Parquet offset does not match the indexed site ID')
        token_values = table['sequence_tokens'][offset]
        if not token_values.is_valid:
            raise ValueError('Site has null reference tokens')
        tokens = token_values.values.to_numpy(zero_copy_only=False)
        # The outer list is variable depth; the window and channel axes are fixed.
        features = restore_read_inputs(table[tensor_column(table.schema)][offset], window)
        depth = len(features)
        if depth != int(row['retained_depth']) or tokens.shape != (window - 2,):
            raise ValueError('Site tensor dimensions disagree with its metadata')
        if not np.isfinite(features).all() or tokens.dtype != np.int32:
            raise ValueError('Site bag contains invalid numeric tensors')
        if self.require_labels:
            expected = CLASSES[int(row['label'])]
            actual = table['training_class'][offset].as_py()
            flags = [table[name][offset].as_py() for name in LABEL_COLUMNS[1:]]
            if actual != expected or flags != [name == expected for name in CLASSES]:
                raise ValueError('Stored training flags disagree with the indexed label')
        mask_values = table['sequence_repeat_mask'][offset]
        if not mask_values.is_valid or mask_values.values.null_count:
            raise ValueError('A complete sequence repeat mask is required')
        mask = mask_values.values.to_numpy(zero_copy_only=False)
        if mask.shape != (window,) or mask.dtype != np.bool_:
            raise ValueError('Sequence repeat-mask window/dtype differs from NN settings')
        return tokens, features, mask

    def _batch(self, rows):
        if not rows:
            raise ValueError('Cannot build an empty NN batch')
        window = len(rows[0]['reference_sequence'])
        count = len(rows)
        inputs = dict(sequence_tokens=np.zeros((count, window - 2), np.int32),
                      sequence_repeat_mask=np.zeros((count, window), bool),
                      read_features=np.zeros((count, self.max_reads, window, len(CHANNEL_NAMES)), np.float32),
                      read_mask=np.zeros((count, self.max_reads), bool))
        labels = np.full(count, -1, dtype=np.int32)
        for index, row in enumerate(rows):
            if row['modification'] != 'adenine' or len(row['reference_sequence']) != window:
                raise ValueError('Use uniform adenine site bags for the joint classifier')
            tokens, features, repeat_mask = self._site_tensors(row, window)
            depth = len(features)
            inputs['sequence_tokens'][index] = tokens
            inputs['sequence_repeat_mask'][index] = repeat_mask
            inputs['read_features'][index, :depth] = features
            inputs['read_mask'][index, :depth] = True
            if row['label'] in ('0', '1', '2'):
                labels[index] = int(row['label'])
        return inputs, labels
