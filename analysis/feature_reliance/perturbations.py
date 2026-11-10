"""Label-free empirical replacements for frozen-model reliance diagnostics.

Read-position replacements preserve observed-base/offset/depth strata, not the
original within-read correlations. They are not strict whole-bag permutations.
The caller keeps the original cohort, labels and trained model unchanged.
"""
import json
from pathlib import Path

import numpy as np

SCHEMA = 'stratified_empirical_donors_v1'
BASE_NAMES = ('absent', 'A', 'C', 'G', 'T', 'N')
DEPTH_NAMES = ('1-29', '30-99', '100-299', '300')
READ_CONDITIONS = ('neighbor_ml', 'center_ml', 'all_ml', 'quality')
REFERENCE_CONDITIONS = ('sequence', 'repeat', 'sequence_repeat')


def depth_bins(rows):
    """Use only retained read counts; labels and outcomes never define strata."""
    depths = np.asarray([int(row['retained_depth']) for row in rows], dtype=np.int32)
    if np.any((depths < 1) | (depths > 300)):
        raise ValueError('Retained depth must be in 1..300')
    return np.searchsorted(np.asarray([30, 100, 300]), depths, side='right')


def _copy_inputs(inputs):
    return {name: np.array(values, copy=True) for name, values in inputs.items()}


def _checked_inputs(inputs, rows, window):
    features = np.asarray(inputs['read_features'])
    mask = np.asarray(inputs['read_mask'])
    if (features.ndim != 4 or features.shape[2:] != (window, 18)
            or features.dtype != np.float32 or mask.dtype != np.bool_
            or mask.shape != features.shape[:2] or len(rows) != len(features)):
        raise ValueError('Use float32 [sites,reads,window,18] and a boolean read mask')
    depths = np.asarray([int(row['retained_depth']) for row in rows], dtype=np.int32)
    if not np.array_equal(mask.sum(axis=1), depths):
        raise ValueError('Retained depth must equal the real-read mask count')
    bins = depth_bins(rows)
    return features, mask, bins


def observed_bases(inputs, rows, window=19):
    """Recover RNA-oriented observed A/C/G/T/N from MATCH and mismatch one-hots.

    A MATCH has the corresponding reference base; a mismatching A..N has its
    own channel. All-zero categories and padded reads remain absent. Return
    integer codes 0=absent, 1=A, 2=C, 3=G, 4=T, 5=N.
    """
    features, mask, _ = _checked_inputs(inputs, rows, window)
    sequences = [row['reference_sequence'].upper() for row in rows]
    if any(len(sequence) != window for sequence in sequences):
        raise ValueError('Reference windows must match the tensor window')
    code = {base: index for index, base in enumerate('ACGTN', 1)}
    reference = np.asarray([[code.get(base, 5) for base in sequence]
                            for sequence in sequences], dtype=np.int8)
    one_hot = features[..., 10:16]
    categories = one_hot.argmax(axis=-1).astype(np.int8)
    bases = np.where(categories > 0, categories, reference[:, None, :]).astype(np.int8)
    present = np.any(one_hot != 0, axis=-1) & mask[..., None]
    bases[~present] = 0
    return bases


class PoolBank:
    """Bounded, uniform priority samples of observed input cells by stratum.

    Each ML donor is a joint eight-channel vector. Each quality donor is one
    positive inverse-quality value. Independent uniform priorities across the
    entire update stream select at most ``capacity`` cells per stratum.
    """
    def __init__(self, window=19, capacity=4096, seed=42):
        if type(window) is not int or window < 3 or window % 2 != 1:
            raise ValueError('Use an odd window >=3')
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Capacity must be a positive integer')
        self.window, self.capacity, self.seed = window, capacity, int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.ml, self.quality = {}, {}
        self.finalized = False
        self.replacement_stats = {}

    def _update_pool(self, bank, key, values):
        values = np.asarray(values, dtype=np.float32)
        if not len(values):
            return
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('ML and inverse-quality donors must be finite in 0..1')
        priorities = self.rng.random(len(values))
        previous = bank.get(key)
        seen = len(values)
        if previous is not None:
            seen += previous['seen']
            values = np.concatenate((previous['values'], values), axis=0)
            priorities = np.concatenate((previous['priorities'], priorities))
        if len(values) > self.capacity:
            selected = np.argpartition(priorities, self.capacity - 1)[:self.capacity]
            values, priorities = values[selected], priorities[selected]
        bank[key] = dict(values=values.copy(), priorities=priorities.copy(), seen=seen)

    def update(self, inputs, rows):
        if self.finalized:
            raise RuntimeError('A finalized pool cannot accept new data')
        features, _, bins = _checked_inputs(inputs, rows, self.window)
        bases = observed_bases(inputs, rows, self.window)
        for depth_bin in np.unique(bins):
            site_indices = np.flatnonzero(bins == depth_bin)
            for offset in range(self.window):
                observed = bases[site_indices, :, offset].reshape(-1)
                ml_values = features[site_indices, :, offset, :8].reshape(-1, 8)
                quality_values = features[site_indices, :, offset, 8].reshape(-1, 1)
                for base in np.unique(observed):
                    if base == 0:
                        continue
                    selected = observed == base
                    key = int(depth_bin), offset, int(base)
                    if base <= 4:
                        self._update_pool(self.ml, key, ml_values[selected])
                    positive_quality = selected & (quality_values[:, 0] > 0)
                    self._update_pool(self.quality, key, quality_values[positive_quality])
        return self

    def finalize(self):
        for bank in (self.ml, self.quality):
            for pool in bank.values():
                pool['values'].setflags(write=False)
        self.finalized = True
        return self

    def replace(self, inputs, rows, condition, rng):
        """Return a copy with only the requested eligible input cells replaced.

        Missing strata leave the recipient unchanged. Counts distinguish draw
        opportunities from numerical changes, since valid donors may be zero
        or may equal the original value. No label is read or used.
        """
        if not self.finalized:
            raise RuntimeError('Finalize donor pools before replacement')
        if condition not in READ_CONDITIONS:
            raise ValueError('Unknown read-feature condition: ' + str(condition))
        features, _, bins = _checked_inputs(inputs, rows, self.window)
        bases = observed_bases(inputs, rows, self.window)
        result = _copy_inputs(inputs)
        output = result['read_features']
        stats = self.replacement_stats.setdefault(condition, dict(
            calls=0, eligible_cells=0, replaced_cells=0, changed_cells=0, missing_pool_cells=0))
        stats['calls'] += 1
        center = self.window // 2
        offsets = (range(self.window) if condition in ('quality', 'all_ml')
                   else [center] if condition == 'center_ml'
                   else [offset for offset in range(self.window) if offset != center])
        bank = self.quality if condition == 'quality' else self.ml
        for depth_bin in np.unique(bins):
            for offset in offsets:
                for base in range(1, 6 if condition == 'quality' else 5):
                    selected = ((bins[:, None] == depth_bin) & (bases[:, :, offset] == base))
                    if condition == 'quality':
                        selected &= features[:, :, offset, 8] > 0
                    sites, reads = np.nonzero(selected)
                    count = len(sites)
                    if not count:
                        continue
                    stats['eligible_cells'] += count
                    pool = bank.get((int(depth_bin), offset, base))
                    if pool is None or not len(pool['values']):
                        stats['missing_pool_cells'] += count
                        continue
                    donor_indices = rng.integers(len(pool['values']), size=count)
                    donors = pool['values'][donor_indices]
                    if condition == 'quality':
                        before = features[sites, reads, offset, 8]
                        output[sites, reads, offset, 8] = donors[:, 0]
                        changed = before != donors[:, 0]
                    else:
                        before = features[sites, reads, offset, :8]
                        output[sites, reads, offset, :8] = donors
                        changed = np.any(before != donors, axis=1)
                    stats['replaced_cells'] += count
                    stats['changed_cells'] += int(changed.sum())
        return result

    def report(self):
        strata = []
        for name, bank in [('ml', self.ml), ('quality', self.quality)]:
            for (depth_bin, offset, base), pool in sorted(bank.items()):
                strata.append(dict(feature=name, depth_bin=DEPTH_NAMES[depth_bin],
                    position=offset - self.window // 2, observed_base=BASE_NAMES[base],
                    seen_cells=int(pool['seen']), retained_donors=len(pool['values'])))
        replacements = {}
        for condition, values in self.replacement_stats.items():
            record = dict(values)
            eligible = record['eligible_cells']
            record['replacement_fraction'] = record['replaced_cells'] / eligible if eligible else None
            record['changed_fraction'] = record['changed_cells'] / eligible if eligible else None
            replacements[condition] = record
        return dict(schema=SCHEMA, window=self.window, capacity=self.capacity, seed=self.seed,
            finalized=self.finalized, strata=strata, replacements=replacements,
            interpretation='Stratified empirical donor replacement; not causal or additive importance',
            count_scope='Read-position opportunities accumulated across calls; not unique reads or sites')

    def save(self, path):
        """Save immutable donor arrays and non-pickled JSON metadata, exclusively."""
        if not self.finalized:
            raise RuntimeError('Finalize donor pools before saving')
        arrays, pools = {}, []
        for name, bank in [('ml', self.ml), ('quality', self.quality)]:
            for number, (key, pool) in enumerate(sorted(bank.items())):
                array_name = name + '_' + str(number)
                arrays[array_name] = pool['values']
                pools.append(dict(feature=name, key=list(key), seen=int(pool['seen']), array=array_name))
        metadata = dict(schema=SCHEMA, window=self.window, capacity=self.capacity,
                        seed=self.seed, pools=pools)
        arrays['metadata'] = np.asarray(json.dumps(metadata))
        with Path(path).open('xb') as handle:
            np.savez_compressed(handle, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata'].item()))
            if metadata.get('schema') != SCHEMA:
                raise ValueError('Unknown donor-pool schema')
            bank = cls(metadata['window'], metadata['capacity'], metadata['seed'])
            for entry in metadata['pools']:
                name, key = entry['feature'], tuple(entry['key'])
                if (name not in ('ml', 'quality') or len(key) != 3
                        or not 0 <= key[0] < 4 or not 0 <= key[1] < bank.window
                        or not 1 <= key[2] <= (4 if name == 'ml' else 5)):
                    raise ValueError('Invalid donor-pool stratum')
                values = archive[entry['array']].copy()
                width = 8 if name == 'ml' else 1
                if (values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != width
                        or not 1 <= len(values) <= bank.capacity or entry['seen'] < len(values)
                        or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))
                        or (name == 'quality' and np.any(values <= 0))):
                    raise ValueError('Invalid donor-pool values')
                target = bank.ml if name == 'ml' else bank.quality
                if key in target:
                    raise ValueError('Duplicate donor-pool stratum')
                target[key] = dict(values=values, seen=int(entry['seen']))
        return bank.finalize()


def make_site_donors(rows, seed):
    """Permute whole site references within retained-depth bins, without labels.

    A random cyclic order avoids self-donation whenever a bin has two or more
    sites. Singleton bins remain unchanged and must be reported by the caller.
    This is a whole-reference permutation, not per-position token shuffling.
    """
    bins = depth_bins(rows)
    rng = np.random.default_rng(seed)
    donors = np.arange(len(rows), dtype=np.int64)
    for depth_bin in np.unique(bins):
        sites = np.flatnonzero(bins == depth_bin)
        if len(sites) > 1:
            order = rng.permutation(sites)
            donors[order] = np.roll(order, 1)
    return donors


def apply_reference(inputs, all_tokens, all_repeat, donor_indices, condition):
    """Copy a coherent reference window and/or mask from the same donor site."""
    if condition not in REFERENCE_CONDITIONS:
        raise ValueError('Unknown reference condition: ' + str(condition))
    tokens, repeat = np.asarray(all_tokens), np.asarray(all_repeat)
    indices = np.asarray(donor_indices)
    count = len(inputs['read_mask'])
    window = inputs['read_features'].shape[2]
    if (tokens.ndim != 2 or tokens.shape[1] != window - 2 or tokens.dtype != np.int32
            or repeat.shape != (len(tokens), window) or repeat.dtype != np.bool_
            or indices.shape != (count,) or indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices >= len(tokens))):
        raise ValueError('Use cohort reference arrays and one valid donor index per recipient')
    result = _copy_inputs(inputs)
    if condition in ('sequence', 'sequence_repeat'):
        result['sequence_tokens'] = tokens[indices].copy()
    if condition in ('repeat', 'sequence_repeat'):
        result['sequence_repeat_mask'] = repeat[indices].copy()
    return result
