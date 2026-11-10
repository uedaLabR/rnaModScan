import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.feature_reliance.perturbations import (
    PoolBank, apply_reference, depth_bins, make_site_donors, observed_bases,
)
from utils.nn.inputs import trimer_tokens


def inputs_for(sequences, depths, slots=None):
    """Synthetic coherent MATCH/N records with identifiable joint ML donors."""
    slots = slots or max(depths) + 1
    count, window = len(sequences), len(sequences[0])
    features = np.zeros((count, slots, window, 18), dtype=np.float32)
    read_mask = np.zeros((count, slots), dtype=bool)
    rows = []
    for site, (sequence, depth) in enumerate(zip(sequences, depths)):
        rows.append(dict(reference_sequence=sequence, retained_depth=depth,
                         row_id='site_' + str(site), label='not used'))
        read_mask[site, :depth] = True
        for read in range(depth):
            for offset, base in enumerate(sequence):
                code = 'ACGTN'.index(base) + 1
                features[site, read, offset, :8] = (
                    code * 0.05 + offset * 0.002 + read * 0.0001 + site * 0.001
                    + np.arange(8, dtype=np.float32) * 0.00001)
                features[site, read, offset, 8] = 1 / (20 + code + offset + site + read)
                features[site, read, offset, 10 if base != 'N' else 15] = 1
                features[site, read, offset, 16:] = 1
        # Adversarial nonzero padding proves that masking, not zero values,
        # defines which reads can supply or receive empirical donor cells.
        features[site, depth:, :, :] = 0.875
    inputs = dict(sequence_tokens=np.asarray([trimer_tokens(s) for s in sequences], np.int32),
                  sequence_repeat_mask=np.asarray([
                      [(i + site) % 3 == 0 for i in range(window)] for site in range(count)], bool),
                  read_features=features, read_mask=read_mask)
    return inputs, rows


class FeatureRelianceTest(unittest.TestCase):
    def assert_inputs_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)

    def test_observed_base_reconstruction_respects_mismatch_n_absence_and_padding(self):
        inputs, rows = inputs_for(['ACGTN' + 'A' * 14], [2])
        features = inputs['read_features']
        features[0, 0, 1, 10:16] = 0
        features[0, 0, 1, 13] = 1  # G mismatches reference C.
        features[0, 0, 2, 10:16] = 0
        features[0, 0, 2, 15] = 1  # N mismatches reference G.
        features[0, 0, 3, 10:16] = 0
        bases = observed_bases(inputs, rows)
        np.testing.assert_array_equal(bases[0, 0, :5], [1, 3, 5, 0, 5])
        np.testing.assert_array_equal(bases[0, 1, :5], [1, 2, 3, 4, 5])
        self.assertTrue(np.all(bases[0, 2] == 0))

    def test_uniform_priority_pool_is_bounded_and_keeps_joint_source_vectors(self):
        inputs, rows = inputs_for(['A' * 19] * 2, [20, 20])
        bank = PoolBank(capacity=5, seed=42)
        for site in range(2):
            batch = {name: values[site:site + 1] for name, values in inputs.items()}
            bank.update(batch, rows[site:site + 1])
        bank.finalize()
        for offset in range(19):
            pool = bank.ml[(0, offset, 1)]
            self.assertEqual(pool['seen'], 40)
            self.assertEqual(pool['values'].shape, (5, 8))
            valid = {tuple(v) for v in inputs['read_features'][:, :20, offset, :8].reshape(-1, 8)}
            self.assertTrue(all(tuple(v) in valid for v in pool['values']))
        # Later batches have a real opportunity to replace earlier donors.
        later = {tuple(v) for v in inputs['read_features'][1, :20, 0, :8]}
        self.assertTrue(any(tuple(v) in later for v in bank.ml[(0, 0, 1)]['values']))
        self.assertFalse(bank.ml[(0, 0, 1)]['values'].flags.writeable)

    def test_neighbor_ml_replacement_preserves_strata_center_and_other_inputs(self):
        source, rows = inputs_for([base * 19 for base in 'ACGTN'], [2] * 5)
        bank = PoolBank(capacity=64).update(source, rows).finalize()
        target = {name: values.copy() for name, values in source.items()}
        target['read_features'][:, :2, :, :8] = 0
        # An absent position must not receive ML even if a donor exists.
        target['read_features'][0, 0, 3, 10:16] = 0
        before = {name: values.copy() for name, values in target.items()}
        result = bank.replace(target, rows, 'neighbor_ml', np.random.default_rng(17))
        bases = observed_bases(target, rows)
        for site, read, offset in np.ndindex(bases.shape):
            base = int(bases[site, read, offset])
            after = result['read_features'][site, read, offset, :8]
            if 1 <= base <= 4 and offset != 9:
                valid = bank.ml[(0, offset, base)]['values']
                self.assertTrue(np.any(np.all(valid == after, axis=1)))
            else:
                np.testing.assert_array_equal(after, before['read_features'][site, read, offset, :8])
        np.testing.assert_array_equal(result['read_features'][..., 8:], before['read_features'][..., 8:])
        for name in ('read_mask', 'sequence_tokens', 'sequence_repeat_mask'):
            np.testing.assert_array_equal(result[name], before[name])
        self.assert_inputs_equal(target, before)
        stats = bank.report()['replacements']['neighbor_ml']
        self.assertEqual(stats['eligible_cells'], 4 * 2 * 18 - 1)
        self.assertEqual(stats['changed_cells'], stats['eligible_cells'])
        self.assertEqual(stats['replacement_fraction'], 1)

    def test_center_and_all_ml_controls_use_only_requested_positions(self):
        source, rows = inputs_for(['A' * 19], [2])
        bank = PoolBank().update(source, rows).finalize()
        target = {name: values.copy() for name, values in source.items()}
        target['read_features'][:, :2, :, :8] = 0
        center = bank.replace(target, rows, 'center_ml', np.random.default_rng(43))
        all_ml = bank.replace(target, rows, 'all_ml', np.random.default_rng(43))
        self.assertTrue(np.all(center['read_features'][:, :2, 9, :8] > 0))
        np.testing.assert_array_equal(center['read_features'][:, :2, :9, :8], 0)
        np.testing.assert_array_equal(center['read_features'][:, :2, 10:, :8], 0)
        self.assertTrue(np.all(all_ml['read_features'][:, :2, :, :8] > 0))
        np.testing.assert_array_equal(all_ml['read_features'][..., 8:], target['read_features'][..., 8:])
        np.testing.assert_array_equal(all_ml['read_features'][:, 2:], target['read_features'][:, 2:])

    def test_quality_preserves_zero_missing_absence_padding_and_n(self):
        source, rows = inputs_for(['A' * 19, 'N' * 19], [2, 2])
        bank = PoolBank().update(source, rows).finalize()
        self.assertIn((0, 0, 5), bank.quality)
        self.assertNotIn((0, 0, 5), bank.ml)
        target = {name: values.copy() for name, values in source.items()}
        target['read_features'][:, :2, :, 8] = 0.5
        target['read_features'][0, 0, 0, 8] = 0  # Genuine missing quality.
        target['read_features'][0, 0, 1, 10:16] = 0  # Uncovered position.
        before = {name: values.copy() for name, values in target.items()}
        result = bank.replace(target, rows, 'quality', np.random.default_rng(101))
        self.assertEqual(result['read_features'][0, 0, 0, 8], 0)
        self.assertEqual(result['read_features'][0, 0, 1, 8], 0.5)
        np.testing.assert_array_equal(result['read_features'][:, 2:], before['read_features'][:, 2:])
        np.testing.assert_array_equal(result['read_features'][..., :8], before['read_features'][..., :8])
        np.testing.assert_array_equal(result['read_features'][..., 9:], before['read_features'][..., 9:])
        self.assertNotEqual(result['read_features'][1, 0, 0, 8], 0.5)
        self.assert_inputs_equal(target, before)

    def test_depth_and_offset_strata_cannot_supply_each_other(self):
        source, source_rows = inputs_for(['A' * 19], [2])
        target, target_rows = inputs_for(['A' * 19], [30])
        bank = PoolBank().update(source, source_rows).finalize()
        result = bank.replace(target, target_rows, 'all_ml', np.random.default_rng(17))
        self.assert_inputs_equal(result, target)
        stats = bank.report()['replacements']['all_ml']
        self.assertEqual(stats['eligible_cells'], 30 * 19)
        self.assertEqual(stats['missing_pool_cells'], 30 * 19)
        self.assertEqual(stats['replacement_fraction'], 0)
        # Offset9's large, isolated donor score must not appear at offset0.
        source['read_features'][:, :2, :, :8] = 0
        source['read_features'][:, :2, 9, :8] = 1
        offset_bank = PoolBank().update(source, source_rows).finalize()
        result = offset_bank.replace(source, source_rows, 'all_ml', np.random.default_rng(17))
        np.testing.assert_array_equal(result['read_features'][:, :2, 0, :8], 0)
        np.testing.assert_array_equal(result['read_features'][:, :2, 9, :8], 1)

    def test_zero_ml_is_a_valid_donor_and_change_counts_are_not_draw_counts(self):
        inputs, rows = inputs_for(['A' * 19], [2])
        inputs['read_features'][:, :2, :, :8] = 0
        bank = PoolBank().update(inputs, rows).finalize()
        output = bank.replace(inputs, rows, 'all_ml', np.random.default_rng(43))
        self.assert_inputs_equal(output, inputs)
        stats = bank.report()['replacements']['all_ml']
        self.assertEqual(stats['eligible_cells'], 38)
        self.assertEqual(stats['replaced_cells'], 38)
        self.assertEqual(stats['changed_cells'], 0)
        self.assertEqual(bank.ml[(0, 0, 1)]['seen'], 2)

    def test_saved_pools_restore_identical_replacements_without_pickle(self):
        inputs, rows = inputs_for(['A' * 19, 'C' * 19], [2, 2])
        bank = PoolBank(capacity=3).update(inputs, rows).finalize()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pool.npz'
            bank.save(path)
            loaded = PoolBank.load(path)
            self.assertEqual(loaded.report()['strata'], bank.report()['strata'])
            for condition in ('neighbor_ml', 'center_ml', 'all_ml', 'quality'):
                expected = bank.replace(inputs, rows, condition, np.random.default_rng(43))
                actual = loaded.replace(inputs, rows, condition, np.random.default_rng(43))
                self.assert_inputs_equal(actual, expected)
            with self.assertRaises(FileExistsError):
                bank.save(path)
        with self.assertRaises(RuntimeError):
            loaded.update(inputs, rows)

    def test_reference_donors_are_label_free_depth_matched_true_permutations(self):
        depths = [1, 29, 30, 99, 100, 299, 300, 300, 300]
        rows = [dict(retained_depth=depth, label=str(index % 3)) for index, depth in enumerate(depths)]
        donors = make_site_donors(rows, 17)
        np.testing.assert_array_equal(np.sort(donors), np.arange(len(rows)))
        np.testing.assert_array_equal(depth_bins(rows)[donors], depth_bins(rows))
        self.assertTrue(np.all(donors != np.arange(len(rows))))
        changed_labels = [{**row, 'label': 'arbitrary changed label'} for row in rows]
        np.testing.assert_array_equal(make_site_donors(changed_labels, 17), donors)
        np.testing.assert_array_equal(make_site_donors([dict(retained_depth=30)], 17), [0])

    def test_reference_replacement_keeps_complete_overlapping_windows_and_joint_donor(self):
        inputs, rows = inputs_for(['A' * 19, 'CGTACGTACGTACGTACGT'], [2, 2])
        before = {name: values.copy() for name, values in inputs.items()}
        donors = make_site_donors(rows, 17)
        sequence = apply_reference(inputs, inputs['sequence_tokens'], inputs['sequence_repeat_mask'], donors, 'sequence')
        repeat = apply_reference(inputs, inputs['sequence_tokens'], inputs['sequence_repeat_mask'], donors, 'repeat')
        joint = apply_reference(inputs, inputs['sequence_tokens'], inputs['sequence_repeat_mask'], donors, 'sequence_repeat')
        np.testing.assert_array_equal(sequence['sequence_tokens'], inputs['sequence_tokens'][donors])
        np.testing.assert_array_equal(sequence['sequence_repeat_mask'], inputs['sequence_repeat_mask'])
        np.testing.assert_array_equal(repeat['sequence_repeat_mask'], inputs['sequence_repeat_mask'][donors])
        np.testing.assert_array_equal(repeat['sequence_tokens'], inputs['sequence_tokens'])
        np.testing.assert_array_equal(joint['sequence_tokens'], sequence['sequence_tokens'])
        np.testing.assert_array_equal(joint['sequence_repeat_mask'], repeat['sequence_repeat_mask'])
        for result in (sequence, repeat, joint):
            np.testing.assert_array_equal(result['read_features'], before['read_features'])
            np.testing.assert_array_equal(result['read_mask'], before['read_mask'])
        self.assert_inputs_equal(inputs, before)
        joint['read_mask'][0] = False
        self.assert_inputs_equal(inputs, before)

    def test_input_and_state_guards(self):
        inputs, rows = inputs_for(['A' * 19], [2])
        with self.assertRaises(ValueError):
            PoolBank(capacity=0)
        with self.assertRaises(ValueError):
            depth_bins([dict(retained_depth=301)])
        with self.assertRaises(RuntimeError):
            PoolBank().replace(inputs, rows, 'quality', np.random.default_rng(17))
        wrong_depth = [{**rows[0], 'retained_depth': 1}]
        with self.assertRaises(ValueError):
            PoolBank().update(inputs, wrong_depth)
        wrong_type = dict(inputs, read_features=inputs['read_features'].astype(np.float64))
        with self.assertRaises(ValueError):
            PoolBank().update(wrong_type, rows)
        with self.assertRaises(ValueError):
            apply_reference(inputs, inputs['sequence_tokens'], inputs['sequence_repeat_mask'], np.asarray([-1]), 'sequence')


if __name__ == '__main__':
    unittest.main()
