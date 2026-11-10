import importlib.util
import tempfile
import unittest
from pathlib import Path

TF_AVAILABLE = importlib.util.find_spec('tensorflow') is not None


@unittest.skipUnless(TF_AVAILABLE, 'TensorFlow not installed')
class KerasSiteModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import tensorflow as tf
        from utils.nn.keras_site import ModelConfig, build_site_model
        cls.np, cls.tf = np, tf
        cls.Config, cls.build_model = ModelConfig, staticmethod(build_site_model)
        tf.keras.utils.set_random_seed(42)

    def inputs(self):
        from utils.nn.inputs import CHANNEL_NAMES
        np = self.np
        rng = np.random.default_rng(3)
        return dict(sequence_tokens=rng.integers(0, 65, (2, 17), dtype=np.int32),
                    sequence_repeat_mask=rng.integers(0, 2, (2, 19)).astype(bool),
                    read_features=rng.random((2, 27, 19, len(CHANNEL_NAMES)), dtype=np.float32),
                    read_mask=np.array([[True] * 3 + [False] * 24, [True] * 27]))

    def model(self):
        return self.build_model(self.Config(max_reads=27))

    def test_global_attention_padding_and_permutation(self):
        from utils.nn.keras_site import attention_model
        np = self.np
        inputs, model = self.inputs(), self.model()
        prediction = model(inputs, training=False).numpy()
        self.assertEqual(prediction.shape, (2, 3))
        self.assertTrue(np.all((prediction >= 0) & (prediction <= 1)))
        weights = attention_model(model)(inputs).numpy()
        np.testing.assert_allclose(weights.sum(axis=1), [1, 1], atol=1e-6)
        np.testing.assert_array_equal(weights[0, 3:], 0)
        changed = dict(inputs)
        changed['read_features'] = inputs['read_features'].copy()
        changed['read_features'][0, 3:] = 1e6  # Masked padding must never enter CNN/attention.
        np.testing.assert_allclose(model(changed, training=False), prediction, atol=1e-6)
        permutation = np.arange(27)[::-1]
        changed['read_features'] = inputs['read_features'][:, permutation]
        changed['read_mask'] = inputs['read_mask'][:, permutation]
        np.testing.assert_allclose(model(changed, training=False), prediction, atol=2e-6)

    def test_graph_training_gradients_and_serialization(self):
        from utils.nn.keras_site import compile_site_model
        tf, np = self.tf, self.np
        model, inputs = self.model(), self.inputs()
        labels = tf.constant([0, 1])
        with tf.GradientTape() as tape:
            predicted = model(inputs, training=True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, predicted))
        self.assertEqual(predicted.shape, (2, 3))
        np.testing.assert_allclose(tf.reduce_sum(predicted, axis=1), [1, 1], atol=1e-6)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))
        self.assertTrue(all(np.isfinite(tf.convert_to_tensor(g).numpy()).all() for g in gradients))
        encoder = model.get_layer('shared_read_encoder').encoder
        kernel = encoder.layers[1].kernel
        index = next(i for i, variable in enumerate(model.trainable_variables) if variable is kernel)
        self.assertGreater(float(tf.norm(gradients[index])), 0)
        compile_site_model(model)
        metrics = model.train_on_batch(inputs, labels, return_dict=True)
        self.assertTrue(np.isfinite(metrics['loss']))
        expected = model(inputs, training=False).numpy()
        weight_names = [getattr(weight, 'path', weight.name) for weight in model.weights]
        self.assertEqual(len(weight_names), len(set(weight_names)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.keras'
            model.save(path)
            restored = tf.keras.models.load_model(path)
            self.assertEqual({tensor.name.split(':')[0] for tensor in restored.inputs},
                             {'sequence_tokens', 'sequence_repeat_mask', 'read_features', 'read_mask'})
            np.testing.assert_allclose(restored(inputs, training=False), expected, atol=1e-6)
            restored_read = restored.get_layer('shared_read_encoder')
            self.assertEqual((restored_read.hidden_dim, restored_read.attention_heads,
                              restored_read.transformer_ff), (16, 2, 32))

    def test_zero_read_sites_rejected(self):
        inputs, model = self.inputs(), self.model()
        inputs['read_mask'][0] = False
        with self.assertRaises(self.tf.errors.InvalidArgumentError):
            model(inputs, training=False)

    def test_reference_transformer_and_fixed_joint_input_policy(self):
        transformer = self.model()
        self.assertEqual(transformer.get_layer('sequence_self_attention').get_config()['num_heads'], 2)
        self.assertEqual(transformer.get_layer('site_fusion').output.shape[-1], 128)
        self.assertEqual({tensor.name.split(':')[0] for tensor in transformer.inputs},
                         {'sequence_tokens', 'sequence_repeat_mask', 'read_features', 'read_mask'})
        self.assertFalse(any('position_vector' == layer.name or 'site_stats' == layer.name
                             for layer in transformer.layers))
        self.assertEqual(transformer.output_shape, (None, 3))
        self.assertEqual(self.Config().max_reads, 300)
        with self.assertRaises(ValueError):
            self.Config(max_reads=None)
        with self.assertRaises(ValueError):
            self.Config(target='m6A')
        with self.assertRaises(ValueError):
            self.Config(sequence_embedding=15, attention_heads=2)
        with self.assertRaises(ValueError):
            self.Config(read_hidden_dim=15, read_attention_heads=2)

    def test_one_read_convolution_has_three_base_receptive_field(self):
        tf, np = self.tf, self.np
        encoder = self.model().get_layer('shared_read_encoder').encoder
        # Positive weights/inputs keep ReLUs active: test actual dependency,
        # independently of the randomly initialized classifier and attention.
        for layer in encoder.layers:
            if isinstance(layer, tf.keras.layers.Conv1D):
                layer.kernel.assign(tf.ones_like(layer.kernel))
                layer.bias.assign(tf.zeros_like(layer.bias))
        self.assertEqual(sum(isinstance(layer, tf.keras.layers.Conv1D)
                             for layer in encoder.layers), 1)
        spatial = tf.keras.Model(encoder.inputs, encoder.get_layer('read_local3').output)
        features = tf.Variable(tf.ones((1, 19, self.Config().read_channels)))
        with tf.GradientTape() as tape:
            activations = spatial(features)
            center = tf.reduce_sum(activations[:, 8, :])
        gradient = tape.gradient(center, features).numpy()
        affected = np.flatnonzero(np.any(gradient[0] != 0, axis=1))
        np.testing.assert_array_equal(affected, np.arange(8, 11))
        self.assertEqual(activations.shape, (1, 17, 16))

    def test_read_transformer_relates_all_window_positions_within_each_read(self):
        tf, np = self.tf, self.np
        encoder = self.model().get_layer('shared_read_encoder').encoder
        attention = encoder.get_layer('read_self_attention')
        self.assertEqual(attention.get_config()['num_heads'], 2)
        self.assertEqual(attention.get_config()['key_dim'], 8)
        self.assertEqual(encoder.get_layer('read_position_embedding').length, 17)
        spatial = tf.keras.Model(encoder.inputs, encoder.get_layer('read_ff_norm').output)
        features = tf.Variable(np.random.default_rng(14).random(
            (2, 19, self.Config().read_channels), dtype=np.float32))
        with tf.GradientTape() as tape:
            activations = spatial(features, training=False)
            center = activations[0, 8, 0]
        gradient = tape.gradient(center, features).numpy()
        affected = np.flatnonzero(np.any(gradient[0] != 0, axis=1))
        np.testing.assert_array_equal(affected, np.arange(19))
        np.testing.assert_array_equal(gradient[1], 0)
        self.assertEqual(activations.shape, (2, 17, 16))

    def test_reference_sequence_and_read_encoder_are_independent_branches(self):
        model, inputs = self.model(), self.inputs()
        read_layer = model.get_layer('shared_read_encoder')
        self.assertEqual(len(read_layer.encoder.inputs), 1)
        self.assertEqual(len(read_layer.input), 2)
        read_model = self.tf.keras.Model(model.inputs, read_layer.output)
        expected = read_model(inputs, training=False).numpy()
        changed = dict(inputs, sequence_tokens=(inputs['sequence_tokens'] + 1) % 65)
        self.np.testing.assert_array_equal(read_model(changed, training=False), expected)
        changed = dict(inputs, sequence_repeat_mask=~inputs['sequence_repeat_mask'])
        self.np.testing.assert_array_equal(read_model(changed, training=False), expected)

    def test_binary_repeat_context_changes_only_reference_branch_and_uses_window_ends(self):
        tf, np = self.tf, self.np
        model, inputs = self.model(), self.inputs()
        repeat_layer = model.get_layer('trimer_repeat_embedding')
        self.assertEqual(repeat_layer.count_params(), 48)
        self.assertEqual(model.count_params(), 23731)
        self.assertEqual(model.get_layer('shared_read_encoder').encoder.count_params(), 5488)
        projection = np.arange(48, dtype=np.float32).reshape(3, 16)
        repeat_layer.projection.set_weights([projection])
        mask = np.zeros((1, 19), bool)
        mask[0, [0, 18]] = True
        encoded = repeat_layer(mask).numpy()
        self.assertEqual(encoded.shape, (1, 17, 16))
        np.testing.assert_array_equal(encoded[0, 0], projection[0])
        np.testing.assert_array_equal(encoded[0, 16], projection[2])
        np.testing.assert_array_equal(encoded[0, 1:16], 0)
        # Left/middle/right repeat flags retain their order rather than being
        # replaced by the repeat fraction within a trimer.
        mask[:] = False
        mask[0, 9] = True
        encoded = repeat_layer(mask).numpy()
        np.testing.assert_array_equal(encoded[0, 7:10], projection[::-1])
        # A repeat change must reach the reference branch through the learned
        # projection, while the shared read encoder stays independent.
        sequence_model = tf.keras.Model(model.inputs, model.get_layer('sequence_vector').output)
        inputs['sequence_repeat_mask'][:] = False
        before = sequence_model(inputs, training=False).numpy()
        changed = dict(inputs, sequence_repeat_mask=np.ones((2, 19), bool))
        after = sequence_model(changed, training=False).numpy()
        self.assertFalse(np.allclose(before, after))

    def test_repeat_projection_starts_neutral_and_can_learn(self):
        tf, np = self.tf, self.np
        model, inputs = self.model(), self.inputs()
        projection = model.get_layer('trimer_repeat_embedding').projection
        np.testing.assert_array_equal(projection.kernel.numpy(), 0)
        inputs['sequence_repeat_mask'][:] = False
        before = model(inputs, training=False).numpy()
        changed = dict(inputs, sequence_repeat_mask=np.ones((2, 19), bool))
        np.testing.assert_array_equal(model(changed, training=False).numpy(), before)
        with tf.GradientTape() as tape:
            prediction = model(changed, training=False)
            loss = -tf.math.log(prediction[0, 0])
        gradient = tape.gradient(loss, projection.kernel).numpy()
        self.assertTrue(np.isfinite(gradient).all())
        self.assertGreater(float(np.linalg.norm(gradient)), 0)


class ModelInputsTest(unittest.TestCase):
    def test_compact_channels_and_capped_splice_distances(self):
        import math
        from utils.nn.inputs import CHANNEL_NAMES, encode_splice_distances
        self.assertEqual(CHANNEL_NAMES, tuple('ml_' + name for name in
            ('m6A', 'inosine', 'm5C', 'pseU', '2OmeA', '2OmeC', '2OmeG', '2OmeU'))
            + ('inverse_quality', 'log1p_insertion')
            + ('base_match',) + tuple('base_' + base for base in 'ACGTN')
            + ('splice_before_log_distance', 'splice_after_log_distance'))
        self.assertEqual(encode_splice_distances(dict(
            splice_distance_before=None, splice_distance_after=10000)), [1, 1])
        self.assertEqual(encode_splice_distances(dict(
            splice_distance_before=0, splice_distance_after=1000)), [0, 1])
        observed = encode_splice_distances(dict(
            splice_distance_before=9, splice_distance_after=99))
        self.assertAlmostEqual(observed[0], math.log1p(9) / math.log1p(1000))
        self.assertAlmostEqual(observed[1], math.log1p(99) / math.log1p(1000))

    def test_read_base_identity_distinguishes_matches_and_mismatch_bases(self):
        from utils.nn.inputs import CHANNEL_NAMES, encode_read_row
        bases = list('ACGTN') + [None] + list('ACGTN') + ['A']
        row = dict(base=bases, ml_byte=[[None] * 8 for _ in bases],
                   base_quality=[None] * len(bases), mismatch=[0] * 5 + [None] + [1] * 5 + [None],
                   alignment_state=['base'] * 5 + ['uncovered'] + ['base'] * 6,
                   insertion_length_after=[0] * len(bases), mapq=255,
                   splice_distance_before=None, splice_distance_after=None)
        values = [dict(zip(CHANNEL_NAMES, values)) for values in encode_read_row(row)]
        categories = ('match', 'A', 'C', 'G', 'T', 'N')
        expected = ['match'] * 4 + ['N', None] + list('ACGTN') + ['A']
        for index, category in enumerate(expected):
            self.assertEqual([values[index]['base_' + key] for key in categories],
                             [float(key == category) for key in categories])

    def test_trimers_and_center_alignment(self):
        from utils.nn.inputs import trimer_tokens
        self.assertEqual(trimer_tokens('AACGTN'), [1, 6, 27, 64])
        sequence = 'C' * 9 + 'A' + 'G' * 9
        tokens = trimer_tokens(sequence)
        self.assertEqual(len(tokens), 17)
        self.assertEqual(tokens[8], trimer_tokens('CAG')[0])

    def test_ml_zero_vs_missing_and_quality_zero(self):
        from utils.nn.inputs import CHANNEL_NAMES, encode_read_row
        row = dict(base=['A', None], mm_skip_modes=['?', '.', None, None, None, None, None, None],
                   ml_byte=[[0, None, None, None, None, None, None, None], [None] * 8],
                   modification_observed=[[True] + [False] * 7, [False] * 8],
                   modification_applicable=[[True, True] + [False] * 6, [False] * 8],
                   base_quality=[0, None], mismatch=[0, None],
                   alignment_state=['base', 'deletion'], insertion_length_after=[0, 2], mapq=255,
                   splice_distance_before=0, splice_distance_after=None)
        values = encode_read_row(row)
        first, missing = [dict(zip(CHANNEL_NAMES, x)) for x in values]
        self.assertEqual(len(CHANNEL_NAMES), 18)
        self.assertEqual(first['splice_before_log_distance'], 0)
        self.assertEqual(first['splice_after_log_distance'], 1)
        self.assertEqual(first['ml_m6A'], 0)
        self.assertEqual(missing['ml_m6A'], 0)
        self.assertFalse(any(name.startswith(('ml_observed_', 'ml_applicable_', 'mm_declared_', 'mm_skip_unknown_')) for name in CHANNEL_NAMES))
        for key in ('mm_skip_modes', 'modification_observed', 'modification_applicable'):
            row.pop(key)
        self.assertEqual(encode_read_row(row), values)
        self.assertEqual(first['inverse_quality'], 1)
        self.assertEqual(missing['inverse_quality'], 0)
        self.assertFalse(any(name in CHANNEL_NAMES for name in
                             ('quality_observed', 'mismatch', 'mismatch_observed',
                              'mapq', 'mapq_observed', 'splice_before_observed',
                              'splice_after_observed')))


if __name__ == '__main__':
    unittest.main()
