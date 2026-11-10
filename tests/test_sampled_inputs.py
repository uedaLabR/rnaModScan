"""A single shared encoder handles all real reads and excludes batch padding."""
import unittest
from unittest.mock import patch

import numpy as np

from utils.nn.inputs import CHANNEL_NAMES


class SampledInputsTest(unittest.TestCase):
    def test_all_real_reads_in_batch_enter_one_cnn_call(self):
        from utils.nn.keras_site import ModelConfig, build_site_model
        model = build_site_model(ModelConfig(target='adenine', max_reads=300))
        inputs = dict(sequence_tokens=np.zeros((2, 17), np.int32),
                      sequence_repeat_mask=np.zeros((2, 19), bool),
                      read_features=np.zeros((2, 300, 19, len(CHANNEL_NAMES)), np.float32),
                      read_mask=np.arange(300)[None, :] < np.array([1, 299])[:, None])
        encoder = model.get_layer('shared_read_encoder').encoder
        with patch.object(encoder, 'call', wraps=encoder.call) as call:
            model(inputs, training=False)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.args[0].shape[0], 300)

    def test_model_predictions_attention_and_gradients_ignore_300_slot_padding(self):
        import tensorflow as tf
        from utils.nn.keras_site import ModelConfig, build_site_model, attention_model
        tf.keras.utils.set_random_seed(42)
        fixed = build_site_model(ModelConfig(target='adenine', dropout=0, max_reads=300))
        rng = np.random.default_rng(3)
        for depth in (1, 10, 299, 300):
            unpadded = build_site_model(ModelConfig(dropout=0, max_reads=depth))
            unpadded.set_weights(fixed.get_weights())
            real = rng.random((1, depth, 19, len(CHANNEL_NAMES)), dtype=np.float32)
            padded = np.full((1, 300, 19, len(CHANNEL_NAMES)), 1e6, np.float32)
            padded[:, :depth] = real
            sequence = np.zeros((1, 17), np.int32)
            repeat_mask = tf.constant(rng.integers(0, 2, (1, 19)).astype(bool))
            short = dict(sequence_tokens=tf.constant(sequence), sequence_repeat_mask=repeat_mask,
                         read_features=tf.Variable(real), read_mask=tf.ones((1, depth), tf.bool))
            mask = np.arange(300)[None, :] < depth
            full = dict(sequence_tokens=tf.constant(sequence), sequence_repeat_mask=repeat_mask,
                        read_features=tf.Variable(padded), read_mask=tf.constant(mask))
            def evaluate(model, inputs):
                with tf.GradientTape() as tape:
                    prediction = model(inputs, training=True)
                    loss = -tf.math.log(prediction[0, 0])
                gradients = tape.gradient(loss, [inputs['read_features']] + model.trainable_variables)
                return prediction.numpy(), [tf.convert_to_tensor(g).numpy() for g in gradients]
            prediction, grads = evaluate(unpadded, short)
            padded_prediction, padded_grads = evaluate(fixed, full)
            np.testing.assert_allclose(prediction, padded_prediction, atol=2e-6)
            np.testing.assert_allclose(grads[0], padded_grads[0][:, :depth], atol=2e-6)
            np.testing.assert_array_equal(padded_grads[0][:, depth:], 0)
            for expected, actual in zip(grads[1:], padded_grads[1:]):
                np.testing.assert_allclose(expected, actual, atol=5e-6, rtol=2e-5)
            weights = attention_model(fixed)(full).numpy()
            np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-6)
            np.testing.assert_array_equal(weights[:, depth:], 0)
