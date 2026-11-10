"""Device selection fails clearly when an explicitly requested GPU is absent."""
import os
import types
import unittest
from unittest.mock import Mock, patch

from prediction.runtime import configure_inference


class PredictionRuntimeTest(unittest.TestCase):
    def tensorflow(self, gpus=()):
        config = types.SimpleNamespace(set_visible_devices=Mock(),
            list_logical_devices=Mock(return_value=list(gpus)),
            threading=types.SimpleNamespace(set_intra_op_parallelism_threads=Mock(),
                                            set_inter_op_parallelism_threads=Mock()))
        return types.SimpleNamespace(config=config, __version__='test')

    def test_auto_falls_back_to_cpu_but_explicit_gpu_fails(self):
        with patch.dict('sys.modules', tensorflow=self.tensorflow()):
            self.assertEqual(configure_inference()['device'], 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'no usable GPU'):
                configure_inference('gpu')

    def test_auto_uses_visible_gpu_without_hiding_it(self):
        tf = self.tensorflow([types.SimpleNamespace(name='/device:GPU:0')])
        with patch.dict('sys.modules', tensorflow=tf):
            report = configure_inference()
        self.assertEqual(report['device'], 'gpu')
        self.assertEqual(report['gpu_devices'], ['/device:GPU:0'])
        tf.config.set_visible_devices.assert_not_called()

    def test_cpu_hides_gpu_and_requires_early_configuration(self):
        tf = self.tensorflow()
        with patch.dict('sys.modules', tensorflow=tf), patch.dict(os.environ):
            report = configure_inference('cpu', 2)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '-1')
            self.assertEqual(report['device'], 'cpu')
            tf.config.set_visible_devices.assert_called_once_with([], 'GPU')
            tf.config.set_visible_devices.side_effect = RuntimeError('Already initialized')
            with self.assertRaisesRegex(RuntimeError, 'fresh process'):
                configure_inference('cpu')
