import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from xml.etree import ElementTree


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('tensorflow', 'matplotlib'))


@unittest.skipUnless(AVAILABLE, 'TensorFlow and Matplotlib required')
class TrainingMonitoringTest(unittest.TestCase):
    def read_progress(self, out):
        return json.loads((out / 'progress.json').read_text())

    def test_small_fit_records_training_validation_and_curves(self):
        import numpy as np
        import tensorflow as tf
        from training.monitoring import TrainingProgress

        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        inputs = np.asarray([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [2., 0.], [0., 2.]], dtype=np.float32)
        labels = np.asarray([0, 1, 2, 0, 2, 1], dtype=np.int32)
        train = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(2).with_options(options)
        validation = tf.data.Dataset.from_tensor_slices((inputs[:4], labels[:4])).batch(2).with_options(options)
        model = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(3, activation='softmax')])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            progress = TrainingProgress(out, interval_seconds=.1)
            self.assertEqual(self.read_progress(out)['status'], 'initializing')
            model.fit(train, validation_data=validation, epochs=2, verbose=0,
                      class_weight={0: 1., 1: 2., 2: 1.},
                      callbacks=[tf.keras.callbacks.CSVLogger(str(out / 'history.csv')), progress])
            saved = self.read_progress(out)
            self.assertEqual((saved['status'], saved['phase']), ('complete', 'finished'))
            self.assertEqual((saved['completed_epochs'], saved['batches'], saved['validation_batches']), (2, 6, 4))
            self.assertEqual((saved['epoch_batches'], saved['epoch_validation_batches']), (3, 2))
            self.assertTrue({'loss', 'accuracy', 'val_loss', 'val_accuracy'} <= set(saved['metrics']))
            self.assertFalse(progress._heartbeat_thread.is_alive())
            self.assertFalse(saved['stopped_early'])
            self.assertNotIn('plot_error', saved)
            for kind in ('png', 'pdf', 'svg'):
                self.assertGreater(Path(saved['curves'][kind]).stat().st_size, 0)
            root = ElementTree.parse(out / 'learning_curves.svg').getroot()
            ns = {'svg': 'http://www.w3.org/2000/svg'}
            self.assertTrue(root.findall('.//svg:text', ns))

    def test_curves_allow_missing_accuracy_and_ignore_test_columns(self):
        from training.monitoring import plot_learning_curves
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            history = out / 'history.csv'
            with history.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['epoch', 'loss', 'val_loss', 'test_loss'])
                writer.writerows([[0, 1.2, 1.3, 999999], [1, .8, '', 999999]])
            paths = plot_learning_curves(history, out)
            self.assertEqual(set(paths), {'png', 'pdf', 'svg'})
            text = (out / 'learning_curves.svg').read_text()
            self.assertIn('Accuracy not available', text)
            self.assertNotIn('test_loss', text)
            self.assertNotIn('999999', text)
            self.assertEqual(plot_learning_curves(out / 'missing.csv', out), {})

    def test_idle_heartbeat_and_failure_cleanup(self):
        from training.monitoring import TrainingProgress
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            callback = TrainingProgress(out, interval_seconds=.02)
            callback.set_params(dict(epochs=3, steps=4))
            callback.on_train_begin()
            try:
                initial = self.read_progress(out)['heartbeat_unix']
                for _ in range(100):
                    time.sleep(.01)
                    if self.read_progress(out)['heartbeat_unix'] > initial:
                        break
                else:
                    self.fail('Heartbeat did not advance while no batch callbacks ran')
                callback.on_test_begin()
                self.assertEqual(self.read_progress(out)['phase'], 'validation')
            finally:
                callback.mark_failed(RuntimeError('synthetic failure'))
            saved = self.read_progress(out)
            self.assertEqual((saved['status'], saved['phase']), ('failed', 'failed'))
            self.assertEqual(saved['error'], 'synthetic failure')
            self.assertFalse(callback._heartbeat_thread.is_alive())


if __name__ == '__main__':
    unittest.main()
