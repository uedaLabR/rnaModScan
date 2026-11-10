"""Check the installed TensorFlow GPU runtime using a tiny convolution/gradient."""
import argparse
import json
from pathlib import Path
import sys
import tempfile


def check_site_model(tf):
    """Check the actual classifier's train step and saved-model round trip."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from utils.nn.inputs import CHANNEL_NAMES
    from utils.nn.keras_site import ModelConfig, build_site_model, compile_site_model

    tf.keras.utils.set_random_seed(42)
    # Keras input adapters contain CPU-only tf.data operations. Allow these,
    # while still requiring the classifier output itself to reside on the GPU.
    tf.config.set_soft_device_placement(True)
    with tf.device('/GPU:0'):
        model = compile_site_model(build_site_model(ModelConfig(max_reads=3)))
        inputs = dict(sequence_repeat_mask=np.tile(np.arange(19) % 2 == 0, (2, 1)),
                      sequence_tokens=np.zeros((2, 17), dtype=np.int32),
                      read_features=np.ones((2, 3, 19, len(CHANNEL_NAMES)), dtype=np.float32),
                      read_mask=np.array([[True, False, False], [True, True, True]]))
        metrics = model.train_on_batch(inputs, np.array([0, 1], dtype=np.int32), return_dict=True)
        expected = model(inputs, training=False)
        if 'GPU:' not in expected.device.upper() or not np.isfinite(metrics['loss']):
            raise ValueError('Classifier train step did not finish on the GPU with finite loss')
        with tempfile.TemporaryDirectory(prefix='rnamodscan_gpu_check_') as directory:
            path = Path(directory) / 'model.keras'
            model.save(path)
            restored = tf.keras.models.load_model(path)
            np.testing.assert_allclose(restored(inputs, training=False).numpy(), expected.numpy(), atol=1e-6)
    return dict(status='complete', tensor_device=expected.device, loss=float(metrics['loss']),
                model_save_load='passed', input_names=list(inputs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='store_true', help='Also train/save/reload a tiny actual classifier')
    arguments = parser.parse_args()
    import tensorflow as tf
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from utils.nn.inputs import CHANNEL_NAMES
    report = dict(python=sys.version.split()[0], tensorflow=tf.__version__,
                  build=tf.sysconfig.get_build_info())
    devices = tf.config.list_physical_devices('GPU')
    report['gpu_devices'] = [device.name for device in devices]
    if not devices:
        report.update(status='failed', error='TensorFlow cannot use a GPU in this environment')
        print(json.dumps(report, default=str, indent=2), flush=True)
        return 2
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.set_soft_device_placement(False)
    try:
        with tf.device('/GPU:0'):
            layer = tf.keras.layers.Conv1D(2, 3)
            x = tf.ones((1, 19, len(CHANNEL_NAMES)))
            with tf.GradientTape() as tape:
                y = layer(x)
                loss = tf.reduce_sum(y)
            gradients = tape.gradient(loss, layer.trainable_variables)
            report.update(status='complete', tensor_device=y.device, loss=float(loss.numpy()),
                          finite_gradients=all(bool(tf.reduce_all(tf.math.is_finite(g)).numpy())
                                               for g in gradients if g is not None))
        if not report['finite_gradients']:
            raise ValueError('GPU gradients are not finite')
        if arguments.model:
            report['classifier'] = check_site_model(tf)
    except Exception as error:
        report.update(status='failed', error=str(error))
    print(json.dumps(report, default=str, indent=2), flush=True)
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
