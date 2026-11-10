"""Configure the TensorFlow inference process before loading a model."""
import os


def validate_runtime(device, cpu_threads):
    if device not in ('auto', 'cpu', 'gpu'):
        raise ValueError('nn.device must be auto, cpu or gpu')
    if cpu_threads is not None and (type(cpu_threads) is not int or cpu_threads < 1):
        raise ValueError('nn.cpu_threads must be a positive integer or null')


def configure_inference(device='auto', cpu_threads=None):
    """CPU hides GPUs; auto uses an available GPU and otherwise falls back to CPU."""
    validate_runtime(device, cpu_threads)
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    try:
        if device == 'cpu':
            tf.config.set_visible_devices([], 'GPU')
        if cpu_threads is not None:
            tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
            tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError as error:
        raise RuntimeError('Set the inference device/threads in a fresh process before TensorFlow initializes') from error
    gpus = tf.config.list_logical_devices('GPU')
    if device == 'gpu' and not gpus:
        raise RuntimeError('GPU inference requested but TensorFlow has no usable GPU; use --device cpu or auto')
    return dict(requested_device=device, device='gpu' if gpus else 'cpu',
                cpu_threads=cpu_threads, gpu_devices=[gpu.name for gpu in gpus],
                tensorflow_version=tf.__version__)
