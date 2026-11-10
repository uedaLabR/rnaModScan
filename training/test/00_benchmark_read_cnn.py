"""Time the fixed 300-read classifier. Every batch uses a single shared CNN call."""
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import tensorflow as tf
    from utils.nn.inputs import CHANNEL_NAMES
    from utils.nn.keras_site import ModelConfig, build_site_model, compile_site_model
    devices = tf.config.list_physical_devices('GPU')
    if not devices:
        raise RuntimeError('A GPU is required')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(42)
    rng = np.random.default_rng(42)
    batch_sites = 16
    inputs = dict(sequence_repeat_mask=rng.integers(0, 2, (batch_sites, 19), dtype=np.int8).astype(bool),
                  sequence_tokens=rng.integers(0, 64, (batch_sites, 17), dtype=np.int32),
                  read_features=rng.random((batch_sites, 300, 19, len(CHANNEL_NAMES)), dtype=np.float32),
                  read_mask=np.ones((batch_sites, 300), bool))
    labels = np.arange(batch_sites, dtype=np.int32) % 3
    model = compile_site_model(build_site_model(ModelConfig(target='adenine', max_reads=300)))
    model.train_on_batch(inputs, labels)
    start = time.perf_counter()
    for _ in range(10):
        metrics = model.train_on_batch(inputs, labels, return_dict=True)
    seconds = (time.perf_counter() - start) / 10
    print(json.dumps(dict(synthetic_only=True, tensorflow=tf.__version__, sites=batch_sites,
                         read_channels=len(CHANNEL_NAMES), parameters=model.count_params(),
                         reads_per_site=300, cnn_calls_per_batch=1, seconds_per_batch=seconds,
                         sites_per_second=batch_sites / seconds, finite_loss=bool(np.isfinite(metrics['loss']))), indent=2))


if __name__ == '__main__':
    main()
