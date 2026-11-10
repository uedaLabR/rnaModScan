"""Training heartbeats and learning curves, without using held-out test results."""
import csv
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time

from tensorflow import keras


METRIC_NOTE = ('Metrics compare operational weak labels. When class weights are used, '
               'training loss is weighted and validation loss is unweighted. '
               'This progress file covers model.fit, not held-out test evaluation.')


def scalar_metrics(logs, prefix=''):
    """Keep finite scalar metrics JSON-readable; preserve nonfinite values as null."""
    result = {}
    for name, value in (logs or {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        result[prefix + name] = number if math.isfinite(number) else None
    return result


def plot_learning_curves(history_path, out):
    """Save loss/accuracy panels from CSVLogger history; missing accuracy is allowed."""
    history_path, out = Path(history_path), Path(out)
    if not history_path.is_file():
        return {}
    with history_path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        return {}

    cache = Path(tempfile.gettempdir()) / 'rnamodscan_matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    import matplotlib
    matplotlib.use('Agg', force=True)
    from matplotlib import pyplot as plt

    def values(name):
        series = []
        for row in rows:
            try:
                value = float(row.get(name, ''))
            except (TypeError, ValueError):
                value = math.nan
            series.append(value if math.isfinite(value) else math.nan)
        return series

    epoch_values = values('epoch')
    epochs = [int(value) + 1 if math.isfinite(value) else index + 1
              for index, value in enumerate(epoch_values)]
    accuracy_names = ('accuracy', 'sparse_categorical_accuracy', 'categorical_accuracy',
                      'binary_accuracy', 'acc')
    accuracy = next((name for name in accuracy_names
                     if name in fields or 'val_' + name in fields), None)
    out.mkdir(parents=True, exist_ok=True)
    paths = {kind: str((out / ('learning_curves.' + kind)).resolve()) for kind in ('png', 'pdf', 'svg')}
    with plt.rc_context({'svg.fonttype': 'none', 'pdf.fonttype': 42}):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        try:
            for axis, metric, title in zip(axes, ('loss', accuracy), ('Loss', 'Accuracy')):
                for prefix, label in (('', 'Train'), ('val_', 'Validation')):
                    if metric is not None and prefix + metric in fields:
                        axis.plot(epochs, values(prefix + metric), marker='o', markersize=3, label=label)
                axis.set(title=title, xlabel='Epoch', ylabel=title)
                axis.grid(alpha=.25)
                if axis.lines:
                    axis.legend()
                else:
                    axis.text(.5, .5, title + ' not available', ha='center', va='center', transform=axis.transAxes)
                if metric is not None and metric != 'loss':
                    axis.set_ylim(0, 1.05)
            figure.text(.5, .025,
                        'Operational weak labels. With class weights, train loss is weighted; validation loss is unweighted.',
                        ha='center', fontsize=8)
            figure.tight_layout(rect=(0, .09, 1, 1))
            for kind, path in paths.items():
                figure.savefig(path, format=kind, dpi=150)
        finally:
            plt.close(figure)
    return paths


class TrainingProgress(keras.callbacks.Callback):
    """Write train/progress.json and epoch curves under the supplied train folder.

    Put CSVLogger before this callback. A small timer refreshes the heartbeat
    during long batches too. Call mark_failed from the fitting exception handler
    because Keras does not always invoke on_train_end after an exception.
    """
    def __init__(self, out, interval_seconds=60):
        super().__init__()
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError('Progress interval must be positive and finite')
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = interval_seconds
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._heartbeat_thread = None
        self._started_monotonic = time.monotonic()
        self._last_write = -math.inf
        self._state = dict(schema='training_progress_v1', status='initializing', phase='initializing',
                           started_unix=time.time(), epoch=0, epochs=None, completed_epochs=0,
                           batches=0, epoch_batches=0, batches_per_epoch=None,
                           validation_batches=0, epoch_validation_batches=0,
                           metrics={}, curves={}, metric_note=METRIC_NOTE)
        self._write_progress(force=True)

    def _write_progress(self, force=False):
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last_write < self.interval_seconds:
                return
            wall_time = time.time()
            self._state.update(updated_unix=wall_time, heartbeat_unix=wall_time,
                               elapsed_seconds=round(now - self._started_monotonic, 3))
            temporary = self.out / 'progress.json.tmp'
            temporary.write_text(json.dumps(self._state, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            temporary.replace(self.out / 'progress.json')
            self._last_write = now

    def _heartbeat(self):
        while not self._stop.wait(self.interval_seconds):
            self._write_progress(force=True)

    def _stop_heartbeat(self):
        self._stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1)

    def on_train_begin(self, logs=None):
        with self._lock:
            self._state.update(status='running', phase='training',
                               epochs=self.params.get('epochs'), batches_per_epoch=self.params.get('steps'))
        self._write_progress(force=True)
        self._stop.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat, name='training-heartbeat', daemon=True)
        self._heartbeat_thread.start()

    def on_epoch_begin(self, epoch, logs=None):
        with self._lock:
            self._state.update(epoch=int(epoch) + 1, epoch_batches=0, epoch_validation_batches=0,
                               phase='training', metrics={})
        self._write_progress(force=True)

    def on_train_batch_end(self, batch, logs=None):
        with self._lock:
            self._state['batches'] += 1
            self._state['epoch_batches'] += 1
            self._state['metrics'].update(scalar_metrics(logs))
        self._write_progress()

    def on_test_begin(self, logs=None):
        # During model.fit these hooks refer to validation, never the later CV test.
        with self._lock:
            self._state['phase'] = 'validation'
        self._write_progress(force=True)

    def on_test_batch_end(self, batch, logs=None):
        with self._lock:
            self._state['validation_batches'] += 1
            self._state['epoch_validation_batches'] += 1
            self._state['metrics'].update(scalar_metrics(logs, prefix='val_'))
        self._write_progress()

    def on_test_end(self, logs=None):
        with self._lock:
            self._state['phase'] = 'validation_complete'
            self._state['metrics'].update(scalar_metrics(logs, prefix='val_'))
        self._write_progress(force=True)

    def on_epoch_end(self, epoch, logs=None):
        with self._lock:
            self._state.update(completed_epochs=int(epoch) + 1, phase='epoch_complete')
            self._state['metrics'].update(scalar_metrics(logs))
        self._write_progress(force=True)
        try:
            paths = plot_learning_curves(self.out / 'history.csv', self.out)
            with self._lock:
                self._state['curves'] = paths
                self._state.pop('plot_error', None)
        except Exception as error:
            # Plotting must not discard a long-running fit; retain a visible error.
            with self._lock:
                self._state['plot_error'] = str(error)
        self._write_progress(force=True)

    def on_train_end(self, logs=None):
        self._stop_heartbeat()
        with self._lock:
            self._state.update(status='complete', phase='finished')
            self._state['metrics'].update(scalar_metrics(logs))
            total = self._state['epochs']
            self._state['stopped_early'] = total is not None and self._state['completed_epochs'] < total
        self._write_progress(force=True)

    def mark_failed(self, error):
        self._stop_heartbeat()
        with self._lock:
            self._state.update(status='failed', phase='failed', error=str(error))
        self._write_progress(force=True)
