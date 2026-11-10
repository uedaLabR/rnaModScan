"""Train one modification/fold. Validation selects epochs; test is read afterwards."""
import json
from dataclasses import asdict
from pathlib import Path

from utils.nn.keras_site import MODEL_INPUT_SCHEMA, build_site_model, compile_site_model
from utils.references.catalog import sha256
from utils.features.index import completed_run
from training.model_data import checked_training_index, training_inputs, balanced_class_weights
from training.monitoring import TrainingProgress
from training.validation import predict_split, write_metrics_summary


def train_fold(index_path, cv_directory, out, target, *, batch_sites=16,
               epochs=20, patience=3, seed=42, learning_rate=1e-3, threshold=0.5,
               max_reads=300, sampling_seed=42):
    """Use final labels as operational targets; never claim independent validation."""
    import tensorflow as tf
    index_path, cv_directory, out = Path(index_path).resolve(), Path(cv_directory), Path(out)
    if not 0 <= threshold <= 1 or epochs < 1 or patience < 0:
        raise ValueError('Invalid training settings')
    if out.exists():
        raise FileExistsError(out)
    index_run = checked_training_index(index_path, target)
    cv_run = completed_run(cv_directory.parent)
    if index_run['target'] != target or cv_run.get('target', target) != target:
        raise ValueError('Target mismatch')
    for split in ('train', 'validation', 'test'):
        name = f'{cv_directory.name}/{split}/sites.tsv'
        if cv_run['split_sha256'].get(name) != sha256(cv_directory / split / 'sites.tsv'):
            raise ValueError('CV manifest changed after splitting; regenerate the splits')
    if not any(i['target'] == target and i['sha256'] == sha256(index_path) for i in cv_run['inputs']):
        raise ValueError('Index changed after creating CV; regenerate the splits')
    split_paths = {split: cv_directory / split / 'sites.tsv' for split in ('train', 'validation', 'test')}
    tf.keras.utils.set_random_seed(seed)
    datasets, config = training_inputs(index_path, split_paths, target,
        batch_sites=batch_sites,
        max_reads=max_reads, sampling_seed=sampling_seed)
    groups = {name: {r['group_id'] for r in data.rows} for name, data in datasets.items()}
    if any(groups[a] & groups[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise ValueError('Locus groups leak between train/validation/test')
    model = compile_site_model(build_site_model(config), learning_rate)
    out.mkdir(parents=True)
    (out / 'INCOMPLETE.txt').write_text('Training not complete.', encoding='utf-8')
    try:
        training_directory = out / 'train'
        validation_directory = out / 'validation'
        test_directory = out / 'test'
        for directory in (training_directory, validation_directory, test_directory):
            directory.mkdir()
        class_weights = balanced_class_weights(datasets['train'])
        settings = dict(model=asdict(config), feature_set=MODEL_INPUT_SCHEMA,
                        split_sites={name: len(data.rows) for name, data in datasets.items()},
                        batch_sites=batch_sites, epochs=epochs, patience=patience,
                        seed=seed, learning_rate=learning_rate, class_weights=class_weights,
                        tensorflow_version=tf.__version__)
        (training_directory / 'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
        print(json.dumps(settings, indent=2), flush=True)
        progress = TrainingProgress(training_directory)
        callbacks = [tf.keras.callbacks.CSVLogger(str(training_directory / 'history.csv')),
                     tf.keras.callbacks.ModelCheckpoint(str(training_directory / 'best.keras'), monitor='val_loss', save_best_only=True),
                     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                     progress]
        try:
            model.fit(datasets['train'].tensorflow_dataset(shuffle=True, seed=seed),
                      validation_data=datasets['validation'].tensorflow_dataset(), epochs=epochs,
                      callbacks=callbacks, class_weight=class_weights, verbose=2)
        except BaseException as exc:
            progress.mark_failed(exc)
            raise
        # Load the saved best epoch, then evaluate the held-out test exactly once.
        model = tf.keras.models.load_model(training_directory / 'best.keras')
        interpretation = 'metrics against operational weak labels; internal grouped CV'
        for split, directory in (('validation', validation_directory), ('test', test_directory)):
            print(f'Evaluating {split}: {len(datasets[split].rows)} sites', flush=True)
            metrics = predict_split(model, datasets[split], directory / 'predictions.tsv', threshold)
            report = dict(metrics, interpretation=interpretation)
            (directory / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            write_metrics_summary(report, directory)
            print(f'{split} metrics saved: {directory}', flush=True)
        report = dict(status='complete', target=target, model=asdict(config), feature_set=MODEL_INPUT_SCHEMA,
                      index_path=str(index_path), index_sha256=sha256(index_path),
                      feature_sources=index_run['feature_sources'],
                      split_files={split: dict(path=str(path.resolve()), sha256=sha256(path))
                                   for split, path in split_paths.items()},
                      output_directories=dict(train='train', validation='validation', test='test'),
                      batch_sites=batch_sites,
                      epochs=epochs, seed=seed, learning_rate=learning_rate, patience=patience,
                      class_weights=class_weights, threshold=threshold,
                      threshold_policy='argmax; threshold unused' if target == 'adenine' else 'fixed before test; not tuned on test',
                      tensorflow_version=tf.__version__, evaluation='internal CV, not independent external test')
        (out / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as exc:
        (out / 'FAILED.txt').write_text(str(exc), encoding='utf-8')
        raise
