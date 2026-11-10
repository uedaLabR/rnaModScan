"""Train on one cell only, choose epochs on its validation set, then refit.

HeLa remains a separate analysis dataset. This entry does not benchmark m6anet,
open BAM files, copy raw Parquet, or use another cell for model selection.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def best_validation_epoch(history_path):
    """Select the minimum finite validation loss; Keras epochs are zero-based."""
    with Path(history_path).open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not math.isfinite(float(row['val_loss'])) for row in rows):
        raise ValueError('Validation history must contain finite validation losses')
    return int(min(rows, key=lambda row: float(row['val_loss']))['epoch']) + 1


def train_cell_model(index_path, datasets_root, samples, out, *, target='adenine',
                     training_cell='HEK293T', separate_cell='HeLa', flank=9,
                     epochs=20, patience=3, batch_sites=4,
                     seed=42, learning_rate=0.001, threshold=0.5,
                     max_reads=300, sampling_seed=42):
    """Use train/validation from one cell, with no training-time test set.

    The final bundle is refitted from scratch on that cell's train + validation
    observations. Validation metrics describe weak-label model selection only.
    """
    from training.cell_model_data import prepare_cell_model_data
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'Choose an unused cell-model output: {out}')
    if (epochs < 1 or patience < 0 or batch_sites < 1
            or not 0 <= threshold <= 1 or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('Invalid cell-model training settings')
    out.mkdir(parents=True)
    marker = out / 'INCOMPLETE.txt'
    marker.write_text('Cell-only model selection and final refit are not complete.\n', encoding='utf-8')
    progress = None
    try:
        data = prepare_cell_model_data(index_path, datasets_root, samples, out / 'data',
                                      training_cell=training_cell, separate_cell=separate_cell, flank=flank)
        from utils.io.artifacts import completed_run
        subset_run = completed_run(Path(data['index_path']).parent)
        if subset_run['target'] != target:
            raise ValueError('Cell-only input target differs from requested model')

        import tensorflow as tf
        from utils.nn.keras_site import MODEL_INPUT_SCHEMA, build_site_model, compile_site_model
        from training.model_data import training_inputs, balanced_class_weights
        from training.validation import predict_split
        from training.final_model import export_final_model
        from training.monitoring import TrainingProgress
        from utils.references.catalog import sha256

        tf.keras.utils.set_random_seed(seed)
        datasets, architecture = training_inputs(data['index_path'], data['split_paths'], target,
            batch_sites=batch_sites, max_reads=max_reads, sampling_seed=sampling_seed)
        training = out / 'train'
        validation = out / 'validation'
        training.mkdir()
        validation.mkdir()
        model = compile_site_model(build_site_model(architecture), learning_rate)
        progress = TrainingProgress(training)
        callbacks = [
            tf.keras.callbacks.CSVLogger(str(training / 'history.csv')),
            progress,
            tf.keras.callbacks.ModelCheckpoint(str(training / 'best.keras'),
                                              monitor='val_loss', save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        class_weights = balanced_class_weights(datasets['train'])
        print(f'{training_cell}: selecting epochs using this cell only', flush=True)
        model.fit(datasets['train'].tensorflow_dataset(shuffle=True, seed=seed),
                  validation_data=datasets['validation'].tensorflow_dataset(), epochs=epochs,
                  callbacks=callbacks, class_weight=class_weights, verbose=2)
        chosen_epochs = best_validation_epoch(training / 'history.csv')
        model = tf.keras.models.load_model(training / 'best.keras')
        metrics = predict_split(model, datasets['validation'], validation / 'predictions.tsv', threshold)
        metrics.update(interpretation='internal weak-label validation used for epoch selection; not independent test',
                       training_cell=training_cell, excluded_cell=separate_cell)
        (validation / 'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n', encoding='utf-8')
        selection = dict(status='complete', training_cell=training_cell, best_epoch=chosen_epochs,
                         class_weights=class_weights, history_sha256=sha256(training / 'history.csv'),
                         model_sha256=sha256(training / 'best.keras'),
                         validation_manifest_sha256=sha256(data['split_paths']['validation']),
                         validation_role='early stopping only; not a held-out test')
        (training / 'run.json').write_text(json.dumps(selection, indent=2)+'\n', encoding='utf-8')

        # Release the selection model/read caches before creating the final fit.
        del model, datasets, callbacks
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        datasets, architecture = training_inputs(data['index_path'], {'train': None}, target,
            batch_sites=batch_sites, max_reads=max_reads, sampling_seed=sampling_seed)
        final_data = datasets['train']  # This index contains the training cell only.
        allowed = set(data['training_scope']['sample_ids'])
        if any(row['sample_id'] not in allowed for row in final_data.rows):
            raise ValueError('A separate-cell observation reached the final refit')
        final = out / 'final_model'
        final.mkdir()
        final_marker = final / 'INCOMPLETE.txt'
        final_marker.write_text('Cell-only final refit is not complete.\n', encoding='utf-8')
        final_model = compile_site_model(build_site_model(architecture), learning_rate)
        final_weights = balanced_class_weights(final_data)
        progress = TrainingProgress(final)
        print(f'{training_cell}: refitting all {len(final_data.rows):,} observations for {chosen_epochs} epochs', flush=True)
        final_model.fit(final_data.tensorflow_dataset(shuffle=True, seed=seed), epochs=chosen_epochs,
                        class_weight=final_weights,
                        callbacks=[tf.keras.callbacks.CSVLogger(str(final / 'history.csv')),
                                   progress, tf.keras.callbacks.TerminateOnNaN()], verbose=2)
        with (final / 'history.csv').open(encoding='utf-8', newline='') as handle:
            history = list(csv.DictReader(handle))
        if len(history) != chosen_epochs or any(not math.isfinite(float(row['loss'])) for row in history):
            raise ValueError('Cell-only final refit did not finish with finite loss')
        export_final_model(final_model, final_data, architecture, data['index_path'], final,
            epochs=chosen_epochs, class_weights=final_weights, seed=seed, learning_rate=learning_rate,
            threshold=threshold, provenance=dict(training_scope=data['training_scope'],
                epoch_policy='best validation epoch from training cell only',
                epoch_selection=selection, source_cell_datasets=data['source_cell_datasets'],
                evaluation='no held-out evaluation after refit; separate cell was not used'))
        final_marker.unlink()
        report = dict(status='complete', target=target, feature_set=MODEL_INPUT_SCHEMA,
                      training_cell=training_cell, separate_cell=separate_cell,
                      training_scope=data['training_scope'], sizes=data['sizes'], best_epoch=chosen_epochs,
                      model_directory=str(final), selection_directory=str(training),
                      data_manifest_sha256=sha256(out / 'data/run.json'),
                      policies=dict(separate_cell='not used for fitting, early stopping, thresholds or training-time test',
                                    raw_features='reuse existing Parquet and read offsets; no copying',
                                    final_refit='all selected observations of the training cell only'))
        (out / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        marker.unlink()
        return report
    except BaseException as error:
        if progress is not None:
            progress.mark_failed(error)
        (out / 'FAILED.txt').write_text(str(error), encoding='utf-8')
        raise


def main(argv=None):
    from utils.config import load_config, index_paths
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'training/config.yaml')
    parser.add_argument('--dry-run', action='store_true', help='Show paths without preparing data or training')
    args = parser.parse_args(argv)
    config = load_config(args.config, 'train')
    settings = config['cell_dataset']
    if config['nn']['task'] != 'adenine_three_class' or settings['training_cell'] != 'HEK293T':
        raise ValueError('This entry requires the joint adenine model with HEK293T as training_cell')
    target = 'adenine'
    inputs = dict(index_path=index_paths(config)[target], datasets_root=config['output_dir'] / 'cell_datasets',
                  samples=config['samples'], out=config['output_dir'] / 'cell_models' / 'HEK293T' / target)
    if args.dry_run:
        print(json.dumps(dict(stage='HEK293T-only selection and final refit', **inputs,
                              no_training_started=True), ensure_ascii=False, indent=2, default=str))
        return
    nn = config['nn']
    result = train_cell_model(**inputs, target=target, training_cell='HEK293T',
        separate_cell=settings['separate_cell'], flank=config['features']['flank'],
        epochs=nn['epochs'], patience=nn['patience'], batch_sites=nn['batch_sites'],
        max_reads=nn['max_reads'], sampling_seed=nn['sampling_seed'], seed=settings['seed'], learning_rate=nn['learning_rate'],
        threshold=nn['target_settings'][target]['threshold'])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
