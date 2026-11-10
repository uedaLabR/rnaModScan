"""Refit all labeled sites after CV. This model is for deployment, not evaluation."""
import csv
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path

from utils.nn.inputs import CHANNEL_NAMES, MODIFICATIONS
from utils.adenine import CLASSES
from utils.nn.keras_site import MODEL_INPUT_SCHEMA, build_site_model, compile_site_model
from utils.references.catalog import sha256
from utils.features.index import completed_run
from training.model_data import checked_training_index, training_inputs, balanced_class_weights


def cv_training_choice(cv_models, index_path, target):
    """Use only CV validation loss to fix epochs and architecture before refit."""
    best_epochs, sources, architecture = [], [], None
    index_sha256 = sha256(index_path)
    for fold in range(5):
        directory = Path(cv_models) / f'fold_{fold}'
        run = completed_run(directory)
        if run['target'] != target or run['index_sha256'] != index_sha256:
            raise ValueError('CV models must use the current labeled index')
        if run.get('feature_set') != MODEL_INPUT_SCHEMA:
            raise ValueError('Regenerate CV models with the current sequence/read inputs')
        settings = dict(run['model'])
        if architecture is not None and settings != architecture:
            raise ValueError('CV model architectures differ')
        architecture = settings
        history_path = directory / 'train' / 'history.csv'
        with history_path.open(encoding='utf-8', newline='') as handle:
            history = list(csv.DictReader(handle))
        if not history or any(not math.isfinite(float(row['val_loss'])) for row in history):
            raise ValueError('CV history needs finite validation loss')
        best = min(history, key=lambda row: float(row['val_loss']))
        best_epochs.append(int(best['epoch']) + 1)
        sources.append(dict(path=str(directory.resolve()), best_epoch=best_epochs[-1],
                            history_sha256=sha256(history_path), run_sha256=sha256(directory / 'run.json')))
    return math.ceil(statistics.median(best_epochs)), architecture, sources


def export_final_model(model, dataset, config, index_path, out, *, epochs, class_weights,
                       seed, learning_rate, threshold=0.5, provenance=None):
    """Save a standard deployment bundle for an already fitted model.

    The caller owns run markers and the training history. Additional provenance
    may describe CV or cell-specific selection, but cannot replace core model,
    input-schema or checksum fields.
    """
    import tensorflow as tf

    index_path, out = Path(index_path).resolve(), Path(out)
    if dataset.target != config.target:
        raise ValueError('Dataset and exported model target differ')
    if not isinstance(epochs, int) or epochs < 1 or not 0 <= threshold <= 1:
        raise ValueError('Invalid exported epochs or threshold')
    report = dict(status='complete', role='final_all_labeled_sites', target=config.target,
                  model=asdict(config), model_file='model.keras', model_sha256=None,
                  feature_set=MODEL_INPUT_SCHEMA, threshold=threshold,
                  input_schema=dict(feature_set=MODEL_INPUT_SCHEMA,
                      missing_ml='zero', window=config.window, modification_channels=list(MODIFICATIONS),
                      read_channel_names=list(CHANNEL_NAMES),
                      splice_distance='same_read_CIGAR_N_query_bases_adjacent_exon_zero',
                      splice_distance_cap=1000, splice_missing_distance=1000,
                      splice_distance_transform='log1p(min(d,1000))/log1p(1000)',
                      sequence_repeat_mask='softmasked_FASTA_lowercase_acgt', repeat_mask_window=config.window,
                      ml_transform='ML/255', quality_transform='1/(Q+1)',
                      window_mode='genomic', orientation='RNA_5to3', max_reads=config.max_reads, sampling_seed=config.sampling_seed),
                  index_path=str(index_path), index_sha256=sha256(index_path),
                  feature_sources=completed_run(index_path.parent)['feature_sources'],
                  training_rows=len(dataset.rows), epochs=epochs, class_weights=class_weights,
                  seed=seed, learning_rate=learning_rate, tensorflow_version=tf.__version__)
    if config.target == 'adenine':
        report['input_schema']['class_names'] = list(CLASSES)
    additional = dict(provenance or {})
    protected = sorted(set(report) & set(additional))
    if protected:
        raise ValueError('Export provenance cannot replace core fields: ' + ', '.join(protected))
    for name in ('model.keras', 'training_ids.tsv', 'run.json'):
        if (out / name).exists():
            raise FileExistsError(out / name)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / 'model.keras')
    with (out / 'training_ids.tsv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['row_id', 'sample_id', 'label'])
        writer.writerows((row['row_id'], row['sample_id'], row['label']) for row in dataset.rows)
    report['model_sha256'] = sha256(out / 'model.keras')
    report.update(additional)
    (out / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def train_final_model(index_path, cv_models, out, target, *, batch_sites=16,
                      epochs=None, seed=42, learning_rate=1e-3, threshold=0.5,
                      max_reads=300, sampling_seed=42):
    """All available final positive/negative rows, including previous CV test rows.

    No validation/test split is used during refitting and no evaluation score is
    produced. Epochs default to the median CV best-validation epoch, rounded up.
    """
    import tensorflow as tf
    index_path, out = Path(index_path).resolve(), Path(out)
    if out.exists():
        raise FileExistsError(out)
    checked_training_index(index_path, target)
    chosen_epochs, architecture, cv_sources = cv_training_choice(cv_models, index_path, target)
    explicit_epochs = epochs is not None
    epochs = chosen_epochs if epochs is None else epochs
    if not isinstance(epochs, int) or epochs < 1 or not 0 <= threshold <= 1:
        raise ValueError('Invalid final epochs or threshold')
    tf.keras.utils.set_random_seed(seed)
    datasets, config = training_inputs(index_path, {'train': None}, target,
        batch_sites=batch_sites, model_settings=architecture,
        max_reads=max_reads, sampling_seed=sampling_seed)
    model = compile_site_model(build_site_model(config), learning_rate)
    out.mkdir(parents=True)
    (out / 'INCOMPLETE.txt').write_text('Final model not complete.', encoding='utf-8')
    try:
        class_weights = balanced_class_weights(datasets['train'])
        model.fit(datasets['train'].tensorflow_dataset(shuffle=True, seed=seed), epochs=epochs, class_weight=class_weights,
                  callbacks=[tf.keras.callbacks.CSVLogger(str(out / 'history.csv')),
                             tf.keras.callbacks.TerminateOnNaN()])
        with (out / 'history.csv').open() as handle:
            history = list(csv.DictReader(handle))
        if len(history) != epochs or any(not math.isfinite(float(row['loss'])) for row in history):
            raise ValueError('Final fitting did not finish with finite loss')
        report = export_final_model(model, datasets['train'], config, index_path, out,
            epochs=epochs, class_weights=class_weights, seed=seed, learning_rate=learning_rate,
            threshold=threshold,
            provenance=dict(epoch_policy='explicit setting' if explicit_epochs else 'median CV best validation epoch (ceil)',
                            cv_sources=cv_sources,
                            evaluation='no held-out evaluation; use previously saved CV results'))
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as exc:
        (out / 'FAILED.txt').write_text(str(exc), encoding='utf-8')
        raise
