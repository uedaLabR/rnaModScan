"""Common raw-feature checks and sequence/read inputs for CV/final fitting."""
from pathlib import Path
from collections import Counter

from utils.nn.keras_site import ModelConfig
from utils.references.catalog import sha256
from utils.features.index import completed_run
from utils.features.site_dataset import SiteBagBatches
from utils.nn.inputs import label_values


def balanced_class_weights(dataset):
    if dataset.target != 'adenine':
        return None
    counts = Counter(int(row['label']) for row in dataset.rows)
    if set(counts) != {0, 1, 2}:
        raise ValueError('All three classes are required for joint model fitting')
    return {label: len(dataset.rows) / (3 * count) for label, count in counts.items()}


def checked_training_index(index_path, target):
    index_path = Path(index_path)
    run = completed_run(index_path.parent)
    if run.get('schema') != 'site_bag_index_v3':
        raise ValueError('Re-extract site Parquet with saved sequence repeat masks')
    if run['target'] != target or not run.get('require_labels', True):
        raise ValueError('Use a labeled feature index for the target modification')
    if run['counts'].get('not_extracted_sites', 0):
        raise ValueError('Extract all labeled sites before training')
    for source in run['feature_sources']:
        if sha256(Path(source['path']) / 'run.json') != source['run_sha256']:
            raise ValueError('Feature run changed after building index')
    return run


def training_inputs(index_path, split_paths, target, *, batch_sites=16, model_settings=None,
                    max_reads=300, sampling_seed=42):
    """Read all four NN inputs directly from the common per-site Parquet."""
    checked_training_index(index_path, target)
    datasets = {split: SiteBagBatches(index_path, path, batch_sites=batch_sites,
                max_reads=max_reads, sampling_seed=sampling_seed) for split, path in split_paths.items()}
    if {row['label'] for row in datasets['train'].rows} != label_values(target):
        raise ValueError('Training data must contain all target classes')
    windows = {len(row['reference_sequence']) for data in datasets.values() for row in data.rows}
    if len(windows) != 1:
        raise ValueError('Feature windows must be identical')
    settings = dict(model_settings or {})
    settings.update(target=target, window=windows.pop(), max_reads=max_reads, sampling_seed=sampling_seed)
    return datasets, ModelConfig(**settings)
