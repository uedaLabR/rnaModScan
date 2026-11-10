"""Checked final-model metadata, shared by CPU preparation and GPU inference."""
from pathlib import Path

from utils.nn.inputs import CHANNEL_NAMES, MODIFICATIONS
from utils.adenine import CLASSES
from utils.io.artifacts import completed_run
from utils.references.catalog import sha256


def validate_input_schema(run):
    """Reject final bundles with a different feature encoding or channel order."""
    if run.get('role') != 'final_all_labeled_sites':
        raise ValueError('Use the final all-labeled-sites model, not a CV fold model')
    schema, config = run['input_schema'], run['model']
    if run['target'] != 'adenine' or config['target'] != 'adenine':
        raise ValueError('Use the joint adenine three-class model')
    if type(config.get('max_reads')) is not int or config['max_reads'] < 1:
        raise ValueError('Model must declare a positive maximum read depth')
    expected = dict(feature_set='sequence_reads_repeat18_v1', missing_ml='zero', window=config['window'], modification_channels=list(MODIFICATIONS),
                    read_channel_names=list(CHANNEL_NAMES),
                    splice_distance='same_read_CIGAR_N_query_bases_adjacent_exon_zero',
                    splice_distance_cap=1000, splice_missing_distance=1000,
                    splice_distance_transform='log1p(min(d,1000))/log1p(1000)',
                    sequence_repeat_mask='softmasked_FASTA_lowercase_acgt', repeat_mask_window=config['window'],
                    ml_transform='ML/255',
                    quality_transform='1/(Q+1)', window_mode='genomic', orientation='RNA_5to3')
    for key, value in expected.items():
        if schema.get(key) != value:
            raise ValueError(f'Model input schema mismatch: {key}')
    for key in ('max_reads', 'sampling_seed'):
        if schema.get(key) != config.get(key):
            raise ValueError('Model sampling policy mismatch: ' + key)
    removed = ('repeat_encoding', 'repeat_slots', 'repeat_scope', 'position_channel_names', 'site_stat_names')
    if any(key in schema for key in removed):
        raise ValueError('Model input schema mismatch: obsolete site-level NN inputs')
    if schema.get('class_names') != list(CLASSES):
        raise ValueError('Joint model class order mismatch')
    if config['target'] != run['target'] or config['read_channels'] != len(CHANNEL_NAMES):
        raise ValueError('Model target or input-channel count disagrees with bundle')
    return schema



def load_model_bundle(directory):
    """Validate a sequence/read-only final model without importing TensorFlow."""
    directory = Path(directory).resolve()
    run = completed_run(directory)
    validate_input_schema(run)
    filename = run['model_file']
    if Path(filename).name != filename:
        raise ValueError('model_file must be a filename inside the final model directory')
    model_path = directory / filename
    if sha256(model_path) != run['model_sha256']:
        raise ValueError('Final model checksum mismatch')
    return {}, run


def model_fingerprint(directory, run):
    """A cache is tied to the complete metadata and weights of one final model."""
    return dict(run_sha256=sha256(Path(directory) / 'run.json'),
                model_sha256=run['model_sha256'])
