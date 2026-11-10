"""Validate frozen adenine labels and their original bedMethyl provenance."""
import csv
import json
from collections import Counter
from pathlib import Path

from utils.adenine import CLASSES, site_key
from utils.config import enabled_samples
from utils.io.artifacts import completed_run, file_info
from utils.references.catalog import sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_REPORT = ROOT / 'paper/source_data/call_overlap_20261104/site_index_run.json'


def verify_source_report(config, source_report):
    """Check the saved labeling rules and original BED/log identities, without rescanning BED."""
    source = json.loads(Path(source_report).read_text(encoding='utf-8'))
    thresholds = config['candidates']
    expected = {key: thresholds[key] for key in ('min_depth', 'min_modified', 'min_af')}
    if (source.get('status') != 'complete' or source.get('schema') != 'site_lists_v2'
            or source.get('assembly') != 'GRCh38' or source.get('coordinates') != 'BED0'
            or source['thresholds'] != expected or source['ivt_min_modified'] != thresholds['ivt_min_modified']
            or source['ivt_af_ratio'] != thresholds['ivt_af_ratio']):
        raise ValueError('Saved site-list rules differ from the current configuration')
    inputs = {entry['sample_id']: entry for entry in source['inputs']}
    pair_cells = {}
    for pair in source['pairs']:
        for key in ('wt_sample', 'ivt_sample'):
            pair_cells[pair[key]] = pair['cell'], pair['reference_cell']
    for sample in config['samples']:
        entry = inputs.get(sample['sample_id'])
        if entry is None or entry['status'] != 'available':
            raise ValueError('All configured WT/IVT inputs must have been available during labeling')
        if pair_cells[sample['sample_id']] != (sample['cell'], sample['reference_cell']):
            raise ValueError('Sample/cell mapping differs from the saved site lists')
        if entry['input'] != file_info(sample['bedmethyl']):
            raise ValueError(f"Original bedMethyl changed: {sample['sample_id']}")
        if entry['pileup_log']['sha256'] != sha256(sample['pileup_log']):
            raise ValueError(f"Original pileup log changed: {sample['sample_id']}")
    return source


def verify_selected_sites(config, directory):
    """Validate the self-contained selected TSV before starting expensive extraction."""
    run = completed_run(directory)
    if (config['nn']['task'] != 'adenine_three_class' or set(config['targets']) != {'m6A', 'inosine'}
            or run.get('schema') != 'adenine_training_sites_v1' or run.get('class_names') != list(CLASSES)
            or run['seed'] != config['selection']['seed'] or run['window_flank'] != config['features']['flank']):
        raise ValueError('Use the current selected adenine dataset and matching training settings')
    path = directory / 'adenine/training_sites.tsv'
    if sha256(path) != run['selected_sha256']:
        raise ValueError('Selected site TSV changed after it was created')
    samples = {sample['sample_id']: sample for sample in enabled_samples(config)}
    sample_counts, class_counts = Counter(), Counter()
    previous = None
    with path.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            sample = samples.get(row['sample_id'])
            key = site_key(row)
            if (sample is None or row['modification'] != 'adenine' or row['label'] not in ('0', '1', '2')
                    or row['label_status'] != 'final' or (previous is not None and key <= previous)):
                raise ValueError('Invalid, duplicate or unknown selected training site')
            if (row['cell'], row['reference_cell'], row['condition']) != (
                    sample['cell'], sample['reference_cell'], sample['condition']):
                raise ValueError('Selected site sample metadata differs from the configuration')
            previous = key
            sample_counts[row['sample_id']] += 1
            class_counts[CLASSES[int(row['label'])]] += 1
    if sum(sample_counts.values()) != run['selected_sites'] or set(sample_counts) != set(samples):
        raise ValueError('Selected dataset site/sample counts disagree')
    if set(class_counts) != set(CLASSES):
        raise ValueError('Selected dataset must contain all three classes')
    return path, run, dict(sample_counts), dict(class_counts)
