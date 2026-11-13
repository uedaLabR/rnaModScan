"""Read safe YAML settings; paths are relative to the configuration file."""
from pathlib import Path

TARGETS = ('m6A', 'm5C', 'inosine', 'pseU')
TRAIN_STAGES = ('quality', 'sites', 'summarize', 'select', 'features', 'annotation', 'split', 'cell_dataset', 'fit', 'final', 'inspect')
PREDICT_STAGES = ('quality', 'pileup', 'candidates', 'features', 'annotation', 'inputs', 'infer', 'filter', 'bam')
REFERENCE_KEYS = {'fasta', 'gtf', 'catalog'}
SAMPLE_KEYS = {'sample_id', 'cell', 'reference_cell', 'condition', 'bedmethyl', 'pileup_log', 'bam', 'enabled'}


def check_keys(values, allowed, label):
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f'Unknown {label} settings: {sorted(unknown)}')


def resolve_path(value, directory):
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError('Paths must be nonempty strings or null')
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = directory / path
    return path.resolve()


def read_sample(sample, directory):
    check_keys(sample, SAMPLE_KEYS, 'sample')
    result = dict(sample)
    sample_id = result.get('sample_id')
    if not sample_id or any(character.isspace() for character in sample_id):
        raise ValueError('Each sample needs a nonempty sample_id without whitespace')
    if result.get('condition') not in ('WT', 'IVT'):
        raise ValueError('Sample condition must be WT or IVT')
    if not result.get('cell') or not result.get('reference_cell'):
        raise ValueError('Set explicit cell and reference_cell; HEK293 is not inferred as HEK293T')
    result.setdefault('enabled', True)
    if not isinstance(result['enabled'], bool):
        raise ValueError('Sample enabled must be true or false')
    for key in ('bedmethyl', 'pileup_log', 'bam'):
        result[key] = resolve_path(result.get(key), directory)
    return result


def load_config(path, mode=None, *, genome=None):
    """Load settings without opening BAM, Parquet, GTF or TensorFlow."""
    path = Path(path).resolve()
    import yaml
    try:
        with path.open(encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise ValueError(f'Invalid YAML configuration: {path}: {error}') from error
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a YAML mapping')
    mode = mode or config.get('mode')
    if mode not in ('train', 'predict'):
        raise ValueError('Configuration mode must be train or predict')
    if genome is not None:
        if mode != 'predict':
            raise ValueError('Genome override is for prediction only')
        config['genome'] = genome
    common = {'schema_version', 'mode', 'output_dir', 'targets', 'references', 'features',
              'candidates', 'nn', 'quality'}
    allowed = common | ({'samples', 'cv', 'cell_dataset', 'selection'} if mode == 'train' else {
        'sample', 'model_root', 'model_directories', 'bam_threads', 'recalibrate_bam', 'filtering', 'modkit', 'genome'})
    check_keys(config, allowed, 'top-level')
    if config.get('schema_version', 1) != 1 or config.get('mode', mode) != mode:
        raise ValueError('Configuration schema_version or mode does not match the command')
    config['mode'] = mode
    config['config_path'] = path
    config['output_dir'] = resolve_path(config.get('output_dir'), path.parent)
    if config['output_dir'] is None:
        raise ValueError('Set output_dir')
    targets = config.get('targets', ['m6A', 'inosine'] if mode == 'train' else list(TARGETS))
    if not targets or len(targets) != len(set(targets)) or any(target not in TARGETS for target in targets):
        raise ValueError('targets must be distinct supported modifications')
    config['targets'] = targets

    references = config.setdefault('references', {})
    check_keys(references, REFERENCE_KEYS, 'references')
    for key in REFERENCE_KEYS:
        references[key] = resolve_path(references.get(key), path.parent)
    if mode == 'predict':
        config.setdefault('genome', 'hg38')
        if config['genome'] not in ('hg38', 'mm10', 'other'):
            raise ValueError('genome must be hg38, mm10 or other')
        if config['genome'] == 'other':
            references['catalog'] = None  # Never open DB/short-read data for other genomes.
        elif references['catalog'] is None:
            root = Path(__file__).resolve().parents[1] / 'data/references'
            name = 'hg38/catalog.json' if config['genome'] == 'hg38' else 'catalog.mm10.json'
            references['catalog'] = root / name
    features = config.setdefault('features', {})
    feature_keys = {'flank', 'min_mapq', 'sample_workers', 'bam_threads', 'progress_seconds',
                    'candidate_group_gap', 'read_batch_size', 'group_workers', 'group_batch_size',
                    'max_fetch_span'}
    check_keys(features, feature_keys, 'features')
    features.setdefault('flank', 9)
    features.setdefault('min_mapq', 0)
    for key, value in dict(sample_workers=1, bam_threads=1, progress_seconds=60,
                           group_workers=1, group_batch_size=32, max_fetch_span=10000000).items():
        features.setdefault(key, value)
    for key in ('sample_workers', 'bam_threads', 'group_workers', 'group_batch_size', 'max_fetch_span'):
        if not isinstance(features[key], int) or isinstance(features[key], bool) or features[key] < 1:
            raise ValueError(f'features.{key} must be a positive integer')
    if not isinstance(features['progress_seconds'], (int, float)) or features['progress_seconds'] <= 0:
        raise ValueError('features.progress_seconds must be positive')
    features.setdefault('candidate_group_gap', 10000)
    features.setdefault('read_batch_size', 256)
    gap = features['candidate_group_gap']
    if gap is not None and (not isinstance(gap, int) or isinstance(gap, bool) or gap < 1):
        raise ValueError('features.candidate_group_gap must be a positive integer or null')
    if features['group_workers'] > 1 and (features['sample_workers'] > 1 or gap is None):
        raise ValueError('Group parallelism requires features.sample_workers=1 and a candidate_group_gap')
    size = features['read_batch_size']
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError('features.read_batch_size must be a positive integer')
    candidates = config.setdefault('candidates', {})
    check_keys(candidates, {'min_depth', 'min_modified', 'min_af', 'ivt_min_modified', 'ivt_af_ratio'}, 'candidates')
    for key, value in dict(min_depth=10, min_modified=3, min_af=.1, ivt_min_modified=3, ivt_af_ratio=.6).items():
        candidates.setdefault(key, value)
    quality = config.setdefault('quality', {})
    check_keys(quality, {'percentile', 'sample_fraction', 'seed', 'only_aligned'}, 'quality')
    for key, value in dict(percentile=.1, sample_fraction=1., seed=42, only_aligned=True).items():
        quality.setdefault(key, value)
    nn = config.setdefault('nn', {})
    nn_keys = {'batch_sites', 'epochs', 'patience', 'learning_rate', 'folds',
               'target_settings', 'task', 'max_reads', 'sampling_seed'}
    if mode == 'predict':
        nn_keys.update(('device', 'cpu_threads'))
    check_keys(nn, nn_keys, 'nn')
    if mode == 'predict':
        from prediction.runtime import validate_runtime
        nn.setdefault('device', 'auto')
        nn.setdefault('cpu_threads', None)
        validate_runtime(nn['device'], nn['cpu_threads'])
    nn.setdefault('task', 'adenine_three_class')
    if nn['task'] != 'adenine_three_class':
        raise ValueError('nn.task must be adenine_three_class; binary models are no longer supported')
    adenine_targets = {'m6A', 'inosine'} & set(targets)
    if adenine_targets and adenine_targets != {'m6A', 'inosine'}:
        raise ValueError('Joint adenine classification needs both m6A and inosine targets')
    if mode == 'train' and not adenine_targets:
        raise ValueError('Neural training requires both m6A and inosine targets')
    defaults = dict(batch_sites=4, epochs=20, patience=3, learning_rate=.001,
                    folds=list(range(5)),
                    target_settings={})
    for key, value in defaults.items():
        nn.setdefault(key, value)
    nn.setdefault('max_reads', 300)
    nn.setdefault('sampling_seed', 42)
    if type(nn['max_reads']) is not int or nn['max_reads'] < 1:
        raise ValueError('nn.max_reads must be a positive integer')
    if type(nn['sampling_seed']) is not int or nn['sampling_seed'] < 0:
        raise ValueError('nn.sampling_seed must be nonnegative')
    if not nn['folds'] or len(set(nn['folds'])) != len(nn['folds']) or any(fold not in range(5) for fold in nn['folds']):
        raise ValueError('nn.folds must contain distinct fold numbers 0..4')
    check_keys(nn['target_settings'], {'adenine'}, 'nn.target_settings')
    for target in (['adenine'] if adenine_targets else []):
        settings = nn['target_settings'].setdefault(target, {})
        check_keys(settings, {'threshold', 'final_epochs'}, f'nn.target_settings.{target}')
        settings.setdefault('threshold', .5)
        settings.setdefault('final_epochs', None)

    if mode == 'train':
        selection = config.setdefault('selection', {})
        check_keys(selection, {'enabled', 'source_dir', 'seed'}, 'selection')
        selection.setdefault('enabled', False)
        selection.setdefault('seed', 42)
        selection['source_dir'] = resolve_path(selection.get('source_dir'), path.parent)
        if not isinstance(selection['enabled'], bool) or not isinstance(selection['seed'], int):
            raise ValueError('selection.enabled must be boolean and selection.seed integer')
        if selection['enabled'] and any(target not in ('m6A', 'inosine') for target in targets):
            raise ValueError('Selected neural training supports m6A and inosine; m5C/psi use known sites only')
        config['samples'] = [read_sample(sample, path.parent) for sample in config.get('samples', [])]
        ids = [sample['sample_id'] for sample in config['samples']]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Training needs samples with distinct sample_id values')
        cv = config.setdefault('cv', {})
        check_keys(cv, {'seed', 'validation_fraction', 'require_all_samples'}, 'cv')
        for key, value in dict(seed=42, validation_fraction=.1, require_all_samples=True).items():
            cv.setdefault(key, value)
        cell_dataset = config.setdefault('cell_dataset', {})
        check_keys(cell_dataset, {'training_cell', 'separate_cell', 'validation_fraction', 'seed'}, 'cell_dataset')
        for key, value in dict(training_cell='HEK293T', separate_cell='HeLa',
                               validation_fraction=.1, seed=42).items():
            cell_dataset.setdefault(key, value)
        build_sample_pairs(config['samples'])  # Check WT/IVT identities before doing any work.
    else:
        modkit = config.setdefault('modkit', {})
        check_keys(modkit, {'run_pileup', 'executable', 'threads', 'filter_threshold', 'mod_thresholds', 'include_bed', 'seed'}, 'modkit')
        for key, value in dict(run_pileup=False, executable='modkit', threads=4,
                               filter_threshold=None, mod_thresholds=None, include_bed=None, seed=None).items():
            modkit.setdefault(key, value)
        if not isinstance(modkit['run_pileup'], bool):
            raise ValueError('modkit.run_pileup must be true or false')
        if not isinstance(modkit['executable'], str) or not modkit['executable']:
            raise ValueError('modkit.executable must be a command name or path')
        if '/' in modkit['executable']:
            modkit['executable'] = str(resolve_path(modkit['executable'], path.parent))
        if not isinstance(modkit['threads'], int) or isinstance(modkit['threads'], bool) or modkit['threads'] < 1:
            raise ValueError('modkit.threads must be a positive integer')
        modkit['include_bed'] = resolve_path(modkit['include_bed'], path.parent)
        filtering = config.setdefault('filtering', {})
        check_keys(filtering, {'reference_only_targets', 'm6a_drach_keep', 'm6a_mettl16_keep'}, 'filtering')
        filtering.setdefault('reference_only_targets', ['m5C', 'pseU'])
        filtering.setdefault('m6a_drach_keep', False)
        filtering.setdefault('m6a_mettl16_keep', False)
        reference_only = filtering['reference_only_targets']
        if (not isinstance(reference_only, list) or len(reference_only) != len(set(reference_only))
                or any(target not in ('m5C', 'pseU') for target in reference_only)):
            raise ValueError('filtering.reference_only_targets accepts m5C and pseU')
        if ({'m5C', 'pseU'} & set(targets)) - set(reference_only):
            raise ValueError('m5C and pseU targets require reference-only filtering')
        if not isinstance(filtering['m6a_drach_keep'], bool):
            raise ValueError('filtering.m6a_drach_keep must be boolean')
        if not isinstance(filtering['m6a_mettl16_keep'], bool):
            raise ValueError('filtering.m6a_mettl16_keep must be boolean')
        config['sample'] = read_sample(config.get('sample', {}), path.parent)
        config['model_root'] = resolve_path(config.get('model_root'), path.parent)
        directories = config.setdefault('model_directories', {})
        check_keys(directories, {'adenine'}, 'model_directories')
        for target, value in list(directories.items()):
            directories[target] = resolve_path(value, path.parent)
        config.setdefault('bam_threads', 2)
        config.setdefault('recalibrate_bam', False)
        if not isinstance(config['recalibrate_bam'], bool):
            raise ValueError('recalibrate_bam must be true or false')
    return config


def build_sample_pairs(samples):
    """Use sample metadata once, then build the existing WT/IVT site-list input."""
    by_cell = {}
    for sample in samples:
        cell_samples = by_cell.setdefault(sample['cell'], {})
        condition = sample['condition'].lower()
        if condition in cell_samples:
            raise ValueError(f'Duplicate {sample["cell"]} {sample["condition"]} sample')
        cell_samples[condition] = sample
    pairs = []
    for cell, observations in by_cell.items():
        if set(observations) != {'wt', 'ivt'}:
            raise ValueError(f'Define both WT and IVT sample records for {cell}; missing paths may be null')
        wt, ivt = observations['wt'], observations['ivt']
        if wt['reference_cell'] != ivt['reference_cell']:
            raise ValueError(f'WT and IVT reference_cell disagree for {cell}')
        pair = dict(cell=cell, reference_cell=wt['reference_cell'])
        for condition, sample in observations.items():
            pair[f'{condition}_sample'] = sample['sample_id']
            pair[f'{condition}_bedmethyl'] = sample['bedmethyl']
            pair[f'{condition}_pileup_log'] = sample['pileup_log']
        pairs.append(pair)
    return pairs


def enabled_samples(config):
    samples = [sample for sample in config['samples'] if sample['enabled']]
    if not samples:
        raise ValueError('Enable at least one sample in the configuration')
    return samples


def require_path(value, name):
    if value is None:
        raise ValueError(f'Set {name} in the configuration file')
    return value


def index_paths(config):
    root = config['output_dir'] / 'features' / 'index'
    return {target: root / target / 'sites.tsv' for target in model_targets(config)}


def joint_adenine(config):
    return (config.get('nn', {}).get('task') == 'adenine_three_class'
            and {'m6A', 'inosine'} <= set(config.get('targets', ())))


def model_targets(config):
    if joint_adenine(config):
        if config.get('mode') == 'train':
            return ['adenine']
        return ['adenine'] + [target for target in config['targets'] if target not in ('m6A', 'inosine')]
    return config['targets']


def training_site_source(config):
    return config.get('selection', {}).get('source_dir') or config['output_dir'] / 'sites'


def neural_prediction_targets(config):
    return ['adenine'] if joint_adenine(config) else []


def annotation_path(config):
    return config['output_dir'] / 'annotation' / 'sites.tsv'


def model_annotation_path(config):
    """Optional reporting annotation; never a neural-model input requirement."""
    path = annotation_path(config)
    return path if path.exists() else None


def model_directory(config, target):
    value = config['model_directories'].get(target)
    if value is not None:
        return value
    return require_path(config['model_root'], 'model_root') / target


def describe_stage(config, mode, stage):
    """Preview stage paths and unset inputs without opening research data."""
    stages = TRAIN_STAGES if mode == 'train' else PREDICT_STAGES
    if stage not in stages:
        raise ValueError(f'Unknown {mode} stage: {stage}')
    directory_names = dict(quality='quality', pileup='pileup', sites='sites', select='selection', summarize='sites/summary_recount', candidates='candidates',
                           features='features', annotation='annotation', split='splits',
                           cell_dataset='cell_datasets', fit='models', final='final_model',
                           inputs='nn_inputs', infer='scores', filter='predictions', bam='bam')
    output = None if stage == 'inspect' else config['output_dir'] / directory_names[stage]
    if mode == 'train' and stage == 'sites':
        output = training_site_source(config)
    unset = []
    uses_neural = mode == 'train' or bool(neural_prediction_targets(config))
    if stage in ('quality', 'pileup', 'features', 'candidates', 'bam'):
        samples = [sample for sample in config.get('samples', []) if sample['enabled']] if mode == 'train' else [config['sample']]
        if not samples:
            unset.append('enabled samples')
        fields = ('bedmethyl', 'pileup_log') if stage == 'candidates' else ('bam',)
        if mode == 'predict' and stage == 'features' and not uses_neural:
            fields = ()
        for sample in samples:
            for field in fields:
                if stage == 'candidates' and mode == 'predict' and config['modkit']['run_pileup']:
                    continue
                if sample[field] is None:
                    unset.append(f'{sample["sample_id"]}.{field}')
    required_references = {'sites': ('catalog',), 'features': ('fasta',), 'annotation': ('gtf',),
                           'filter': ('catalog',)}
    for name in required_references.get(stage, ()):
        if ((stage == 'sites' and mode == 'train' and config.get('selection', {}).get('source_dir'))
                or (mode == 'predict' and name == 'catalog' and config['genome'] == 'other')):
            continue
        if config['references'][name] is None:
            unset.append(f'references.{name}')
    if stage in ('inputs', 'infer') and config['model_root'] is None:
        for target in neural_prediction_targets(config):
            if config['model_directories'].get(target) is None:
                unset.append(f'model_directories.{target}')
    runtime = dict(genome=config['genome'], device=config['nn']['device'], cpu_threads=config['nn']['cpu_threads']) if mode == 'predict' else {}
    return dict(mode=mode, stage=stage, config=config['config_path'], output=output,
                indices=index_paths(config), annotation=annotation_path(config), unset_inputs=unset,
                **runtime,
                dry_run=True, note='No data files opened; existing upstream outputs are checked only during execution')


def describe_pipeline(config, stages):
    """Preview an ordered group of stages without opening research data."""
    reports = [describe_stage(config, config['mode'], stage) for stage in stages]
    unset = sorted({name for report in reports for name in report['unset_inputs']})
    runtime = dict(genome=config['genome'], device=config['nn']['device'], cpu_threads=config['nn']['cpu_threads']) if config['mode'] == 'predict' else {}
    return dict(mode=config['mode'], stage='all', config=config['config_path'],
                output=config['output_dir'], planned_stages=list(stages), stages=reports,
                **runtime,
                unset_inputs=unset, recalibrate_bam='bam' in stages, dry_run=True)
