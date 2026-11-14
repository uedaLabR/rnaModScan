"""Prediction stages reuse training extraction and model input conversion."""
from pathlib import Path
from utils.config import (index_paths, model_annotation_path, model_directory, require_path,
                          neural_prediction_targets, joint_adenine)


def prediction_pileup_paths(config):
    if config.get('modkit', {}).get('run_pileup', False):
        directory = config['output_dir'] / 'pileup'
        from utils.io.artifacts import completed_run, file_info
        run = completed_run(directory)
        if run['inputs'][0] != file_info(require_path(config['sample']['bam'], 'sample.bam')):
            raise ValueError('Completed modkit pileup belongs to another input BAM')
        return directory / 'all_modifications.bed', directory / 'pileup.log'
    return config['sample']['bedmethyl'], config['sample']['pileup_log']


def run_prediction_stage(config, stage, *, isolate_gpu=False):
    root = config['output_dir']
    references = config['references']
    sample = config['sample']
    features = config['features']
    neural_targets = neural_prediction_targets(config)
    if stage == 'pileup':
        from prediction.modkit_pileup import run_modkit_pileup
        settings = config['modkit']
        return run_modkit_pileup(require_path(sample['bam'], 'sample.bam'), root / 'pileup',
            executable=settings['executable'], threads=settings['threads'],
            reference_fasta=references['fasta'], filter_threshold=settings['filter_threshold'],
            mod_thresholds=settings['mod_thresholds'], include_bed=settings['include_bed'], seed=settings['seed'])
    if stage == 'quality':
        from utils.quality.stats import estimate_quality
        bam = require_path(sample['bam'], 'sample.bam')
        return estimate_quality(bam, root / 'quality', **config['quality'])
    if stage == 'candidates':
        from prediction.candidates import prepare_prediction_sites
        bedmethyl, pileup_log = prediction_pileup_paths(config)
        bedmethyl = require_path(bedmethyl, 'sample.bedmethyl')
        pileup_log = require_path(pileup_log, 'sample.pileup_log')
        thresholds = {key: config['candidates'][key] for key in ('min_depth', 'min_modified', 'min_af')}
        by_target = {target: dict(thresholds) for target in config['targets']}
        return prepare_prediction_sites(bedmethyl, pileup_log, sample['sample_id'],
            root / 'candidates', by_target, features['flank'])
    if stage == 'features':
        from prediction.reference_catalog import checked_catalog_path
        catalog_path = checked_catalog_path(config)
        fasta = require_input_file(references['fasta'], 'references.fasta')
        require_input_file(Path(str(fasta) + '.fai'), 'FASTA index')
        from utils.io.artifacts import completed_run
        completed_run(root / 'candidates')
        from prediction.routing import route_prediction_sites
        route_prediction_sites(root / 'candidates', root / 'routing', config['targets'],
                               config['filtering'], references['fasta'], joint=joint_adenine(config))
        if not neural_targets:
            return dict(status='not_required', reason='all targets use known-site or enabled motif rules without neural inputs')
        require_path(sample['bam'], 'sample.bam')
        sites = {target: root / 'routing' / target / 'nn_sites.tsv' for target in neural_targets}
        if joint_adenine(config):
            from prediction.model_bundle import load_model_bundle
            _, model_run = load_model_bundle(model_directory(config, 'adenine'))
            maximum = model_run['input_schema'].get('max_reads')
            if maximum is not None:
                from utils.features.site_bag_pipeline import extract_site_bags_and_index
                return extract_site_bags_and_index(sites['adenine'], [sample], fasta, root / 'features',
                    catalog_path, mode='inference', max_reads=maximum,
                    sampling_seed=model_run['input_schema'].get('sampling_seed', 42),
                    flank=features['flank'], min_mapq=features['min_mapq'], bam_threads=features['bam_threads'],
                    progress_seconds=features['progress_seconds'], candidate_group_gap=features['candidate_group_gap'],
                    read_batch_size=features['read_batch_size'], group_workers=features['group_workers'],
                    group_batch_size=features['group_batch_size'], max_fetch_span=features['max_fetch_span'])
        raise ValueError('Use the joint adenine model with saved maximum read depth for neural prediction')
    if stage == 'annotation':
        from utils.features.position import annotate_positions
        from utils.io.artifacts import completed_run
        gtf = require_path(references['gtf'], 'references.gtf')
        completed_run(root / 'candidates')
        sites = [root / 'candidates' / target / 'sites.tsv' for target in config['targets']]
        report = dict(sites=annotate_positions(sites, gtf, root / 'annotation'))
        return report
    if stage == 'inputs':
        from prediction.input_parquet import prepare_prediction_inputs, prepare_policy_inputs
        from utils.io.artifacts import completed_run
        completed_run(root / 'routing')
        nn = config['nn']
        reports = {}
        for target, index in index_paths(config).items():
            policy_sites = root / 'routing' / target / 'policy_sites.tsv'
            if target not in neural_targets:
                reports[target] = prepare_policy_inputs(policy_sites, target, root / 'nn_inputs' / target,
                    position_sites=model_annotation_path(config))
                continue
            reports[target] = prepare_prediction_inputs(index, model_directory(config, target),
                root / 'nn_inputs' / target, position_sites=model_annotation_path(config), batch_sites=nn['batch_sites'],
                policy_sites=policy_sites)
        return reports
    if stage == 'infer':
        if isolate_gpu:
            return run_isolated_inference(config)
        from prediction.scoring import score_prepared_inputs
        reports = {}
        for target in neural_targets:
            reports[target] = score_prepared_inputs(root / 'nn_inputs' / target,
                model_directory(config, target), root / 'scores' / target,
                batch_sites=config['nn']['batch_sites'], device=config['nn']['device'],
                cpu_threads=config['nn']['cpu_threads'])
        return reports
    if stage == 'filter':
        from prediction.reference_catalog import checked_catalog_path
        from prediction.filtering import filter_scored_inputs
        from prediction.summary import combine_predictions
        from utils.references.catalog import ReferenceCatalog
        catalog = ReferenceCatalog(checked_catalog_path(config))
        if joint_adenine(config):
            from prediction.adenine_filtering import filter_joint_inputs
            filter_joint_inputs(root / 'nn_inputs' / 'adenine', root / 'scores' / 'adenine',
                root / 'predictions', catalog, sample['reference_cell'], condition=sample['condition'],
                m6a_drach_keep=config['filtering']['m6a_drach_keep'],
                m6a_mettl16_keep=config['filtering'].get('m6a_mettl16_keep', False))
        directories = []
        for target in config['targets']:
            out = root / 'predictions' / target
            if joint_adenine(config) and target in ('m6A', 'inosine'):
                directories.append(out)
                continue
            scores = root / 'scores' / target if target in neural_targets else None
            filter_scored_inputs(root / 'nn_inputs' / target, scores, out,
                                 catalog, sample['reference_cell'], condition=sample['condition'],
                                 auxiliary_motif_min_depth=config['filtering'].get('auxiliary_motif_min_depth',20),
                                 auxiliary_motif_min_af=config['filtering'].get('auxiliary_motif_min_af',.40))
            directories.append(out)
        report = combine_predictions(directories, root / 'predictions' / 'summary', include_sites=False)
        from prediction.bed_output import export_filtered_bed
        bedmethyl, _ = prediction_pileup_paths(config)
        report['filtered_bed'] = export_filtered_bed(bedmethyl, directories, sample['sample_id'],
                                                    root / 'predictions' / 'bedmethyl',
                                                    auxiliary_motif_min_depth=config['filtering'].get('auxiliary_motif_min_depth',20),
                                                    auxiliary_motif_min_af=config['filtering'].get('auxiliary_motif_min_af',.40))
        return report
    if stage == 'bam':
        from prediction.bam_recalibration import write_filtered_bam
        bam = require_path(sample['bam'], 'sample.bam')
        directories = [root / 'predictions' / target for target in config['targets']]
        return write_filtered_bam(bam, directories, sample['sample_id'], root / 'bam',
                                  threads=config['bam_threads'])
    raise ValueError(f'Unknown prediction stage: {stage}')


def run_isolated_inference(config):
    """Release TensorFlow memory by exiting the CPU/GPU inference process."""
    import json
    import os
    import subprocess
    import sys
    if not neural_prediction_targets(config):
        return {}
    command = [sys.executable, '-B', '-m', 'prediction', '--config', str(config['config_path']),
               '--genome', config['genome'], '--device', config['nn']['device']]
    if config['nn']['cpu_threads'] is not None:
        command.extend(['--cpu-threads', str(config['nn']['cpu_threads'])])
    command.extend(['--stage', 'infer'])
    environment = os.environ.copy()
    if config['nn']['device'] == 'cpu':
        environment['CUDA_VISIBLE_DEVICES'] = '-1'
    result = subprocess.run(command, capture_output=True, text=True, env=environment)
    if result.returncode:
        raise RuntimeError(f'Inference failed ({result.returncode}): {result.stderr}')
    return json.loads(result.stdout)


def prediction_stages(config, recalibrate=None):
    """Choose the complete prediction workflow and optional BAM recalibration."""
    if recalibrate is None:
        recalibrate = config['recalibrate_bam']
    stages = ['candidates', 'features']
    if config.get('modkit', {}).get('run_pileup', False):
        stages.insert(0, 'pileup')
    if config['references']['gtf'] is not None:
        stages.append('annotation')
    stages.extend(['inputs', 'infer', 'filter'])
    if recalibrate:
        stages.append('bam')
    return stages


def require_input_file(value, name):
    """Check a configured input before starting the expensive workflow."""
    path = require_path(value, name)
    if not path.is_file():
        raise FileNotFoundError(f'{name}: {path}')
    return path


def validate_prediction_inputs(config, stages):
    """Check all required files before creating any prediction output."""
    sample = config['sample']
    references = config['references']
    if 'features' in stages:
        fasta = require_input_file(references['fasta'], 'references.fasta')
        require_input_file(Path(str(fasta) + '.fai'), 'FASTA index')
    if 'candidates' in stages and 'pileup' not in stages:
        bed, log = prediction_pileup_paths(config)
        require_input_file(bed, 'sample.bedmethyl')
        require_input_file(log, 'sample.pileup_log')
    neural_targets = neural_prediction_targets(config)
    if ('features' in stages and neural_targets) or 'bam' in stages or 'pileup' in stages:
        bam = require_input_file(sample['bam'], 'sample.bam')
        if ('features' in stages and neural_targets) or 'pileup' in stages:
            bam_indices = [Path(str(bam) + '.bai'), Path(str(bam) + '.csi'),
                           bam.with_suffix('.bai'), bam.with_suffix('.csi')]
            if not any(path.is_file() for path in bam_indices):
                raise FileNotFoundError(f'BAM index required for {bam}')
            if ('features' in stages and neural_targets) or references['fasta'] is not None:
                fasta = require_input_file(references['fasta'], 'references.fasta')
                require_input_file(Path(str(fasta) + '.fai'), 'FASTA index')
    if 'pileup' in stages:
        import shutil
        from prediction.modkit_pileup import threshold_arguments
        settings = config['modkit']
        if shutil.which(settings['executable']) is None:
            raise FileNotFoundError(f'modkit executable not found: {settings["executable"]}')
        threshold_arguments(settings['filter_threshold'], settings['mod_thresholds'])
        if settings['include_bed'] is not None:
            require_input_file(settings['include_bed'], 'modkit.include_bed')
    if 'features' in stages or 'filter' in stages:
        from prediction.reference_catalog import checked_catalog_path
        checked_catalog_path(config)
    if 'annotation' in stages:
        require_input_file(references['gtf'], 'references.gtf')
    if 'inputs' in stages or 'infer' in stages:
        for target in neural_targets:
            directory = model_directory(config, target)
            for filename in ('model.keras', 'run.json'):
                require_input_file(directory / filename, f'{target} model bundle')

    stage_directories = dict(pileup='pileup', candidates='candidates', features='features', annotation='annotation',
                             inputs='nn_inputs', infer='scores', filter='predictions', bam='bam')
    for stage in stages:
        directory = config['output_dir'] / stage_directories[stage]
        if directory.exists():
            raise FileExistsError(f'Use a new output directory: {directory}')
    if 'features' in stages and (config['output_dir'] / 'routing').exists():
        raise FileExistsError(f'Use a new output directory: {config["output_dir"] / "routing"}')


def run_prediction_pipeline(config, recalibrate=None):
    """Run candidate extraction through prediction, optionally writing a new BAM."""
    stages = prediction_stages(config, recalibrate)
    validate_prediction_inputs(config, stages)
    reports = {}
    for stage in stages:
        reports[stage] = run_prediction_stage(config, stage, isolate_gpu=True)
    return dict(status='complete', sample_id=config['sample']['sample_id'], genome=config['genome'],
                recalibrate_bam='bam' in stages, stages=reports)


def run_prediction_phase(config, phase, recalibrate=None):
    """Separate CPU preparation, GPU scoring and CPU filtering for actual runs."""
    if phase == 'prepare':
        stages = prediction_stages(config, False)
        stages = stages[:stages.index('infer')]
    elif phase == 'filter':
        stages = ['filter']
        write_bam = config['recalibrate_bam'] if recalibrate is None else recalibrate
        if write_bam:
            stages.append('bam')
    else:
        raise ValueError(f'Unknown prediction phase: {phase}')
    validate_prediction_inputs(config, stages)
    reports = {}
    for stage in stages:
        reports[stage] = run_prediction_stage(config, stage)
    return reports
