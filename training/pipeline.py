"""Training stages: sites, shared features, annotations, splits and models."""
import json
from pathlib import Path

from utils.config import (build_sample_pairs, enabled_samples, index_paths,
                     require_path, training_site_source, joint_adenine, model_targets)


def training_batch_settings(config):
    """Batch settings for the four tensors already saved in site Parquet."""
    nn = config['nn']
    return dict(batch_sites=nn['batch_sites'], max_reads=nn['max_reads'], sampling_seed=nn['sampling_seed'])


def run_training_stage(config, stage):
    root = config['output_dir']
    references = config['references']
    features = config['features']
    thresholds = config['candidates']
    if stage == 'quality':
        from utils.quality.stats import estimate_quality
        reports = {}
        for sample in enabled_samples(config):
            bam = require_path(sample['bam'], f'samples.{sample["sample_id"]}.bam')
            reports[sample['sample_id']] = estimate_quality(
                bam, root / 'quality' / sample['sample_id'], **config['quality'])
        return reports
    if stage == 'sites':
        from training.site_lists import build_site_lists
        if training_site_source(config) != root / 'sites':
            from utils.io.artifacts import completed_run
            run = completed_run(training_site_source(config))
            expected = {key: thresholds[key] for key in ('min_depth', 'min_modified', 'min_af')}
            if (run['thresholds'] != expected or run['ivt_min_modified'] != thresholds['ivt_min_modified']
                    or run['ivt_af_ratio'] != thresholds['ivt_af_ratio'] or run['window_flank'] < features['flank']):
                raise ValueError('Existing site-list rules differ from the training configuration')
            return dict(status='reused', source=training_site_source(config))
        catalog = require_path(references['catalog'], 'references.catalog')
        return build_site_lists(build_sample_pairs(config['samples']), catalog, root / 'sites',
            min_depth=thresholds['min_depth'], min_modified=thresholds['min_modified'],
            min_af=thresholds['min_af'], ivt_min_modified=thresholds['ivt_min_modified'],
            ivt_af_ratio=thresholds['ivt_af_ratio'], flank=features['flank'])
    if stage == 'select':
        if joint_adenine(config):
            from training.adenine_sites import select_adenine_sites
            return select_adenine_sites(training_site_source(config), root / 'selection', seed=config['selection']['seed'])
        raise ValueError('Use nn.task: adenine_three_class for current training')
    if stage == 'features':
        from utils.io.artifacts import completed_run
        site_root = training_site_source(config)
        if config.get('selection', {}).get('enabled', False) or joint_adenine(config):
            run_training_stage(config, 'select')
            site_root = root / 'selection'
        completed_run(site_root)
        if joint_adenine(config):
            from training.selection_validation import verify_selected_sites
            # Reject missing target classes before opening or scanning any BAM.
            verify_selected_sites(config, site_root)
        samples = enabled_samples(config)
        for sample in samples:
            require_path(sample['bam'], f'samples.{sample["sample_id"]}.bam')
        fasta = require_path(references['fasta'], 'references.fasta')
        site_lists = {target: site_root / target / 'training_sites.tsv' for target in model_targets(config)}
        if joint_adenine(config) and config['nn']['max_reads'] is not None:
            from utils.features.site_bag_pipeline import extract_site_bags_and_index
            return extract_site_bags_and_index(site_lists['adenine'], samples, fasta, root / 'features',
                references['catalog'], mode='training', max_reads=config['nn']['max_reads'],
                sampling_seed=config['nn']['sampling_seed'], sample_workers=features['sample_workers'],
                flank=features['flank'], min_mapq=features['min_mapq'], bam_threads=features['bam_threads'],
                progress_seconds=features['progress_seconds'],
                candidate_group_gap=features['candidate_group_gap'],
                read_batch_size=features['read_batch_size'], group_workers=features['group_workers'],
                group_batch_size=features['group_batch_size'], max_fetch_span=features['max_fetch_span'])
        raise ValueError('Use the joint adenine model and max_reads=300 for site Parquet extraction')
    if stage == 'summarize':
        from training.site_summary import summarize_site_list
        from utils.io.artifacts import completed_run
        run = completed_run(training_site_source(config))
        output = root / 'sites' / 'summary_recount'
        summarize_site_list(training_site_source(config) / 'all_sites.tsv', output, run['pairs'], run['reference_sources'])
        return dict(summary=output)
    if stage == 'annotation':
        from utils.features.position import annotate_positions
        from utils.io.artifacts import completed_run
        gtf = require_path(references['gtf'], 'references.gtf')
        catalog = require_path(references['catalog'], 'references.catalog')
        assembly = json.loads(Path(catalog).read_text(encoding='utf-8-sig')).get('assembly')
        if not isinstance(assembly, str) or not assembly.strip():
            raise ValueError('The training reference catalog must declare an assembly')
        indices = index_paths(config)
        for path in indices.values():
            completed_run(path.parent)
        report = dict(sites=annotate_positions(list(indices.values()), gtf, root / 'annotation',
                                             assembly=assembly))
        return report
    if stage == 'split':
        from training.cross_validation import build_cross_validation
        from utils.io.artifacts import completed_run
        if config['cv']['require_all_samples']:
            run = completed_run(training_site_source(config))
            ready = {sample['sample_id'] for sample in run['inputs'] if sample['status'] == 'available'}
            expected = {sample['sample_id'] for sample in config['samples']}
            if not expected <= ready:
                raise ValueError(f'Wait for all modkit inputs, then rebuild sites: missing {sorted(expected - ready)}')
        outputs = {target: root / 'splits' / target for target in model_targets(config)}
        return build_cross_validation(index_paths(config), root / 'splits' / 'summary',
            flank=features['flank'], seed=config['cv']['seed'],
            validation_fraction=config['cv']['validation_fraction'], target_outputs=outputs)
    if stage == 'cell_dataset':
        from training.cell_datasets import build_cell_datasets
        return build_cell_datasets(index_paths(config), config['samples'], root / 'cell_datasets',
                                   flank=features['flank'], **config['cell_dataset'])
    if stage == 'fit':
        options = training_batch_settings(config)
        from training.fit import train_fold
        nn = config['nn']
        reports = {}
        for target, index in index_paths(config).items():
            for fold in nn['folds']:
                name = f'fold_{fold}'
                reports[f'{target}/{name}'] = train_fold(index, root / 'splits' / target / name,
                    root / 'models' / target / name, target,
                    **options, epochs=nn['epochs'], patience=nn['patience'],
                    seed=config['cv']['seed'] + fold, learning_rate=nn['learning_rate'],
                    threshold=nn['target_settings'][target]['threshold'])
        return reports
    if stage == 'final':
        options = training_batch_settings(config)
        from training.final_model import train_final_model
        nn = config['nn']
        reports = {}
        for target, index in index_paths(config).items():
            settings = nn['target_settings'][target]
            reports[target] = train_final_model(index, root / 'models' / target,
                root / 'final_model' / target, target, **options,
                epochs=settings['final_epochs'], seed=config['cv']['seed'],
                learning_rate=nn['learning_rate'], threshold=settings['threshold'])
        return reports
    if stage == 'inspect':
        from utils.nn.keras_site import ModelConfig, build_site_model
        target = model_targets(config)[0]
        model = build_site_model(ModelConfig(target=target, window=2 * features['flank'] + 1,
                                            max_reads=config['nn']['max_reads']))
        model.summary()
        return dict(target=target, interpretation='Architecture only; no research data opened')
    raise ValueError(f'Unknown training stage: {stage}')
