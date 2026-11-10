"""Shared launcher for preparation and model fitting; never rerun overlap analyses.

Run this file directly for CPU preparation, or use the two launchers in test/.
Inputs and outputs are specified in the YAML. Completed outputs are not replaced.
"""
from pathlib import Path
import importlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / 'training/config.yaml'
RUN_NEURAL_NETWORK = False  # True: also run CV training and fit final models.
PREPARATION_STAGES = ('sites', 'select', 'features', 'annotation', 'split')
MODEL_STAGES = ('fit', 'final')


def check_dependencies(include_training):
    # A package can exist but fail to import because Python or a shared library
    # is incomplete. Test imports before the expensive BED scan, not find_spec.
    packages = ['ctypes', 'yaml', 'pysam', 'pyarrow']
    if include_training:
        packages.append('tensorflow')
    for name in packages:
        try:
            importlib.import_module(name)
        except (ImportError, OSError) as error:
            if name == 'ctypes' or getattr(error, 'name', None) == '_ctypes':
                message = ('This Python installation cannot import ctypes/_ctypes. '
                           'Use a working Python installation and recreate the virtual environment; '
                           'reinstalling pysam alone will not fix the Python standard library.')
            else:
                message = (f'Cannot import {name}: {error}. '
                           'Check this Python environment and install requirements.txt.')
            raise RuntimeError(f'{message}\nInterpreter: {sys.executable}') from error


def require_files(paths):
    """Check paths without scanning BAM, BED or annotation file contents."""
    for path in paths:
        if path is None or not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}')


def check_inputs(config, stages):
    """Model fitting can continue from prepared outputs without rereading raw inputs."""
    from utils.config import enabled_samples, index_paths, training_site_source, model_targets
    from utils.io.artifacts import completed_run

    references, root = config['references'], config['output_dir']
    paths, prepared = [], set()
    if 'sites' in stages:
        if training_site_source(config) != root / 'sites':
            prepared.add(training_site_source(config))
        else:
            paths.append(references['catalog'])
            for sample in config['samples']:
                paths.extend(sample[key] for key in ('bedmethyl', 'pileup_log'))
    if 'select' in stages and 'sites' not in stages:
        prepared.add(training_site_source(config))
    if 'features' in stages:
        paths.append(references['fasta'])
        if references['fasta'] is not None:
            paths.append(Path(str(references['fasta']) + '.fai'))
        for sample in enabled_samples(config):
            bam = sample['bam']
            paths.append(bam)
            if bam is not None:
                indices = [Path(str(bam) + '.bai'), Path(str(bam) + '.csi'),
                           bam.with_suffix('.bai'), bam.with_suffix('.csi')]
                if not any(path.is_file() for path in indices):
                    raise FileNotFoundError(f'Missing BAI/CSI index: {bam}')
        if 'sites' not in stages:
            prepared.add(training_site_source(config))
    if 'annotation' in stages:
        paths.append(references['gtf'])
    if any(stage in stages for stage in ('annotation', 'split', 'fit', 'final')) and 'features' not in stages:
        for index in index_paths(config).values():
            paths.append(index)
            prepared.add(index.parent)
    if 'split' in stages and 'sites' not in stages:
        prepared.add(training_site_source(config))
    if 'fit' in stages and 'split' not in stages:
        for target in model_targets(config):
            prepared.add(root / 'splits' / target)
            for fold in config['nn']['folds']:
                for split in ('train', 'validation', 'test'):
                    paths.append(root / 'splits' / target / f'fold_{fold}' / split / 'sites.tsv')
    if 'final' in stages and 'fit' not in stages:
        for target in model_targets(config):
            for fold in range(5):
                prepared.add(root / 'models' / target / f'fold_{fold}')
    require_files(paths)
    for directory in sorted(prepared):
        completed_run(directory)


def check_outputs(config, stages):
    """Check only the stages being started; preparation stays intact during fitting."""
    directory_names = dict(sites='sites', select='selection', features='features', annotation='annotation',
                           split='splits', fit='models', final='final_model')
    for stage in stages:
        output = config['output_dir'] / directory_names[stage]
        if output.exists():
            if stage == 'select':
                from utils.io.artifacts import completed_run
                completed_run(output)  # The selection stage checks its exact source hashes/seed.
                continue
            names = {entry.name for entry in output.iterdir()}
            if stage == 'features' and 'input_sites' in names and names <= {'input_sites', 'samples'}:
                # Reuse only completed caches. Interrupted BAMs must first be
                # reviewed and moved aside after their writer has stopped.
                from utils.io.artifacts import completed_run
                completed_run(output / 'input_sites')
                samples = output / 'samples'
                if samples.exists():
                    for sample in samples.iterdir():
                        if sample.is_dir():
                            completed_run(sample)
                continue
            raise FileExistsError(f'Output exists: {output}; select a new output_dir in the YAML')


def run(config_path=CONFIG_PATH, stages=None, *, dry_run=False):
    """Run the selected stages, stopping immediately if any stage fails."""
    if sys.version_info < (3, 9):
        raise RuntimeError(f'Python 3.9 or later is required; interpreter: {sys.executable}')
    from utils.config import describe_pipeline, load_config

    if stages is None:
        stages = PREPARATION_STAGES + MODEL_STAGES if RUN_NEURAL_NETWORK else PREPARATION_STAGES
    stages = tuple(stages)
    ordered = PREPARATION_STAGES + MODEL_STAGES
    if not stages or tuple(stage for stage in ordered if stage in stages) != stages:
        raise ValueError('STAGES must follow sites, select, features, annotation, split, fit, final, without duplicates')
    if not dry_run:
        check_dependencies(any(stage in MODEL_STAGES for stage in stages))
    config = load_config(config_path, 'train')
    if dry_run:
        print(json.dumps(describe_pipeline(config, stages), ensure_ascii=False, default=str, indent=2))
        return
    from training.pipeline import run_training_stage
    check_inputs(config, stages)
    check_outputs(config, stages)
    print(f'Python: {sys.executable}', flush=True)
    print(f'Config: {Path(config_path).resolve()}', flush=True)
    print(f'Output: {config["output_dir"]}', flush=True)
    print('Steps: ' + ' -> '.join(stages), flush=True)
    for number, stage in enumerate(stages, 1):
        print(f'\n[{number}/{len(stages)}] Starting {stage}', flush=True)
        report = run_training_stage(config, stage)
        print(json.dumps(report, ensure_ascii=False, default=str, indent=2), flush=True)
        print(f'[{number}/{len(stages)}] Finished {stage}', flush=True)
    print(f'\nCompleted: {config["output_dir"]}', flush=True)


def main(config_path=CONFIG_PATH, stages=None):
    """Optional overrides; both launchers also work without any arguments."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=config_path)
    parser.add_argument('--dry-run', action='store_true', help='Show settings without opening input data')
    arguments = parser.parse_args()
    run(arguments.config, stages, dry_run=arguments.dry_run)


if __name__ == '__main__':
    main()
