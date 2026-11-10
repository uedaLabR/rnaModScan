"""Training command; development scripts select a named stage."""
import argparse
import json
from pathlib import Path

from utils.config import TRAIN_STAGES, describe_pipeline, describe_stage, load_config

PIPELINE_STAGES = ('sites', 'select', 'features', 'annotation', 'split', 'fit', 'final')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Prepare training sites, cross-validate and fit final models')
    parser.add_argument('--config', type=Path, required=True, help='YAML configuration file')
    parser.add_argument('--stage', choices=('all', *TRAIN_STAGES), default='all',
                        help='Run one training stage; default runs all preparation and fitting stages')
    parser.add_argument('--dry-run', action='store_true', help='Show stage paths and unset inputs without opening data files')
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config, 'train')
        if arguments.dry_run:
            if arguments.stage == 'all':
                report = describe_pipeline(config, PIPELINE_STAGES)
            else:
                report = describe_stage(config, 'train', arguments.stage)
        else:
            from training.pipeline import run_training_stage
            if arguments.stage == 'all':
                report = {}
                for stage in PIPELINE_STAGES:
                    report[stage] = run_training_stage(config, stage)
            else:
                report = run_training_stage(config, arguments.stage)
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as error:
        parser.exit(2, f'error: {error}\n')
