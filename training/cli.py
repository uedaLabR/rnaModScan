"""Training command; development scripts select a named stage."""
import argparse
import json
from pathlib import Path
import sys

from utils.config import TRAIN_STAGES, describe_pipeline, describe_stage, load_config

PIPELINE_STAGES = ('sites', 'select', 'features', 'annotation', 'split', 'fit', 'final')


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == 'init':
        from training.onboarding import init_main
        return init_main(arguments[1:])
    if arguments and arguments[0] == 'check':
        from training.onboarding import check_main
        return check_main(arguments[1:])
    parser = argparse.ArgumentParser(description='Prepare training sites, cross-validate and fit final models',
        epilog='New dataset: rnamodscan train init --help. Check inputs first: rnamodscan train check --help.')
    parser.add_argument('--config', type=Path, required=True, help='YAML configuration file')
    parser.add_argument('--stage', choices=('all', *TRAIN_STAGES), default='all',
                        help='Run one training stage; default runs all preparation and fitting stages')
    parser.add_argument('--dry-run', action='store_true', help='Show stage paths and unset inputs without opening data files')
    arguments = parser.parse_args(arguments)
    try:
        config = load_config(arguments.config, 'train')
        from training.onboarding import planned_stages
        stages = planned_stages(config, arguments.stage)
        if arguments.dry_run:
            if arguments.stage == 'all':
                report = describe_pipeline(config, stages)
                report['skipped_optional_stages'] = ['annotation'] if 'annotation' not in stages else []
            else:
                report = describe_stage(config, 'train', arguments.stage)
        else:
            from training.pipeline import run_training_stage
            if arguments.stage == 'all':
                from training.onboarding import check_training
                preflight = check_training(config, stages)
                if preflight['status'] != 'ready':
                    print(json.dumps(preflight, indent=2, ensure_ascii=False, default=str))
                    return 2
                report = {}
                for stage in stages:
                    report[stage] = run_training_stage(config, stage)
            else:
                report = run_training_stage(config, arguments.stage)
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as error:
        parser.exit(2, f'error: {error}\n')
