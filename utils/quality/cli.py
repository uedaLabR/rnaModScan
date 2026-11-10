"""Inspect modification scores independently of prediction or training."""
import argparse
import json
from pathlib import Path

from utils.config import describe_stage, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description='Inspect ML distributions and estimate advisory confidence thresholds')
    parser.add_argument('--config', type=Path, required=True, help='Training or prediction YAML configuration')
    parser.add_argument('--dry-run', action='store_true', help='Show inputs and outputs without reading BAM')
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config)
        if arguments.dry_run:
            report = describe_stage(config, config['mode'], 'quality')
        elif config['mode'] == 'train':
            from training.pipeline import run_training_stage
            report = run_training_stage(config, 'quality')
        else:
            from prediction.pipeline import run_prediction_stage
            report = run_prediction_stage(config, 'quality')
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        parser.exit(2, f'error: {error}\n')
