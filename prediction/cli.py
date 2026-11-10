"""User-facing prediction command with optional BAM recalibration."""
import argparse
import json
from pathlib import Path

from utils.config import PREDICT_STAGES, describe_pipeline, describe_stage, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description='Predict RNA modifications and rescue reference-supported sites')
    parser.add_argument('--config', type=Path, required=True, help='YAML configuration file')
    parser.add_argument('--genome', choices=('hg38', 'mm10', 'other'),
                        help='Override genome and select human, mouse RMBase-only or no reference overlap')
    parser.add_argument('--dry-run', action='store_true', help='Show the planned workflow without opening data files')
    parser.add_argument('--device', choices=('auto', 'cpu', 'gpu'),
                        help='Override nn.device: auto (GPU if available), cpu (hide GPUs), gpu (require GPU)')
    parser.add_argument('--cpu-threads', type=int, help='Override nn.cpu_threads (positive TensorFlow thread count)')
    parser.add_argument('--stage', choices=('all', 'prepare', *PREDICT_STAGES), default='all',
                        help='prepare: CPU to Parquet; infer: Parquet to scores; filter: CPU decisions; default: all')
    recalibration = parser.add_mutually_exclusive_group()
    recalibration.add_argument('--recalibrate', dest='recalibrate', action='store_true',
                               help='Also write a new BAM with rejected target ML bytes set to zero')
    recalibration.add_argument('--no-recalibrate', dest='recalibrate', action='store_false',
                               help='Produce predictions without rewriting BAM')
    parser.set_defaults(recalibrate=None)
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config, 'predict', genome=arguments.genome)
        if arguments.device is not None:
            config['nn']['device'] = arguments.device
        if arguments.cpu_threads is not None:
            config['nn']['cpu_threads'] = arguments.cpu_threads
        from prediction.runtime import validate_runtime
        validate_runtime(config['nn']['device'], config['nn']['cpu_threads'])
        from prediction.pipeline import (prediction_stages, run_prediction_pipeline,
                                         run_prediction_stage, run_prediction_phase)
        if arguments.stage in ('prepare', 'filter'):
            if arguments.stage == 'prepare' and arguments.recalibrate is not None:
                raise ValueError('Use --recalibrate/--no-recalibrate with all or filter')
            if arguments.stage == 'prepare':
                stages = prediction_stages(config, False)
                stages = stages[:stages.index('infer')]
            else:
                write_bam = config['recalibrate_bam'] if arguments.recalibrate is None else arguments.recalibrate
                stages = ['filter', 'bam'] if write_bam else ['filter']
            if arguments.dry_run:
                report = describe_pipeline(config, stages)
            else:
                report = run_prediction_phase(config, arguments.stage, arguments.recalibrate)
        elif arguments.stage != 'all':
            if arguments.recalibrate is not None:
                raise ValueError('Use --recalibrate/--no-recalibrate with all or filter')
            if arguments.dry_run:
                report = describe_stage(config, 'predict', arguments.stage)
            else:
                report = run_prediction_stage(config, arguments.stage)
        elif arguments.dry_run:
            report = describe_pipeline(config, prediction_stages(config, arguments.recalibrate))
        else:
            report = run_prediction_pipeline(config, arguments.recalibrate)
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as error:
        parser.exit(2, f'error: {error}\n')
