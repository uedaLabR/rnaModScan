"""Run the existing prediction stages in a detached, logged process."""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from utils.config import load_config
from utils.references.catalog import sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(values, indent=2, ensure_ascii=False, default=str) + '\n')
    temporary.replace(path)


def prediction_plan(config):
    """Resolve paths and stages without opening research inputs or TensorFlow."""
    from prediction.pipeline import prediction_stages
    out = config['output_dir']
    return dict(output_dir=str(out), sample_id=config['sample']['sample_id'],
                stages=prediction_stages(config), device=config['nn']['device'],
                execution_log=str(out / 'execution.log'), status_path=str(out / 'run.json'),
                recalibrate_bam=config['recalibrate_bam'],
                policies=dict(inference='separate process; prepared Parquet and final model only',
                              outputs='unused output root required; no automatic restart'))


@contextmanager
def execution_log(path):
    """Include Python and native-library stdout/stderr in the execution log."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    try:
        with Path(path).open('w', buffering=1) as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            with redirect_stdout(log), redirect_stderr(log):
                try:
                    yield
                finally:
                    sys.stdout.flush()
                    sys.stderr.flush()
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def source_hashes():
    paths = []
    for directory in ('prediction', 'utils', 'rnamodscan'):
        paths.extend((PROJECT_ROOT / directory).rglob('*.py'))
    return {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in sorted(paths)}


def run_prediction(config):
    """Supervise CPU preparation, isolated inference and CPU filtering."""
    from prediction.pipeline import validate_prediction_inputs, run_prediction_stage
    plan = prediction_plan(config)
    out = Path(plan['output_dir'])
    if out.exists():
        raise FileExistsError(out)
    validate_prediction_inputs(config, plan['stages'])
    out.mkdir(parents=True, exist_ok=False)
    marker = out / 'INCOMPLETE.txt'
    marker.write_text('Prediction is incomplete; see run.json and execution.log.\n')
    run = dict(plan, status='running', current_stage='initializing', completed_stages=[],
               pid=os.getpid(), started_utc=utc_now(), config_path=str(config['config_path']), reports={})
    write_json(out / 'run.json', run)
    print(f"Prediction log: {plan['execution_log']}", flush=True)
    with execution_log(out / 'execution.log'):
        try:
            snapshot = out / 'config.snapshot.yaml'
            shutil.copyfile(config['config_path'], snapshot)
            run['config_sha256'] = sha256(snapshot)
            write_json(out / 'resolved_config.json', config)
            hashes = out / 'source_code_sha256.json'
            write_json(hashes, source_hashes())
            run['source_manifest_sha256'] = sha256(hashes)
            for stage in plan['stages']:
                run.update(current_stage=stage, stage_started_utc=utc_now())
                write_json(out / 'run.json', run)
                print(f'{utc_now()} Starting {stage}', flush=True)
                run['reports'][stage] = run_prediction_stage(config, stage, isolate_gpu=True)
                run['completed_stages'].append(stage)
                write_json(out / 'run.json', run)
                print(f'{utc_now()} Completed {stage}', flush=True)
            run.update(status='complete', current_stage='finished', finished_utc=utc_now())
            write_json(out / 'run.json', run)
            marker.unlink()
        except BaseException as error:
            run.update(status='failed', finished_utc=utc_now(),
                       error_type=type(error).__name__, error=str(error))
            write_json(out / 'run.json', run)
            (out / 'FAILED.txt').write_text(str(error) + '\n')
            traceback.print_exc()
            raise
    return run


def start_background_prediction(config):
    """Detach a worker and refuse existing outputs or launch records."""
    from prediction.pipeline import validate_prediction_inputs
    plan = prediction_plan(config)
    out = Path(plan['output_dir'])
    launch_log = out.parent / (out.name + '.launch.log')
    launch_record = out.parent / (out.name + '.launch.json')
    for path in (out, launch_log, launch_record):
        if path.exists():
            raise FileExistsError(path)
    validate_prediction_inputs(config, plan['stages'])
    command = [sys.executable, '-u', '-m', 'prediction.launcher', '--config', str(config['config_path'])]
    report = dict(plan, status='starting', started_utc=utc_now(), command=command,
                  config_path=str(config['config_path']), launch_log=str(launch_log),
                  launch_record=str(launch_record))
    out.parent.mkdir(parents=True, exist_ok=True)
    with launch_record.open('x') as record, launch_log.open('x') as log:
        try:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            report['pid'] = process.pid
        except BaseException as error:
            report.update(status='failed', finished_utc=utc_now(), error=str(error))
            json.dump(report, record, indent=2, ensure_ascii=False)
            record.write('\n')
            raise
        json.dump(report, record, indent=2, ensure_ascii=False)
        record.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, 'predict')
        if args.dry_run:
            report = dict(prediction_plan(config), status='planned', dry_run=True, background=args.background)
        elif args.background:
            report = start_background_prediction(config)
        else:
            report = run_prediction(config)
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
