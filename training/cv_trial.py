"""Detached five-fold trials with one isolated process per fold and validation summaries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from training.fold_trial import PROJECT_ROOT, utc_now, write_json
from training.run_training import check_inputs
from utils.config import load_config
from utils.io.artifacts import completed_run
from utils.references.catalog import sha256


def check_comparison(config, baseline):
    """Require the same data and effective settings apart from learning rate."""
    completed_run(baseline)
    launch = json.loads((baseline.parents[1] / 'launch_run.json').read_text())
    original = load_config(launch['config_path'], 'train')
    if sha256(original['config_path']) != launch['config_sha256']:
        raise ValueError('The original configuration changed after the baseline trial')
    def comparable(settings):
        settings = dict(settings, nn=dict(settings['nn']))
        settings.pop('config_path')
        settings['nn'].pop('learning_rate')
        return settings
    if comparable(original) != comparable(config):
        raise ValueError('Only nn.learning_rate may differ from the preserved baseline')
    if config['nn']['learning_rate'] != 0.0003 or original['nn']['learning_rate'] != 0.001:
        raise ValueError('This comparison requires trial LR=0.0003 and baseline LR=0.001')
    if config['nn']['folds'] != list(range(5)):
        raise ValueError('This trial requires all five folds in order 0..4')
    check_inputs(config, ('fit',))
    return original


def save_config_snapshot(config, destination):
    """Keep the YAML portable to its run folder by making input paths absolute."""
    import yaml
    values = yaml.safe_load(config['config_path'].read_text())
    values['output_dir'] = str(config['output_dir'])
    values['references'] = {name: str(path) if path is not None else None
                            for name, path in config['references'].items()}
    if 'selection' in values:
        path = config['selection']['source_dir']
        values['selection']['source_dir'] = str(path) if path is not None else None
    for raw, resolved in zip(values['samples'], config['samples']):
        for name in ('bam', 'bedmethyl', 'pileup_log'):
            raw[name] = str(resolved[name]) if resolved[name] is not None else None
    destination.write_text(yaml.safe_dump(values, sort_keys=False))
    reloaded = load_config(destination, 'train')
    if {k: v for k, v in reloaded.items() if k != 'config_path'} != {
            k: v for k, v in config.items() if k != 'config_path'}:
        raise ValueError('Resolved configuration changed while saving the snapshot')


def trial_plan(config, out, baseline, gpus):
    if not gpus or len(set(gpus)) != len(gpus) or any(not gpu.isdigit() for gpu in gpus):
        raise ValueError('Use distinct numeric GPU IDs, for example --gpus 0 1')
    return dict(config_path=str(config['config_path']), output_dir=str(out), baseline_dir=str(baseline),
                learning_rate=config['nn']['learning_rate'], folds=config['nn']['folds'], gpu_ids=gpus,
                summary=str(out / 'summary/validation_comparison.xlsx'),
                policy='Same prepared inputs, splits and settings; validation-only learning-rate comparison')


def launch_fold(config_path, out, fold, gpu):
    """A fresh worker releases TensorFlow/GPU/read caches on exit."""
    command = [sys.executable, '-u', '-m', 'training.fold_trial', '--config', str(config_path),
               '--fold', str(fold), '--output-dir', str(out / 'workers' / f'fold_{fold}'),
               '--fold-output-dir', str(out / 'adenine' / f'fold_{fold}')]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    log = out / 'logs' / f'fold_{fold}.launch.log'
    with log.open('x') as handle:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    return process


def run_cv_trial(out, baseline, gpus):
    from training.validation_comparison import summarize_validation_folds
    config_path = out / 'config.snapshot.yaml'
    config = load_config(config_path, 'train')
    report = json.loads((out / 'run.json').read_text())
    report.update(status='running', pid=os.getpid(), started_utc=utc_now())
    pending = list(config['nn']['folds'])
    active = {}
    def refresh_summary():
        try:
            summarize_validation_folds(out, baseline)
            report.pop('summary_error', None)
        except Exception as error:
            # An Excel/report error must not abandon pending training folds.
            report['summary_error'] = str(error)
            print(f'Validation summary update failed: {error}', flush=True)
    refresh_summary()
    while pending or active:
        for gpu in gpus:
            if gpu not in active and pending:
                fold = pending.pop(0)
                try:
                    process = launch_fold(config_path, out, fold, gpu)
                except OSError as error:
                    report['folds'][str(fold)].update(status='failed', gpu=gpu,
                        error=str(error), finished_utc=utc_now())
                    print(f'fold {fold} could not start: {error}', flush=True)
                    continue
                active[gpu] = process, fold
                report['folds'][str(fold)].update(status='running', gpu=gpu, pid=process.pid,
                                                 started_utc=utc_now())
                print(f'fold {fold} started on GPU {gpu}, PID {process.pid}', flush=True)
        for gpu, (process, fold) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            state_path = out / 'workers' / f'fold_{fold}' / 'launch_run.json'
            state = json.loads(state_path.read_text()) if state_path.is_file() else {}
            success = code == 0 and state.get('status') == 'complete'
            report['folds'][str(fold)].update(status='complete' if success else 'failed',
                returncode=code, finished_utc=utc_now(), error=state.get('error'))
            print(f'fold {fold}: {report["folds"][str(fold)]["status"]}', flush=True)
            del active[gpu]
            refresh_summary()
        report['updated_utc'] = utc_now()
        write_json(out / 'run.json', report)
        if pending or active:
            time.sleep(10)
    report.update(status='complete' if all(row['status'] == 'complete' for row in report['folds'].values())
                  else 'failed', finished_utc=utc_now())
    summarize_validation_folds(out, baseline)
    for path, digest in report['baseline_sha256'].items():
        if sha256(path) != digest:
            raise ValueError(f'Preserved baseline changed: {path}')
    write_json(out / 'run.json', report)
    if report['status'] == 'complete':
        (out / 'INCOMPLETE.txt').unlink()
    return report


def start_trial(config, out, baseline, gpus):
    """Freeze a separate trial, leaving baseline results and source config intact."""
    plan = trial_plan(config, out, baseline, gpus)
    original = check_comparison(config, baseline)
    out.mkdir(parents=True, exist_ok=False)
    for name in ('workers', 'logs'):
        (out / name).mkdir()
    (out / 'INCOMPLETE.txt').write_text('Five-fold trial is not complete; see run.json and per-fold logs.\n')
    save_config_snapshot(config, out / 'config.snapshot.yaml')
    paths = [baseline / 'run.json', baseline / 'train/best.keras', baseline / 'train/history.csv',
             baseline / 'validation/metrics.json', original['config_path']]
    report = dict(plan, status='starting', created_utc=utc_now(),
        baseline_sha256={str(path): sha256(path) for path in paths},
        folds={str(fold): dict(status='pending') for fold in config['nn']['folds']})
    write_json(out / 'run.json', report)
    from training.validation_comparison import summarize_validation_folds
    summarize_validation_folds(out, baseline)
    command = [sys.executable, '-u', '-m', 'training.cv_trial', '--worker', '--output-dir', str(out),
               '--baseline-dir', str(baseline), '--gpus', *gpus]
    with (out / 'execution.log').open('x') as handle:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(out / 'launch.json', dict(pid=process.pid, command=command, started_utc=utc_now()))
    return dict(report, pid=process.pid)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'training/config_lr0003.yaml')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'results/training_v10/trials/lr_0p0003_5fold')
    parser.add_argument('--baseline-dir', type=Path, default=PROJECT_ROOT / 'results/training_v10/trials/fold_0_300reads/adenine/fold_0')
    parser.add_argument('--gpus', nargs='+', default=['0', '1'])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    out, baseline = arguments.output_dir.resolve(), arguments.baseline_dir.resolve()
    if arguments.worker:
        try:
            report = run_cv_trial(out, baseline, arguments.gpus)
        except BaseException as error:
            report = json.loads((out / 'run.json').read_text())
            report.update(status='failed', error=str(error), finished_utc=utc_now())
            write_json(out / 'run.json', report)
            raise
    else:
        config = load_config(arguments.config, 'train')
        report = trial_plan(config, out, baseline, arguments.gpus) if arguments.dry_run else start_trial(config, out, baseline, arguments.gpus)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str), flush=True)
    return 0 if report.get('status') != 'failed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
