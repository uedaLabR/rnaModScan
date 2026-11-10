"""Run one isolated adenine fold using the existing index and splits."""
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

from utils.config import index_paths, joint_adenine, load_config
from utils.references.catalog import sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def trial_plan(config, output_dir=None, *, fold=0, fold_output_dir=None):
    """Describe paths and settings without opening features or importing TensorFlow."""
    if not joint_adenine(config):
        raise ValueError('Fold-0 trial requires nn.task: adenine_three_class')
    if type(fold) is not int or fold not in range(5):
        raise ValueError('fold must be 0..4')
    name = f'fold_{fold}'
    root = Path(output_dir).resolve() if output_dir is not None else config['output_dir'] / 'trials' / (name + '_300reads' if config['nn'].get('max_reads') == 300 else name)
    model_output = Path(fold_output_dir).resolve() if fold_output_dir is not None else root / 'adenine' / name
    settings = config['nn']
    return dict(target='adenine', fold=fold, output_dir=str(root),
                index_path=str(index_paths(config)['adenine']),
                cv_directory=str(config['output_dir'] / 'splits' / 'adenine' / name),
                fold_output=str(model_output), execution_log=str(root / 'execution.log'),
                settings=dict(batch_sites=settings['batch_sites'],
                              epochs=settings['epochs'], patience=settings['patience'],
                              seed=config['cv']['seed'] + fold, learning_rate=settings['learning_rate'],
                              threshold=settings['target_settings']['adenine']['threshold'],
                              max_reads=settings['max_reads'],
                              sampling_seed=settings['sampling_seed']),
                policies=dict(scope=f'fold {fold} only; no final model or other folds',
                              reads='uniformly sample at most nn.max_reads; pad and mask remaining slots',
                              device='CUDA GPU required; fail before fitting instead of falling back to CPU',
                              evaluation='held-out fold metrics against operational weak labels'))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')


@contextmanager
def execution_log(path):
    """Capture Python output and native TensorFlow stdout/stderr in one log."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    try:
        with Path(path).open('w', encoding='utf-8', buffering=1) as log:
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


def tensorflow_devices(tf):
    """Record the devices visible under the launcher's CUDA environment."""
    physical = []
    for device in tf.config.list_physical_devices():
        record = dict(name=device.name, device_type=device.device_type)
        if device.device_type == 'GPU':
            record['details'] = tf.config.experimental.get_device_details(device)
        physical.append(record)
    return dict(tensorflow_version=tf.__version__, physical_devices=physical,
                tensorflow_build=tf.sysconfig.get_build_info(), built_with_cuda=tf.test.is_built_with_cuda(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def require_gpu_runtime(tf):
    """Enable memory growth and verify an actual GPU operation before fitting."""
    devices = tf.config.list_physical_devices('GPU')
    if not devices or not tf.test.is_built_with_cuda():
        raise RuntimeError('Training requires a working CUDA GPU. No CPU fallback is allowed; '
                           'use a verified TensorFlow/CUDA environment and inspect execution.log.')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    with tf.device('/GPU:0'):
        value = tf.constant([[1.0]])
        probe = tf.matmul(value, value)
    if 'GPU:' not in probe.device.upper() or float(probe.numpy()[0][0]) != 1.0:
        raise RuntimeError('TensorFlow GPU probe did not execute successfully on the GPU; refusing CPU fallback')
    return dict(gpu_memory_growth=True, gpu_probe_device=probe.device)


def run_fold_trial(config, output_dir=None, *, fold=0, fold_output_dir=None):
    """Fit one fold, keeping its history, validation/test results and logs."""
    plan = trial_plan(config, output_dir, fold=fold, fold_output_dir=fold_output_dir)
    out = Path(plan['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    marker = out / 'INCOMPLETE.txt'
    marker.write_text(f'Fold-{fold} trial is running; see execution.log and launch_run.json.\n', encoding='utf-8')
    launch = dict(plan, status='running', pid=os.getpid(), started_utc=utc_now(),
                  config_path=str(config['config_path']), feature_set='sequence_reads_repeat18_v1')
    write_json(out / 'launch_run.json', launch)
    print(f"Fold-{fold} trial log: {plan['execution_log']}", flush=True)
    with execution_log(out / 'execution.log'):
        try:
            snapshot = out / 'config.snapshot.yaml'
            shutil.copyfile(config['config_path'], snapshot)
            launch['config_sha256'] = sha256(snapshot)
            write_json(out / 'resolved_config.json', config)
            print(json.dumps(launch, indent=2, ensure_ascii=False), flush=True)
            launch['phase'] = 'training'
            write_json(out / 'launch_run.json', launch)
            # Import only after launch metadata/logging are in place. Dry-run
            # never reaches TensorFlow or reads the training data.
            import tensorflow as tf
            launch.update(tensorflow_devices(tf))
            write_json(out / 'launch_run.json', launch)
            print(json.dumps(dict(devices=launch['physical_devices'], build=launch['tensorflow_build']),
                             indent=2, default=str), flush=True)
            launch.update(require_gpu_runtime(tf))
            write_json(out / 'launch_run.json', launch)
            from training.fit import train_fold
            result = train_fold(plan['index_path'], plan['cv_directory'], plan['fold_output'], 'adenine',
                                **plan['settings'])
            launch.update(status='complete', finished_utc=utc_now(),
                          feature_set=result['feature_set'],
                          result_run=str(Path(plan['fold_output']) / 'run.json'),
                          history=str(Path(plan['fold_output']) / 'train' / 'history.csv'),
                          validation_metrics=str(Path(plan['fold_output']) / 'validation' / 'metrics.json'),
                          test_metrics=str(Path(plan['fold_output']) / 'test' / 'metrics.json'))
            write_json(out / 'launch_run.json', launch)
            marker.unlink()
            print(f'Fold-{fold} trial complete.', flush=True)
        except BaseException as error:
            launch.update(status='failed', finished_utc=utc_now(),
                          error_type=type(error).__name__, error=str(error))
            write_json(out / 'launch_run.json', launch)
            (out / 'FAILED.txt').write_text(str(error) + '\n', encoding='utf-8')
            traceback.print_exc()
            raise
    return launch


def start_background_trial(config, output_dir=None, *, fold=0, fold_output_dir=None):
    """Start a detached foreground worker; inherit the selected CUDA environment."""
    plan = trial_plan(config, output_dir, fold=fold, fold_output_dir=fold_output_dir)
    out = Path(plan['output_dir'])
    if out.exists():
        raise FileExistsError(out)
    launch_log = out.parent / (out.name + '.launch.log')
    launch_record = out.parent / (out.name + '.launch.json')
    for path in (launch_log, launch_record):
        if path.exists():
            raise FileExistsError(path)
    command = [sys.executable, '-u', '-m', 'training.fold_trial',
               '--config', str(config['config_path']), '--output-dir', str(out), '--fold', str(fold)]
    if fold_output_dir is not None:
        command.extend(['--fold-output-dir', str(plan['fold_output'])])
    report = dict(plan, status='starting', started_utc=utc_now(),
                  config_path=str(config['config_path']), command=command,
                  launch_log=str(launch_log), launch_record=str(launch_record))
    out.parent.mkdir(parents=True, exist_ok=True)
    with launch_record.open('x', encoding='utf-8') as record, launch_log.open('x', encoding='utf-8') as log:
        try:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            report['pid'] = process.pid
        except BaseException as error:
            report.update(status='failed', error_type=type(error).__name__, error=str(error))
            json.dump(report, record, indent=2, ensure_ascii=False)
            record.write('\n')
            raise
        json.dump(report, record, indent=2, ensure_ascii=False)
        record.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train/evaluate one adenine fold, separately from production models')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'), help='Training YAML')
    parser.add_argument('--output-dir', type=Path, help='New isolated trial directory; default: output_dir/trials/fold_0')
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--fold-output-dir', type=Path, help='Optional separate directory for model/history/evaluation outputs')
    parser.add_argument('--dry-run', action='store_true', help='Show paths/settings without reading features or importing TensorFlow')
    parser.add_argument('--background', action='store_true', help='Detach the worker so training continues after terminal disconnection')
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config, 'train')
        if arguments.dry_run:
            report = dict(trial_plan(config, arguments.output_dir, fold=arguments.fold,
                          fold_output_dir=arguments.fold_output_dir), status='planned', dry_run=True,
                          background=arguments.background)
        elif arguments.background:
            report = start_background_trial(config, arguments.output_dir, fold=arguments.fold,
                                            fold_output_dir=arguments.fold_output_dir)
        else:
            report = run_fold_trial(config, arguments.output_dir, fold=arguments.fold,
                                    fold_output_dir=arguments.fold_output_dir)
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
