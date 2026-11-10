"""Launch HEK293T-only epoch selection and refitting in an isolated process."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from training.fold_trial import (PROJECT_ROOT, execution_log, require_gpu_runtime,
                                 tensorflow_devices, utc_now, write_json)
from utils.config import index_paths, joint_adenine, load_config
from utils.references.catalog import sha256


def cell_trial_plan(config, output_dir=None):
    """Resolve settings without reading tensors or importing TensorFlow."""
    cell = config['cell_dataset']
    if (not joint_adenine(config) or cell['training_cell'] != 'HEK293T'
            or cell['separate_cell'] != 'HeLa'):
        raise ValueError('This entry requires adenine_three_class, training_cell HEK293T and separate_cell HeLa')
    root = (Path(output_dir).resolve() if output_dir is not None else
            config['output_dir'] / 'cell_models' / 'HEK293T')
    nn = config['nn']
    settings = dict(target='adenine', training_cell='HEK293T', separate_cell='HeLa',
                    flank=config['features']['flank'], seed=cell['seed'],
                    epochs=nn['epochs'], patience=nn['patience'], batch_sites=nn['batch_sites'],
                    learning_rate=nn['learning_rate'], max_reads=nn['max_reads'],
                    sampling_seed=nn['sampling_seed'],
                    threshold=nn['target_settings']['adenine']['threshold'])
    return dict(output_dir=str(root), model_output=str(root / 'adenine'),
                index_path=str(index_paths(config)['adenine']),
                datasets_root=str(config['output_dir'] / 'cell_datasets'),
                execution_log=str(root / 'execution.log'), settings=settings,
                policies=dict(scope='HEK293T-only selection and refit; no automatic HeLa evaluation',
                              inputs='reuse existing index, manifests and Parquet without modifying them',
                              device='working CUDA GPU required; no CPU fallback',
                              evaluation='internal weak-label validation for epoch selection only'))


def source_hashes():
    """Record the training and NN sources present when this worker starts."""
    paths = list((PROJECT_ROOT / 'training').rglob('*.py'))
    paths.extend((PROJECT_ROOT / 'utils/nn').rglob('*.py'))
    paths.extend(PROJECT_ROOT / name for name in
                 ('utils/config.py', 'utils/features/site_dataset.py'))
    return {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in sorted(paths)}


def run_cell_trial(config, output_dir=None):
    """Select epochs within HEK293T, then refit using all its observations."""
    plan = cell_trial_plan(config, output_dir)
    out = Path(plan['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    marker = out / 'INCOMPLETE.txt'
    marker.write_text('HEK293T-only training is incomplete; see launch_run.json and execution.log.\n')
    launch = dict(plan, status='running', phase='initializing', pid=os.getpid(),
                  started_utc=utc_now(), config_path=str(config['config_path']))
    write_json(out / 'launch_run.json', launch)
    print(f"HEK293T-only training log: {plan['execution_log']}", flush=True)
    with execution_log(out / 'execution.log'):
        try:
            snapshot = out / 'config.snapshot.yaml'
            shutil.copyfile(config['config_path'], snapshot)
            launch['config_sha256'] = sha256(snapshot)
            write_json(out / 'resolved_config.json', config)
            write_json(out / 'resolved_settings.json', plan)
            hashes = out / 'source_code_sha256.json'
            write_json(hashes, source_hashes())
            launch['source_manifest_sha256'] = sha256(hashes)
            launch['phase'] = 'checking_gpu'
            write_json(out / 'launch_run.json', launch)
            print(json.dumps(launch, indent=2, default=str), flush=True)
            import tensorflow as tf
            launch.update(tensorflow_devices(tf))
            write_json(out / 'launch_run.json', launch)
            launch.update(require_gpu_runtime(tf))
            launch['phase'] = 'selection_and_refit'
            write_json(out / 'launch_run.json', launch)
            from training.cell_model import train_cell_model
            result = train_cell_model(plan['index_path'], plan['datasets_root'],
                                      config['samples'], plan['model_output'], **plan['settings'])
            if result['status'] != 'complete':
                raise RuntimeError('Cell-model selection and refit did not complete')
            model = Path(plan['model_output'])
            launch.update(status='complete', phase='finished', finished_utc=utc_now(),
                          feature_set=result['feature_set'], best_epoch=result['best_epoch'],
                          result_run=str(model / 'run.json'),
                          history=str(model / 'train/history.csv'),
                          validation_metrics=str(model / 'validation/metrics.json'),
                          final_model=str(model / 'final_model'))
            write_json(out / 'launch_run.json', launch)
            marker.unlink()
            print('HEK293T-only selection and final refit complete; no HeLa evaluation was run.', flush=True)
        except BaseException as error:
            launch.update(status='failed', phase='failed', finished_utc=utc_now(),
                          error_type=type(error).__name__, error=str(error))
            write_json(out / 'launch_run.json', launch)
            (out / 'FAILED.txt').write_text(str(error) + '\n', encoding='utf-8')
            traceback.print_exc()
            raise
    return launch


def start_background_cell_trial(config, output_dir=None):
    """Detach one worker, preserving existing outputs and the caller's CUDA setup."""
    plan = cell_trial_plan(config, output_dir)
    out = Path(plan['output_dir'])
    launch_log = out.parent / (out.name + '.launch.log')
    launch_record = out.parent / (out.name + '.launch.json')
    for path in (out, launch_log, launch_record):
        if path.exists():
            raise FileExistsError(path)
    command = [sys.executable, '-u', '-m', 'training.cell_trial',
               '--config', str(config['config_path']), '--output-dir', str(out)]
    report = dict(plan, status='starting', started_utc=utc_now(), command=command,
                  config_path=str(config['config_path']), launch_log=str(launch_log),
                  launch_record=str(launch_record), worker_status=str(out / 'launch_run.json'))
    out.parent.mkdir(parents=True, exist_ok=True)
    with launch_record.open('x', encoding='utf-8') as record, launch_log.open('x', encoding='utf-8') as log:
        try:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            report['pid'] = process.pid
        except BaseException as error:
            report.update(status='failed', finished_utc=utc_now(),
                          error_type=type(error).__name__, error=str(error))
            json.dump(report, record, indent=2, ensure_ascii=False)
            record.write('\n')
            raise
        json.dump(report, record, indent=2, ensure_ascii=False)
        record.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'training/config.yaml')
    parser.add_argument('--output-dir', type=Path, help='Unused output root; default: output_dir/cell_models/HEK293T')
    parser.add_argument('--background', action='store_true', help='Continue after terminal disconnection')
    parser.add_argument('--dry-run', action='store_true', help='Show settings without importing TensorFlow or reading data')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, 'train')
        if args.dry_run:
            report = dict(cell_trial_plan(config, args.output_dir), status='planned',
                          dry_run=True, background=args.background)
        elif args.background:
            report = start_background_cell_trial(config, args.output_dir)
        else:
            report = run_cell_trial(config, args.output_dir)
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
