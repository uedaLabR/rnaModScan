"""Measure the public CPU inference stage using an existing prepared HeLa bundle."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prediction.launcher import write_json
from utils.references.catalog import sha256


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def baseline_times(root):
    events = {}
    for line in (root / 'execution.log').read_text().splitlines():
        words = line.split()
        if len(words) == 3 and words[1] in ('Starting', 'Completed'):
            events[(words[1], words[2])] = datetime.fromisoformat(words[0])
    inference = (events[('Completed', 'infer')] - events[('Starting', 'infer')]).total_seconds()
    run = read_json(root / 'run.json')
    whole = (datetime.fromisoformat(run['finished_utc']) - datetime.fromisoformat(run['started_utc'])).total_seconds()
    return dict(gpu_inference_seconds=inference, full_gpu_workflow_seconds=whole,
                other_stages_seconds=whole - inference)


def plan(args):
    source = args.source.resolve()
    run = read_json(source / 'run.json')
    if run['status'] != 'complete':
        raise ValueError('The source prediction run is incomplete')
    prepared_path = source / 'nn_inputs/adenine'
    prepared = read_json(prepared_path / 'run.json')
    if prepared['status'] != 'complete' or prepared['schema'] != 'prediction_inputs_v9':
        raise ValueError('Use completed prepared prediction inputs')
    config = read_json(source / 'resolved_config.json')
    model = Path(config['model_directories']['adenine']).resolve()
    if not model.is_dir() or not (model / 'model.keras').is_file():
        raise FileNotFoundError(model)
    if args.threads < 1:
        raise ValueError('threads must be positive')
    report = dict(source=str(source), output=str(args.output.resolve()),
                  inputs=str(prepared_path), model=str(model),
                  input_run_sha256=sha256(prepared_path / 'run.json'),
                  model_sha256=sha256(model / 'model.keras'),
                  sites=sum(p['input_count'] for p in prepared['partitions']),
                  partitions=len(prepared['partitions']), batch_sites=config['nn']['batch_sites'],
                  tensorflow_intra_threads=args.threads, tensorflow_inter_threads=1,
                  timing_scope='Fresh public inference process: startup, model/input loading, restoration, NN and score output',
                  **baseline_times(source))
    if config['nn']['batch_sites'] != 16:
        raise ValueError('The audited GPU run used batch_sites=16')
    return report, config, prepared


def prepare_config(output, config, report):
    """Only the output and runtime settings change; inputs are linked read-only by use."""
    import yaml
    config = dict(config)
    config.pop('config_path', None)
    config['output_dir'] = str(output)
    config['nn'] = dict(config['nn'], device='cpu', cpu_threads=report['tensorflow_intra_threads'])
    config['recalibrate_bam'] = False
    path = output / 'config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    (output / 'nn_inputs').symlink_to(Path(report['source']) / 'nn_inputs', target_is_directory=True)
    return path


def completed_partitions(output, prepared):
    """An open Parquet writer has no footer yet, so count only completed partitions."""
    import pyarrow.parquet as pq
    from prediction.input_parquet import partition_directory
    rows, parts = 0, 0
    for partition in prepared['partitions']:
        path = output / 'scores/adenine' / partition_directory(partition['chrom'], partition['strand']) / 'scores.parquet'
        if not path.is_file():
            continue
        try:
            count = pq.ParquetFile(path).metadata.num_rows
        except (OSError, ValueError):
            continue
        if count == partition['input_count']:
            rows += count
            parts += 1
    return rows, parts


def compare_scores(source, output):
    """Compare the complete coordinate-keyed score outputs, allowing floating differences."""
    import csv
    import numpy as np
    import pyarrow.parquet as pq
    from utils.adenine import CLASSES
    old_root, new_root = source / 'scores/adenine', output / 'scores/adenine'
    old, new = read_json(old_root / 'run.json'), read_json(new_root / 'run.json')
    if new['runtime']['device'] != 'cpu' or new['runtime']['gpu_devices']:
        raise ValueError('The measured inference did not run exclusively on CPU')
    if old['model'] != new['model'] or old['input_run_sha256'] != new['input_run_sha256']:
        raise ValueError('CPU and GPU runs have different input/model fingerprints')
    old_parts = {(p['chrom'], p['strand']): p for p in old['partitions']}
    score_columns = ['score_' + name for name in CLASSES]
    total, changed, absolute_sum, maximum = 0, 0, 0.0, 0.0
    counts = {name: dict(gpu=0, cpu=0) for name in CLASSES}
    with (output / 'class_disagreements.tsv').open('x', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['row_id', 'gpu_class', 'cpu_class', *['gpu_' + c for c in score_columns],
                         *['cpu_' + c for c in score_columns]])
        for part in new['partitions']:
            original = old_parts.pop((part['chrom'], part['strand']))
            a = pq.ParquetFile(old_root / original['path']).read()
            b = pq.ParquetFile(new_root / part['path']).read()
            if not a['row_id'].equals(b['row_id']):
                raise ValueError('CPU/GPU row IDs or order differ')
            av = np.column_stack([a[c].to_numpy() for c in score_columns])
            bv = np.column_stack([b[c].to_numpy() for c in score_columns])
            differences = np.abs(av.astype('float64') - bv.astype('float64'))
            absolute_sum += float(differences.sum())
            maximum = max(maximum, float(differences.max(initial=0)))
            ac, bc = a['predicted_class'].to_pylist(), b['predicted_class'].to_pylist()
            total += len(ac)
            for i, (left, right) in enumerate(zip(ac, bc)):
                counts[left]['gpu'] += 1
                counts[right]['cpu'] += 1
                if left != right:
                    changed += 1
                    writer.writerow([a['row_id'][i].as_py(), left, right, *av[i], *bv[i]])
    if old_parts:
        raise ValueError('A CPU partition is missing')
    return dict(sites=total, class_changes=changed, argmax_agreement=1 - changed / total,
                max_absolute_probability_difference=maximum,
                mean_absolute_probability_difference=absolute_sum / (total * len(CLASSES)),
                class_counts=counts, runtime=new['runtime'])


def run(args):
    report, config, prepared = plan(args)
    output = Path(report['output'])
    output.mkdir(parents=True, exist_ok=False)
    config_path = prepare_config(output, config, report)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='-1', TF_NUM_INTRAOP_THREADS=str(args.threads),
                       TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS=str(args.threads),
                       OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    command = ['/usr/bin/time', '-f',
               'wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_status=%x\ncpu_percent=%P',
               '-o', str(output / 'resources.txt'), sys.executable, '-B', '-m', 'prediction',
               '--config', str(config_path), '--device', 'cpu', '--cpu-threads', str(args.threads), '--stage', 'infer']
    report.update(status='running', started_utc=utc_now(), supervisor_pid=os.getpid(),
                  command=command, hostname=platform.node(), platform=platform.platform(),
                  cpu_affinity=list(os.sched_getaffinity(0)), load_average=os.getloadavg(),
                  environment={k: environment[k] for k in ('CUDA_VISIBLE_DEVICES', 'TF_NUM_INTRAOP_THREADS',
                      'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                  source_code_sha256={p: sha256(ROOT / p) for p in
                      ['prediction/scoring.py', 'prediction/runtime.py', 'utils/nn/keras_site.py',
                       'utils/features/site_tensors.py', 'utils/features/compact_inputs.py']})
    write_json(output / 'run.json', report)
    (output / 'cpu_hardware.txt').write_text(subprocess.run(['lscpu'], capture_output=True, text=True).stdout)
    started = time.perf_counter()
    try:
        with (output / 'inference.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            report['child_pid'] = process.pid
            while True:
                try:
                    returncode = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    rows, parts = completed_partitions(output, prepared)
                    elapsed = time.perf_counter() - started
                    progress = dict(updated_utc=utc_now(), elapsed_seconds=elapsed,
                                    completed_sites=rows, total_sites=report['sites'],
                                    completed_partitions=parts, total_partitions=report['partitions'])
                    if rows:
                        progress['rough_remaining_seconds'] = elapsed * (report['sites'] / rows - 1)
                    write_json(output / 'progress.json', progress)
        report.update(process_wall_seconds=time.perf_counter() - started, returncode=returncode)
        if returncode:
            raise RuntimeError('CPU inference failed; see inference.log')
        resources = dict(line.split('=', 1) for line in (output / 'resources.txt').read_text().splitlines())
        report['resources'] = resources
        wall = float(resources['wall_seconds'])
        report.update(measured_cpu_inference_seconds=wall,
                      cpu_over_historical_gpu_time_ratio=wall / report['gpu_inference_seconds'],
                      estimated_cpu_full_workflow_seconds=report['other_stages_seconds'] + wall,
                      full_workflow_note='Estimate only: other stages are from the prior GPU workflow and were not rerun')
        write_json(output / 'run.json', dict(report, status='comparing_scores'))
        report['score_comparison'] = compare_scores(Path(report['source']), output)
        if sha256(Path(report['inputs']) / 'run.json') != report['input_run_sha256']:
            raise ValueError('The prepared input manifest changed')
        if sha256(Path(report['model']) / 'model.keras') != report['model_sha256']:
            raise ValueError('The model weights changed')
        report.update(status='complete', finished_utc=utc_now())
        write_json(output / 'run.json', report)
        print(json.dumps({k: v for k, v in report.items() if k not in ('source_code_sha256', 'command')}, indent=2), flush=True)
        return report
    except BaseException as error:
        report.update(status='failed', finished_utc=utc_now(), error=str(error))
        write_json(output / 'run.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'results/prediction_HeLa_WT_HEKonly_v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/benchmark_HeLa_CPU_20261109')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report, _, _ = plan(args)
    if args.check:
        print(json.dumps(dict(report, status='planned'), indent=2))
    elif args.background:
        output = args.output.resolve()
        launch_log = output.with_suffix('.launch.log')
        launch_record = output.with_suffix('.launch.json')
        for path in (output, launch_log, launch_record):
            if path.exists():
                raise FileExistsError(path)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--source', str(args.source.resolve()),
                   '--output', str(output), '--threads', str(args.threads)]
        with launch_log.open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launch = dict(status='starting', pid=process.pid, pgid=process.pid, command=command,
                      started_utc=utc_now(), output=str(output), log=str(launch_log))
        with launch_record.open('x') as handle:
            json.dump(launch, handle, indent=2)
        print(json.dumps(launch, indent=2))
    else:
        run(args)


if __name__ == '__main__':
    main()
