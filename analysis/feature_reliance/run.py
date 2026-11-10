"""Measure frozen-model reliance using prepared site bags, without BAM or fitting."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONDITIONS = ('sequence', 'repeat', 'neighbor_ml', 'quality', 'sequence_repeat', 'center_ml', 'all_ml')
ROW_FIELDS = ('row_id', 'sample_id', 'chrom', 'start', 'end', 'strand', 'modification',
              'label', 'reference_sequence', 'retained_depth', 'eligible_depth', 'group_id')
SNAPSHOT_FILES = tuple(ROOT / name for name in (
    'analysis/feature_reliance/run.py', 'analysis/feature_reliance/metrics.py',
    'analysis/feature_reliance/perturbations.py', 'analysis/feature_reliance/protocol.json',
    'utils/nn/keras_site.py', 'utils/nn/inputs.py', 'utils/features/site_dataset.py',
    'utils/features/compact_inputs.py', 'utils/features/dataset.py', 'utils/features/parquet.py'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, record):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def source_records(args):
    fold = args.training_root / 'adenine' / f'fold_{args.fold}'
    run_path = fold / 'run.json'
    run = json.loads(run_path.read_text())
    if run['status'] != 'complete' or run['feature_set'] != 'sequence_reads_repeat18_v1':
        raise ValueError('Use completed current-schema CV model outputs')
    if run['model']['window'] != 19 or run['model']['max_reads'] != 300:
        raise ValueError('Expected the frozen 19nt / 300-read classifier')
    index = args.index or Path(run['index_path'])
    split = Path(run['split_files']['test']['path'])
    if sha(index) != run['index_sha256'] or sha(split) != run['split_files']['test']['sha256']:
        raise ValueError('The frozen training index or held-out manifest changed')
    paths = [index, split, run_path, fold / 'train/best.keras', fold / 'train/settings.json',
             fold / 'test/predictions.tsv']
    protocol = Path(__file__).with_name('protocol.json')
    paths.extend(SNAPSHOT_FILES)
    for source in run['feature_sources']:
        path = Path(source['path']) / 'run.json'
        if sha(path) != source['run_sha256']:
            raise ValueError('A frozen extraction manifest changed')
        paths.append(path)
    records = [dict(path=str(path.resolve()), sha256=sha(path), size_bytes=path.stat().st_size)
               for path in paths]
    return dict(fold_directory=str(fold.resolve()), index=str(index.resolve()),
        split=str(split.resolve()), checkpoint=str((fold / 'train/best.keras').resolve()),
        model=run['model'], feature_set=run['feature_set'], protocol=str(protocol),
        protocol_sha256=sha(protocol), files=records)


def snapshot_code(out):
    records = []
    for source in SNAPSHOT_FILES:
        relative = source.relative_to(ROOT)
        destination = out / 'code_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha(source) != sha(destination):
            raise ValueError('Code snapshot changed while copying')
        records.append(dict(path=str(relative), sha256=sha(destination)))
    return records


def check_sources(records):
    for record in records:
        path = Path(record['path'])
        if path.stat().st_size != record['size_bytes'] or sha(path) != record['sha256']:
            raise ValueError('Analysis source changed: ' + str(path))


def progress(out, phase, done, total, started, **extra):
    now = time.time()
    write_json(out / 'progress.json', dict(status='running', phase=phase,
        completed_sites=done, total_sites=total, elapsed_seconds=now - started,
        updated_unix=now, heartbeat_unix=now, **extra))


def site_batches(dataset, order, count):
    for start in range(0, len(order), count):
        indices = order[start:start + count]
        rows = [dataset.rows[int(i)] for i in indices]
        inputs, labels = dataset._batch(rows)
        yield indices, rows, inputs, labels


def prepare(args, out, sources, started):
    from utils.features.site_dataset import SiteBagBatches
    from analysis.feature_reliance.perturbations import PoolBank
    dataset = SiteBagBatches(sources['index'], sources['split'], batch_sites=args.batch_sites,
                            max_reads=300, sampling_seed=42)
    original_count = len(dataset.rows)
    if args.max_sites is not None and args.max_sites < len(dataset.rows):
        selected = np.sort(np.random.default_rng(args.selection_seed).choice(
            len(dataset.rows), args.max_sites, replace=False))
        dataset.rows = [dataset.rows[int(i)] for i in selected]
    rows = dataset.rows
    # Visit each physical row group once; output slots retain the frozen index order.
    order = np.asarray(sorted(range(len(rows)), key=lambda i: (rows[i]['site_parquet'],
        int(rows[i]['site_row_group']), int(rows[i]['site_row_offset']))), dtype=np.int64)
    tokens = np.empty((len(rows), 17), dtype=np.int32)
    repeat_mask = np.empty((len(rows), 19), dtype=bool)
    bank = PoolBank(window=19, capacity=args.pool_capacity, seed=args.pool_seed)
    completed = 0
    for indices, batch_rows, inputs, _ in site_batches(dataset, order, args.pool_batch_sites):
        tokens[indices] = inputs['sequence_tokens']
        repeat_mask[indices] = inputs['sequence_repeat_mask']
        bank.update(inputs, batch_rows)
        completed += len(indices)
        progress(out, 'prepare_donor_pools', completed, len(rows), started)
    bank.finalize()
    bank.save(out / 'donor_pools.npz')
    np.savez_compressed(out / 'reference_inputs.npz', sequence_tokens=tokens,
                        sequence_repeat_mask=repeat_mask)
    with (out / 'sites.tsv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, delimiter='\t',
                                lineterminator='\n', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    preparation = dict(status='complete', sites=len(rows), full_test_sites=original_count,
        selection='all held-out test observations' if args.max_sites is None else
                  'uniform label-blind sampling without replacement; original index order retained',
        selection_seed=args.selection_seed, max_sites=args.max_sites,
        pilot=args.max_sites is not None, class_counts=dict(Counter(row['label'] for row in rows)),
        pool_report=bank.report(), physical_row_group_order=True,
        read_sampling='Existing per-site reservoir, at most300 real reads; no resampling',
        source_files=sources['files'], seconds=time.time() - started)
    write_json(out / 'prepared.json', preparation)
    return dataset, order, tokens, repeat_mask, bank, preparation


def configure_model(args, sources):
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.cpu_threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    physical = tf.config.list_physical_devices('GPU')
    if args.device == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
        physical = []
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.device == 'gpu' and not physical:
        raise RuntimeError('GPU requested but unavailable; use a CPU mode in a fresh output directory')
    import utils.nn.keras_site  # Register the frozen custom layers before loading.
    model = tf.keras.models.load_model(sources['checkpoint'], compile=False)
    if model.count_params() != 23731:
        raise ValueError('The frozen lightweight model parameter count changed')
    model.trainable = False
    signature = dict(sequence_tokens=tf.TensorSpec((None, 17), tf.int32),
        sequence_repeat_mask=tf.TensorSpec((None, 19), tf.bool),
        read_features=tf.TensorSpec((None, 300, 19, 18), tf.float32),
        read_mask=tf.TensorSpec((None, 300), tf.bool))
    predictor = tf.function(lambda inputs: model(inputs, training=False), input_signature=[signature])
    runtime = dict(tensorflow_version=tf.__version__, device='gpu' if physical else 'cpu',
        requested_device=args.device, visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        physical_gpus=[gpu.name for gpu in physical], cpu_threads=args.cpu_threads,
        inter_operation_threads=1, parameter_count=model.count_params(), training=False,
        graph_inference=True, source_batch_sites=args.batch_sites,
        pool_batch_sites=args.pool_batch_sites, inference_batch_sites=args.inference_batch_sites)
    return model, predictor, runtime


def infer(args, out, sources, dataset, order, tokens, repeat_mask, bank, started):
    from analysis.feature_reliance.perturbations import apply_reference, make_site_donors
    model, predictor, runtime = configure_model(args, sources)
    variants = [dict(condition='baseline', seed=None)]
    variants.extend(dict(condition=condition, seed=seed)
                    for condition in CONDITIONS for seed in args.seeds)
    generators = [None] + [np.random.default_rng(np.random.SeedSequence(
        [entry['seed'], CONDITIONS.index(entry['condition'])])) for entry in variants[1:]]
    donors = {seed: make_site_donors(dataset.rows, seed) for seed in args.seeds}
    scores = np.lib.format.open_memmap(out / 'scores.npy', mode='w+', dtype=np.float32,
                                     shape=(len(variants), len(dataset.rows), 3))
    scores[:] = np.nan
    first_probe = None
    identity_control = dict(anchor_chunks=0, anchor_site_comparisons=0,
        max_anchor_probability_difference=0.0, anchor_class_changes=0,
        baseline_copy_site_comparisons=0, max_baseline_copy_probability_difference=0.0,
        baseline_copy_class_changes=0, baseline_copy_exact=True,
        tolerance=1e-6, policy='One baseline anchor per constant-sized model call; last call padded with baseline inputs')
    completed = 0
    inference_started = time.time()
    for indices, batch_rows, inputs, _ in site_batches(dataset, order, args.batch_sites):
        prepared_variants = [inputs]
        for i, entry in enumerate(variants[1:], 1):
            condition = entry['condition']
            if condition in ('sequence', 'repeat', 'sequence_repeat'):
                changed = apply_reference(inputs, tokens, repeat_mask,
                    donors[entry['seed']][indices], condition)
            else:
                changed = bank.replace(inputs, batch_rows, condition, generators[i])
            if not np.array_equal(changed['read_mask'], inputs['read_mask']):
                raise ValueError('An intervention changed the real-read mask')
            prepared_variants.append(changed)
        if first_probe is None:
            eager = model(inputs, training=False).numpy()
            graph = predictor(inputs).numpy()
            identity = {name: value.copy() for name, value in inputs.items()}
            copied = predictor(identity).numpy()
            delta = float(np.max(np.abs(eager - graph)))
            if not np.allclose(eager, graph, atol=5e-3, rtol=1e-4):
                raise ValueError('Eager/graph baseline predictions differ unexpectedly')
            if not np.array_equal(graph, copied):
                raise ValueError('A copied identity intervention changed predictions')
            first_probe = dict(sites=len(indices), eager_graph_max_probability_difference=delta,
                eager_graph_class_changes=int(np.sum(eager.argmax(1) != graph.argmax(1))),
                copied_identity_exact=True)
            write_json(out / 'identity_probe.json', first_probe)
        per_call = args.inference_batch_sites // len(indices)
        actual_slots = per_call - 1
        first_anchor = None
        for start in range(0, len(variants), actual_slots):
            batch = prepared_variants[start:start + actual_slots]
            parts = [inputs, *batch]
            parts.extend([inputs] * (per_call - len(parts)))
            stacked = {name: np.concatenate([part[name] for part in parts], axis=0) for name in inputs}
            predicted = predictor(stacked).numpy()
            if predicted.shape != (per_call * len(indices), 3):
                raise ValueError('Unexpected score batch shape')
            anchor = predicted[:len(indices)]
            identity_control['anchor_chunks'] += 1
            if first_anchor is None:
                first_anchor = anchor.copy()
                if completed == 0:
                    first_probe.update(stacked_batch_sites=len(stacked['read_mask']),
                        eager_stacked_anchor_max_probability_difference=float(np.max(np.abs(eager - anchor))),
                        eager_stacked_anchor_class_changes=int(np.sum(eager.argmax(1) != anchor.argmax(1))))
                    write_json(out / 'identity_probe.json', first_probe)
            else:
                delta = float(np.max(np.abs(first_anchor - anchor)))
                changes = int(np.sum(first_anchor.argmax(1) != anchor.argmax(1)))
                identity_control['anchor_site_comparisons'] += len(indices)
                identity_control['max_anchor_probability_difference'] = max(
                    identity_control['max_anchor_probability_difference'], delta)
                identity_control['anchor_class_changes'] += changes
                if delta > identity_control['tolerance'] or changes:
                    raise ValueError('Baseline anchors drifted across intervention chunks')
            for local in range(len(batch)):
                current = predicted[(local + 1) * len(indices):(local + 2) * len(indices)]
                if start + local == 0:
                    delta = float(np.max(np.abs(current - anchor)))
                    changes = int(np.sum(current.argmax(1) != anchor.argmax(1)))
                    identity_control['baseline_copy_site_comparisons'] += len(indices)
                    identity_control['max_baseline_copy_probability_difference'] = max(
                        identity_control['max_baseline_copy_probability_difference'], delta)
                    identity_control['baseline_copy_class_changes'] += changes
                    identity_control['baseline_copy_exact'] &= bool(np.array_equal(current, anchor))
                    if delta > identity_control['tolerance'] or changes:
                        raise ValueError('The stacked baseline identity copy changed predictions')
                scores[start + local, indices] = current
        completed += len(indices)
        progress(out, 'frozen_model_inference', completed, len(dataset.rows), started,
                 variants=len(variants), inference_seconds=time.time() - inference_started)
    scores.flush()
    if not np.isfinite(scores).all():
        raise ValueError('Some site/condition scores were not written')
    runtime['inference_seconds'] = time.time() - inference_started
    runtime['stacked_identity_control'] = identity_control
    return variants, scores, runtime, first_probe


def compare_saved_baseline(sources, rows, scores):
    original = Path(sources['fold_directory']) / 'test/predictions.tsv'
    lookup = {row['row_id']: i for i, row in enumerate(rows)}
    selected = np.full((len(rows), 3), np.nan, dtype=np.float32)
    with original.open() as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            i = lookup.get(row['row_id'])
            if i is not None:
                selected[i] = [float(row['score_' + name])
                               for name in ('m6A', 'inosine', 'false_positive')]
    if not np.isfinite(selected).all():
        raise ValueError('The saved test predictions are missing cohort row IDs')
    return dict(sites=len(rows), max_probability_difference=float(np.max(np.abs(scores - selected))),
        mean_probability_difference=float(np.mean(np.abs(scores - selected))),
        class_changes=int(np.sum(scores.argmax(1) != selected.argmax(1))),
        interpretation='Diagnostic only: primary drops use the baseline recomputed in this analysis process')


def run(args):
    sources = source_records(args)
    if args.check:
        print(json.dumps(dict(status='checked', output=str(args.output), sources=sources,
            conditions=list(CONDITIONS), seeds=args.seeds, max_sites=args.max_sites), indent=2))
        return
    if args.background:
        launch_background(args)
        return
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_bytes(Path(sources['protocol']).read_bytes())
    code_snapshot = snapshot_code(out)
    (out / 'INCOMPLETE.txt').write_text('Frozen-model feature-reliance analysis is incomplete.\n')
    started = time.time()
    write_json(out / 'run.json', dict(status='running', fold=args.fold, started_unix=started,
        pid=os.getpid(), phase='preparation', sources=sources))
    try:
        dataset, order, tokens, repeat_mask, bank, preparation = prepare(args, out, sources, started)
        variants, scores, runtime, identity = infer(args, out, sources, dataset, order,
                                                   tokens, repeat_mask, bank, started)
        from analysis.feature_reliance.metrics import evaluate_variants
        evaluations = evaluate_variants(dataset.rows, variants, scores)
        comparison = compare_saved_baseline(sources, dataset.rows, scores[0])
        check_sources(sources['files'])
        report = dict(status='complete', schema='frozen_feature_reliance_v1', fold=args.fold,
            started_unix=started, finished_unix=time.time(), elapsed_seconds=time.time() - started,
            started_utc=datetime.fromtimestamp(started, timezone.utc).isoformat(),
            conditions=list(CONDITIONS), variant_seeds=args.seeds,
            cohort=preparation, sources=sources, runtime=runtime, identity_probe=identity,
            saved_baseline_comparison=comparison, evaluations=evaluations,
            score_file='scores.npy', score_shape=list(scores.shape), row_file='sites.tsv',
            source_files_unchanged=True, final_pool_report=bank.report(),
            code_snapshot=code_snapshot,
            code_sha256={str(path.relative_to(ROOT)): sha(path) for path in SNAPSHOT_FILES},
            interpretation='Frozen-model conditional empirical replacements against operational weak labels; not causal importance or independent accuracy. Feature drops are not additive fractions.',
            no_training_or_BAM=True)
        write_json(out / 'run.json', report)
        write_json(out / 'progress.json', dict(status='complete', phase='finished',
            completed_sites=len(dataset.rows), total_sites=len(dataset.rows),
            updated_unix=time.time(), elapsed_seconds=time.time() - started))
        (out / 'INCOMPLETE.txt').unlink()
        print(json.dumps(dict(status='complete', output=str(out), sites=len(dataset.rows),
                             variants=len(variants), seconds=report['elapsed_seconds']), indent=2))
    except BaseException as error:
        write_json(out / 'run.json', dict(status='failed', fold=args.fold, error=str(error),
            started_unix=started, finished_unix=time.time(), sources=sources))
        (out / 'FAILED.txt').write_text(str(error) + '\n')
        raise


def launch_background(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    launch = args.output.with_name(args.output.name + '.launch.json')
    log = args.output.with_name(args.output.name + '.launch.log')
    if launch.exists() or log.exists():
        raise FileExistsError('Preserve earlier launch records; use a fresh output directory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-B', str(Path(__file__).resolve())]
    command.extend(value for value in sys.argv[1:] if value != '--background')
    with log.open('x') as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(status='launched', pid=process.pid, pgid=process.pid,
        started_utc=datetime.now(timezone.utc).isoformat(), command=command,
        output=str(args.output), log=str(log))
    write_json(launch, record)
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, default=ROOT / 'results/training_v10/trials/lr_0p0003_5fold')
    parser.add_argument('--index', type=Path)
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--output', '--out', type=Path)
    parser.add_argument('--device', choices=('auto', 'cpu', 'gpu'), default='gpu')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--batch-sites', type=int, default=16)
    parser.add_argument('--pool-batch-sites', type=int, default=256)
    parser.add_argument('--inference-batch-sites', type=int, default=128)
    parser.add_argument('--pool-capacity', type=int, default=4096)
    parser.add_argument('--pool-seed', type=int, default=42)
    parser.add_argument('--selection-seed', type=int, default=419)
    parser.add_argument('--seeds', default='17,43,101')
    parser.add_argument('--max-sites', type=int, help='Label-blind pilot only; omitted means every held-out site')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    args.seeds = [int(value) for value in args.seeds.split(',')]
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('Provide distinct randomization seeds')
    if min(*args.seeds, args.pool_seed, args.selection_seed, args.gpu) < 0:
        parser.error('Seeds and GPU index must be nonnegative')
    if min(args.batch_sites, args.pool_batch_sites, args.inference_batch_sites,
           args.pool_capacity, args.cpu_threads) < 1:
        parser.error('Batch sizes, capacity and CPU thread count must be positive')
    if args.inference_batch_sites < 2 * args.batch_sites:
        parser.error('The inference cap needs at least two source batches (anchor plus intervention)')
    if args.max_sites is not None and args.max_sites < 1:
        parser.error('--max-sites must be positive')
    args.training_root = args.training_root.resolve()
    args.index = args.index.resolve() if args.index else None
    args.output = (args.output or ROOT / f'results/feature_reliance_fold{args.fold}_20261109').resolve()
    run(args)


if __name__ == '__main__':
    main()
