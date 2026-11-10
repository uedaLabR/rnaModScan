"""CPU/GPU inference: read prepared Parquet tensors and write compact scores."""
import json
import math
from pathlib import Path

from prediction.input_parquet import checked_partition_path, partition_directory
from prediction.model_bundle import load_model_bundle, model_fingerprint
from utils.features.parquet import require_arrow
from utils.io.artifacts import completed_run
from utils.references.catalog import sha256
from utils.adenine import CLASSES
from prediction.runtime import configure_inference, validate_runtime
from utils.features.compact_inputs import SITE_SCHEMAS, tensor_columns


def partition_input_batches(directory, partition, window, max_reads, batch_sites):
    """Read the four saved tensors directly from the chromosome/strand shards."""
    import pyarrow.compute as pc
    from utils.features.site_tensors import site_batch_inputs
    _, pq = require_arrow()
    if partition.get('layout') not in SITE_SCHEMAS:
        raise ValueError('Use prepared site-bag Parquet v3/v4')
    for shard in partition['input_shards']:
        path = checked_partition_path(directory, shard['path'], shard['size_bytes'])
        parquet = pq.ParquetFile(path)
        columns = tensor_columns(parquet.schema_arrow) + ['retained_depth']
        for batch in parquet.iter_batches(batch_size=batch_sites, columns=columns):
            batch = batch.filter(pc.greater(batch.column('retained_depth'), 0))
            if batch.num_rows:
                records = [dict(row_id=value) for value in batch.column('row_id').to_pylist()]
                yield records, site_batch_inputs(batch, window, max_reads)


def score_prepared_inputs(input_directory, model_directory, out, *, batch_sites=4,
                          device='auto', cpu_threads=None):
    """No BAM, GTF, repeat annotation, candidate TSV or reference DB is opened."""
    if not isinstance(batch_sites, int) or batch_sites < 1:
        raise ValueError('batch_sites must be positive')
    validate_runtime(device, cpu_threads)
    pa, pq = require_arrow()
    input_directory, model_directory, out = Path(input_directory), Path(model_directory), Path(out)
    prepared = completed_run(input_directory)
    if prepared.get('schema') != 'prediction_inputs_v9':
        raise ValueError('Use a completed prepared NN input bundle')
    model = None
    runtime = dict(requested_device=device, device='unused', cpu_threads=cpu_threads, gpu_devices=[])
    if any(part['input_count'] for part in prepared['partitions']):
        runtime = configure_inference(device, cpu_threads)
        from prediction.predict import _load_model
        model, _, model_run = _load_model(model_directory)
    else:
        _, model_run = load_model_bundle(model_directory)
    fingerprint = model_fingerprint(model_directory, model_run)
    if (prepared['target'] != model_run['target'] or prepared['model'] != fingerprint
            or prepared['input_schema'] != model_run['input_schema']
            or prepared['threshold'] != model_run['threshold']):
        raise ValueError('Prepared NN inputs belong to another final model')
    joint = prepared['target'] == 'adenine'
    fields = [('row_id', pa.string())]
    if joint:
        fields += [('score_' + name, pa.float32()) for name in CLASSES]
        fields += [('predicted_class', pa.string())]
    else:
        fields += [('keep_score', pa.float32()), ('nn_keep', pa.int8())]
    schema = pa.schema(fields,
                       metadata={b'schema_version': b'prediction_scores_v1'})
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE.txt').write_text('NN scoring is not complete.', encoding='utf-8')
    manifest = []
    try:
        for partition in prepared['partitions']:
            chrom, strand = partition['chrom'], partition['strand']
            relative = partition_directory(chrom, strand) / 'scores.parquet'
            result_path = out / relative
            result_path.parent.mkdir(parents=True)
            count = 0
            with pq.ParquetWriter(result_path, schema, compression='zstd') as writer:
                for records, inputs in partition_input_batches(input_directory, partition,
                        prepared['input_schema']['window'], prepared['input_schema'].get('max_reads'), batch_sites):
                    values = model(inputs, training=False).numpy()
                    if not joint:
                        values = values.reshape(-1)
                    if len(values) != len(records):
                        raise ValueError('NN score count differs from prepared input count')
                    output = []
                    for record, value in zip(records, values):
                        if joint:
                            scores = [float(score) for score in value]
                            if len(scores) != 3 or any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
                                raise ValueError('Joint model returned invalid class scores')
                            if not math.isclose(sum(scores), 1, abs_tol=1e-5):
                                raise ValueError('Joint softmax scores must sum to one')
                            predicted = CLASSES[max(range(3), key=lambda index: scores[index])]
                            output.append(dict(row_id=record['row_id'], predicted_class=predicted,
                                               **{'score_' + name: score for name, score in zip(CLASSES, scores)}))
                            continue
                        score = float(value)
                        if not math.isfinite(score) or not 0 <= score <= 1:
                            raise ValueError('Model returned an invalid probability score')
                        output.append(dict(row_id=record['row_id'], keep_score=score,
                                           nn_keep=int(score >= model_run['threshold'])))
                    writer.write_table(pa.Table.from_pylist(output, schema=schema))
                    count += len(output)
            if count != partition['input_count']:
                raise ValueError('Prepared partition input count differs from manifest')
            manifest.append(dict(chrom=chrom, strand=strand, path=str(relative), rows=count,
                                 size_bytes=result_path.stat().st_size))
        report = dict(status='complete', schema='prediction_scores_v1', target=prepared['target'],
                      input_run_sha256=sha256(input_directory / 'run.json'), model=fingerprint,
                      threshold=model_run['threshold'], partitions=manifest, runtime=runtime)
        if joint:
            report.update(class_names=list(CLASSES), decision_rule='argmax; threshold unused')
        (out / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as error:
        (out / 'FAILED.txt').write_text(str(error), encoding='utf-8')
        raise
    finally:
        if model is not None:
            import tensorflow as tf
            tf.keras.backend.clear_session()
