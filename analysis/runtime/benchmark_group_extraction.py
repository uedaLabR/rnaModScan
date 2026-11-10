"""Time fresh CPU extraction on an explicit small site list; never start inference."""
import argparse
from pathlib import Path
import json
import time

import pyarrow.parquet as pq

from utils.features.site_bag_pipeline import extract_site_bag_sample
from utils.features.compact_inputs import restore_read_inputs, tensor_column


def load_rows(directory):
    report = json.loads((directory / 'run.json').read_text())
    rows = {}
    for item in report['parquet_files']['sites']:
        table = pq.ParquetFile(directory / 'sites' / item['path']).read()
        for index, row in enumerate(table.to_pylist()):
            row['read_features'] = restore_read_inputs(table[tensor_column(table.schema)][index],
                                                      report['window_flank'] * 2 + 1).tolist()
            row.pop('read_inputs', None)
            if row['row_id'] in rows:
                raise ValueError('Duplicate row ID')
            rows[row['row_id']] = row
    return rows


def benchmark(root):
    inputs = json.loads((root / 'input.json').read_text())
    timings = {}
    expected = load_rows(root / 'baseline')
    for workers in (1, 2, 4):
        directory = root / f'integer_workers_{workers}'
        started = time.perf_counter()
        extract_site_bag_sample(root / 'sites.tsv', inputs['sample'], inputs['fasta'], directory,
            None, mode='inference', bam_threads=1, group_workers=workers, group_batch_size=1)
        elapsed = time.perf_counter() - started
        actual = load_rows(directory)
        assert actual == expected
        timings[str(workers)] = elapsed
        print('workers', workers, 'seconds', elapsed, 'all site rows identical', flush=True)
    sizes = {}
    for name in ('baseline', 'integer_workers_1'):
        directory = root / name
        source = json.loads((directory / 'run.json').read_text())
        sizes[name] = sum(item['size_bytes'] for item in source['parquet_files']['sites'])
    # Read conversion timings include Parquet decode, not just array arithmetic.
    decoding = {}
    for name in ('baseline', 'integer_workers_1'):
        started = time.perf_counter()
        load_rows(root / name)
        decoding[name] = time.perf_counter() - started
    report = dict(site_count=len(expected), worker_seconds=timings, parquet_bytes=sizes,
                  decode_to_python_rows_seconds=decoding,
                  equality='all raw/NN/site rows identical to baseline; numerical arrays float32 bit-identical',
                  caveat='Small real-data subset, concurrent host jobs, no guarantee of whole-run speedup or compression ratio. No Numba dependency.')
    (root / 'integer_benchmark_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-directory', type=Path, required=True,
                        help='Existing sites.tsv, input.json and baseline/run.json; fresh workers_N outputs required')
    benchmark(parser.parse_args().input_directory)
