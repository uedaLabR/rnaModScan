"""Convert POD5 files independently; audit and skip corrupt-signal files only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import subprocess


CORRUPT_SIGNAL = 'Compressed signal data is corrupt'


def is_corrupt_signal(log_text):
    """Recognize input-signal corruption, not general zstd or storage errors."""
    if CORRUPT_SIGNAL in log_text:
        return True
    return ('Input data failed to decompress using zstd:' in log_text
            and 'Data corruption detected' in log_text)


def read_count(path):
    """Read header metadata only; unavailable counts stay null, never zero."""
    try:
        import pod5
        with pod5.Reader(path) as reader:
            return reader.num_reads
    except Exception:
        return None


def write_json(path, record):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def convert_one(path, source, out, blue_crab, slow5tools):
    relative = path.relative_to(source)
    key = path.stem + '_' + hashlib.sha256(str(relative).encode()).hexdigest()[:12]
    record_path = out / 'signal_records' / (key + '.json')
    log_path = out / 'signal_file_logs' / (key + '.log')
    output = out / 'blow5_parts_verified' / (key + '.blow5')
    temporary = out / 'signal_work' / (key + '.blow5')
    stat = path.stat()
    identity = dict(pod5=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous['input'] != identity:
            raise ValueError(f'Input changed; use a new conversion output: {path}')
        if previous['status'] == 'skipped_corrupt_signal':
            return previous
        if previous['status'] == 'success' and output.exists():
            with log_path.open('a') as log:
                subprocess.run([slow5tools, 'quickcheck', str(output)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            return previous
    record = dict(input=identity, reads=read_count(path), output=str(output), log=str(log_path))
    # Single-file blue-crab calls its synchronous converter, avoiding the
    # directory-mode multiprocessing queue that can remain waiting after error.
    with log_path.open('w') as log:
        result = subprocess.run([blue_crab, 'p2s', '-p', '1', '-t', '2', str(path),
                                 '-o', str(temporary)], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            result = subprocess.run([slow5tools, 'quickcheck', str(temporary)],
                                    stdout=log, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        temporary.replace(output)
        record.update(status='success', returncode=0)
    else:
        corrupt = is_corrupt_signal(log_path.read_text(errors='replace'))
        record.update(status='skipped_corrupt_signal' if corrupt else 'failed',
                      returncode=result.returncode)
        # Partial outputs remain outside the directory passed to slow5tools merge.
    write_json(record_path, record)
    print(f"{record['status']}: {path}", flush=True)
    return record


def convert_files(source, out, *, workers=2, blue_crab='blue-crab', slow5tools='slow5tools'):
    source, out = Path(source).resolve(), Path(out).resolve()
    files = sorted(source.rglob('*.pod5'))
    if not files or workers < 1:
        raise ValueError('Require POD5 files and at least one worker')
    for directory in ('blow5_parts_verified', 'signal_records', 'signal_file_logs', 'signal_work'):
        (out / directory).mkdir(parents=True, exist_ok=True)
    inventory_path = out / 'signal_input_files.json'
    inventory = [str(path) for path in files]
    if inventory_path.exists() and json.loads(inventory_path.read_text()) != inventory:
        raise ValueError('POD5 file list changed; use a new conversion output')
    inventory_path.write_text(json.dumps(inventory, indent=2) + '\n')
    def convert(path):
        return convert_one(path, source, out, blue_crab, slow5tools)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(files), workers):
            batch = list(pool.map(convert, files[start:start + workers]))
            records.extend(batch)
            if any(row['status'] == 'failed' for row in batch):
                break
    with (out / 'signal_conversion.tsv').open('w', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['pod5', 'status', 'reads', 'returncode', 'log', 'output'])
        for row in records:
            writer.writerow([row['input']['pod5'], row['status'], row['reads'],
                             row['returncode'], row['log'], row['output']])
    success = [r for r in records if r['status'] == 'success']
    skipped = [r for r in records if r['status'] == 'skipped_corrupt_signal']
    failed = [r for r in records if r['status'] == 'failed']
    counts = [r['reads'] for r in skipped]
    summary = dict(input_files=len(files), processed_files=len(records),
                   converted_files=len(success), skipped_files=len(skipped),
                   failed_files=len(failed), skipped_reads_known=sum(n for n in counts if n is not None),
                   skipped_files_unknown_read_count=sum(n is None for n in counts),
                   policy='Exclude entire corrupt POD5, including any readable reads in that file',
                   status='complete' if success and not failed else 'failed')
    write_json(out / 'signal_conversion.json', summary)
    if failed or not success:
        raise RuntimeError('No usable files, or errors other than corrupt signal; inspect signal_conversion.tsv')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pod5-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(convert_files(args.pod5_dir, args.output_dir, workers=args.workers), indent=2))


if __name__ == '__main__':
    main()
