"""Bounded-memory, typed, Zstandard-compressed feature datasets."""
import json
import bisect
from pathlib import Path
from urllib.parse import quote


def require_arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError('Parquet extraction requires pyarrow (pip install -r training/requirements-bam.txt)') from exc
    if not pa.Codec.is_available('zstd'):
        raise RuntimeError('This pyarrow build does not support Zstandard compression')
    return pa, pq


def feature_schema(flank, modifications):
    pa, _ = require_arrow()
    window = 2 * flank + 1
    per_pos = lambda dtype: pa.list_(dtype, window)
    per_mod = lambda dtype: per_pos(pa.list_(dtype, len(modifications)))
    schema = pa.schema([
        ('row_id', pa.string()), ('read_id', pa.string()), ('alignment_start', pa.int64()),
        ('cigar', pa.string()), ('mapq', pa.uint8()), ('query_length', pa.int32()),
        ('splice_distance_before', pa.uint32()), ('splice_distance_after', pa.uint32()),
        ('mm_skip_modes', pa.list_(pa.string(), len(modifications))),
        ('base', per_pos(pa.string())), ('alignment_state', per_pos(pa.string())),
        ('base_quality', per_pos(pa.uint8())), ('mismatch', per_pos(pa.uint8())),
        ('insertion_length_after', per_pos(pa.uint32())), ('ml_byte', per_mod(pa.uint8())),
        ('modification_applicable', per_mod(pa.bool_())), ('modification_observed', per_mod(pa.bool_())),
    ])
    return schema.with_metadata({b'schema_version': b'read_features_v3',
                                 b'modification_channels': json.dumps(modifications).encode(),
                                 b'window_flank': str(flank).encode(),
                                 b'orientation': b'RNA 5-prime to 3-prime',
                                 b'splice_distance': b'nearest full-read CIGAR N; RNA before/after; query bases including I, excluding D/N; adjacent=0; absent=null',
                                 b'ml_encoding': b'byte 0..255; probability interval [N/256,(N+1)/256); null != 0'})


def site_schema():
    pa, _ = require_arrow()
    return pa.schema([
        ('row_id', pa.string()), ('sample_id', pa.string()), ('chrom', pa.string()),
        ('start', pa.int64()), ('end', pa.int64()), ('strand', pa.string()),
        ('modification', pa.string()), ('reference_sequence', pa.string()),
        ('eligible_depth', pa.int64()), ('retained_depth', pa.int32()),
        ('read_offset', pa.int64()),
        ('central_deletions', pa.int64()), ('central_refskips', pa.int64()),
    ])


class ShardedParquetWriter:
    """Write flat read-site rows in row groups; bounded rows per file and buffer.

    Partial filenames are only renamed after the footer has been written. The
    enclosing extraction additionally requires a completed run.json.
    """
    def __init__(self, directory, schema, *, batch_rows=1024, file_rows=100000):
        if batch_rows < 1 or file_rows < 1:
            raise ValueError('Parquet row limits must be positive')
        self.pa, self.pq = require_arrow()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.schema, self.batch_rows, self.file_rows = schema, batch_rows, file_rows
        self.buffer, self.files, self.writer = [], [], None
        self.rows_in_file, self.total_rows = 0, 0

    def append(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= self.batch_rows:
            self.flush()

    def flush(self):
        while self.buffer:
            if self.writer is None:
                self.partial = self.directory / f'part-{len(self.files):06d}.parquet.partial'
                self.writer = self.pq.ParquetWriter(str(self.partial), self.schema, compression='zstd',
                                                  use_dictionary=True, write_statistics=True)
                self.rows_in_file = 0
            count = min(len(self.buffer), self.file_rows - self.rows_in_file)
            table = self.pa.Table.from_pylist(self.buffer[:count], schema=self.schema)
            self.writer.write_table(table, row_group_size=self.batch_rows)
            del self.buffer[:count]
            self.rows_in_file += count
            self.total_rows += count
            if self.rows_in_file == self.file_rows:
                self._finish_file()

    def _finish_file(self):
        self.writer.close()
        dest = self.partial.with_suffix('')
        self.partial.rename(dest)
        self.files.append(dict(path=dest.name, rows=self.rows_in_file, size_bytes=dest.stat().st_size))
        self.writer = None

    def close(self):
        self.flush()
        if self.writer is not None:
            self._finish_file()
        if not self.files:
            path = self.directory / 'part-000000.parquet'
            self.pq.write_table(self.pa.Table.from_pylist([], schema=self.schema), str(path), compression='zstd')
            self.files.append(dict(path=path.name, rows=0, size_bytes=path.stat().st_size))

    def abort(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class PartitionedParquetWriter:
    """Keep chromosome/strand files separate, buffering only the current contig.

    Logical offsets follow site emission order. Separate strand buffers can
    reorder physical rows, so read_ranges records contiguous logical ranges.
    """
    def __init__(self, directory, schema, *, batch_rows=1024, file_rows=100000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.schema, self.batch_rows, self.file_rows = schema, batch_rows, file_rows
        self.writers, self.partitions = {}, {}
        self.current, self.current_chrom = None, None
        self.ranges, self.read_ranges, self.files = [], [], []
        self.total_rows = 0

    def set_partition(self, chrom, strand):
        if not chrom or strand not in ('+', '-'):
            raise ValueError('Parquet partitions need chromosome and explicit strand')
        if self.current_chrom is not None and self.current_chrom != chrom:
            for key, writer in self.writers.items():
                if key[0] == self.current_chrom:
                    writer.close()
        key = chrom, strand
        if key not in self.writers:
            name = Path('chrom=' + quote(chrom, safe='')) / ('strand=' + ('plus' if strand == '+' else 'minus'))
            self.partitions[key] = name
            self.writers[key] = ShardedParquetWriter(self.directory / name, self.schema,
                batch_rows=self.batch_rows, file_rows=self.file_rows)
        self.current, self.current_chrom = key, chrom

    def append(self, row):
        if self.current is None:
            raise ValueError('Set the chromosome/strand partition before appending')
        writer = self.writers[self.current]
        physical_offset = writer.total_rows + len(writer.buffer)
        if self.ranges and self.ranges[-1]['partition'] == self.current:
            self.ranges[-1]['rows'] += 1
        else:
            self.ranges.append(dict(partition=self.current, offset=self.total_rows,
                                    partition_offset=physical_offset, rows=1))
        writer.append(row)
        self.total_rows += 1

    def close(self):
        file_ends = {}
        for key, writer in self.writers.items():
            writer.close()
            ends, total = [], 0
            for item in writer.files:
                total += item['rows']
                ends.append(total)
                self.files.append(dict(item, path=str(self.partitions[key] / item['path']),
                                       chrom=key[0], strand=key[1]))
            file_ends[key] = ends
        for entry in self.ranges:
            key = entry['partition']
            physical, logical, remaining = entry['partition_offset'], entry['offset'], entry['rows']
            ends = file_ends[key]
            while remaining:
                index = bisect.bisect_right(ends, physical)
                start = 0 if index == 0 else ends[index - 1]
                count = min(remaining, ends[index] - physical)
                item = self.writers[key].files[index]
                self.read_ranges.append(dict(offset=logical, rows=count, file_offset=physical - start,
                    path=str(self.partitions[key] / item['path'])))
                physical += count
                logical += count
                remaining -= count
        if not self.files:
            # Empty datasets contain no genomic rows and still have a readable schema.
            pa, pq = require_arrow()
            path = self.directory / 'empty.parquet'
            pq.write_table(pa.Table.from_pylist([], schema=self.schema), str(path), compression='zstd')
            self.files.append(dict(path=path.name, rows=0, size_bytes=path.stat().st_size))

    def abort(self):
        for writer in self.writers.values():
            writer.abort()


def iter_feature_batches(directory, columns=None, batch_size=1024):
    """Shared loader for future NN/filtering: avoid loading all Parquet at once."""
    _, pq = require_arrow()
    directory = Path(directory)
    run = directory.parent / 'run.json'
    if (directory.parent / 'INCOMPLETE.txt').exists() or (directory.parent / 'FAILED.txt').exists() or not run.exists():
        raise ValueError('Feature extraction is incomplete or failed')
    report = json.loads(run.read_text(encoding='utf-8'))
    if report.get('status') != 'complete':
        raise ValueError('Feature extraction is not complete')
    if report.get('schema') != 'read_features_parquet_v3':
        raise ValueError('Splice-aware features require Parquet v3; re-extract older data')
    for entry in report['parquet_files']['reads']:
        path = directory / entry['path']
        if path.stat().st_size != entry['size_bytes']:
            raise ValueError(f'Parquet file size changed: {path}')
        parquet = pq.ParquetFile(str(path))
        if (parquet.schema_arrow.metadata or {}).get(b'schema_version') != b'read_features_v3':
            raise ValueError('Read Parquet schema must be v3; re-extract older data')
        yield from parquet.iter_batches(batch_size=batch_size, columns=columns)
