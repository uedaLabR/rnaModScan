"""One row per candidate site: sampled reads, exact NN tensors and audit metadata."""
import csv
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

from utils.features.parquet import PartitionedParquetWriter, feature_schema, require_arrow
from utils.features.index import INDEX_FIELDS
from utils.features.site_metadata import annotate_site_metadata, metadata_fields, metadata_provenance
from utils.io.artifacts import completed_run
from utils.nn.inputs import MODIFICATIONS, trimer_tokens
from utils.features.compact_inputs import integer_read_type, pack_read_inputs, SITE_SCHEMAS, STORAGE
from utils.references.catalog import sha256

SCHEMA = 'site_bags_parquet_v4'
LOCATION_FIELDS = ['site_parquet', 'site_row_group', 'site_row_offset']


def metadata_type(name, pa):
    """Preserve numeric reporting values and their nulls; other audit columns are text."""
    if name in ('start', 'end', 'read_offset', 'eligible_depth', 'retained_depth',
                'central_deletions', 'central_refskips', 'position', 'sampling_seed'):
        return pa.int64()
    if name.startswith(('af', 'paired_ivt_af')):
        return pa.float64()
    prefixes = ('call_count_', 'total_reads', 'modified_reads', 'valid_coverage',
                'failed_calls', 'different_base_reads', 'no_calls', 'paired_ivt_depth',
                'paired_ivt_modified_reads', 'candidate_')
    if name.startswith(prefixes):
        return pa.int64()
    return pa.string()


def bag_schema(window, source_fields):
    pa, _ = require_arrow()
    audit_names = list(dict.fromkeys(list(source_fields) + INDEX_FIELDS +
        ['cell', 'reference_cell', 'condition', 'position', 'central_deletions', 'central_refskips', 'sampling_seed']))
    derived = {field.name for field in metadata_fields()}
    fields = [pa.field(name, metadata_type(name, pa)) for name in audit_names if name not in derived]
    read_struct = pa.struct([field for field in feature_schema((window - 1) // 2, MODIFICATIONS)
                             if field.name != 'row_id'])
    fields += metadata_fields()
    fields += [pa.field('sequence_tokens', pa.list_(pa.int32(), window - 2)),
               pa.field('sequence_repeat_mask', pa.list_(pa.bool_(), window)),
               pa.field('read_inputs', pa.list_(integer_read_type(window))),
               pa.field('raw_reads', pa.list_(read_struct))]
    return pa.schema(fields, metadata={b'schema_version': SCHEMA.encode(),
        b'orientation': b'RNA_5to3', b'feature_set': b'sequence_reads_repeat18_v1',
        b'read_feature_storage': STORAGE.encode(),
        b'repeat_source': b'FASTA lowercase acgt; N/padding false; reverse for minus strand',
        b'padding': b'none on disk; mask/padding constructed only for NN batches'})


class SiteBagWriter:
    """Lazy sink for the shared BAM scanner; all stored reads are real reads."""
    def __init__(self, out, site_rows, sample, catalog, *, mode, flank=9, max_reads=300,
                 sampling_seed=42, batch_sites=8, file_sites=256):
        self.out = Path(out)
        self.rows = {row['row_id']: row for row in site_rows}
        if not self.rows:
            raise ValueError('No candidate sites')
        self.sample, self.catalog, self.mode = sample, catalog, mode
        self.max_reads, self.sampling_seed = max_reads, sampling_seed
        self.schema = bag_schema(2 * flank + 1, next(iter(self.rows.values())).keys())
        self.batch_sites, self.file_sites = batch_sites, file_sites
        self.writer = None
        self.files = []

    def append_site(self, site, reads, reference_sequence, eligible_depth, central_deletions, central_refskips,
                    *, sequence_repeat_mask):
        if (len(sequence_repeat_mask) != len(reference_sequence)
                or any(type(value) is not bool for value in sequence_repeat_mask)):
            raise ValueError('Every site needs a binary FASTA repeat mask aligned to its reference window')
        if len(reads) > self.max_reads or len(reads) != min(eligible_depth, self.max_reads):
            raise ValueError('BAM reservoir did not retain the expected number of reads')
        source = self.rows.pop(site['row_id'])
        record = dict(source)
        record.update(self.sample)
        record.update(site)
        record.update(reference_sequence=reference_sequence, position=site['start'],
                      eligible_depth=eligible_depth, retained_depth=len(reads), read_offset=0,
                      central_deletions=central_deletions, central_refskips=central_refskips,
                      sampling_seed=self.sampling_seed)
        if self.mode == 'inference':
            for name in list(record):
                if name == 'label' or name.startswith('label_') or name in ('positive_evidence', 'group_id'):
                    record[name] = None
        record.update(annotate_site_metadata(source, self.sample, self.catalog, self.mode))
        raw_reads = []
        for feature in reads:
            raw = dict(feature)
            raw['mm_skip_modes'] = [feature['mm_skip_modes'].get(modification) for modification in MODIFICATIONS]
            raw_reads.append(raw)
        record.update(sequence_tokens=trimer_tokens(reference_sequence), sequence_repeat_mask=sequence_repeat_mask,
                      read_inputs=pack_read_inputs(reads), raw_reads=raw_reads)
        typed = {}
        pa, _ = require_arrow()
        for field in self.schema:
            value = record.get(field.name)
            if value is None or isinstance(value, str) and value == '':
                typed[field.name] = None
            elif pa.types.is_integer(field.type):
                typed[field.name] = int(value)
            elif pa.types.is_floating(field.type):
                typed[field.name] = float(value)
            elif pa.types.is_string(field.type):
                typed[field.name] = str(value)
            else:
                typed[field.name] = value
        if self.writer is None:
            self.writer = PartitionedParquetWriter(self.out / 'sites', self.schema,
                                                   batch_rows=self.batch_sites, file_rows=self.file_sites)
        self.writer.set_partition(site['chrom'], site['strand'])
        self.writer.append(typed)

    def close(self):
        if self.rows:
            raise ValueError('Some candidate sites were not written')
        if self.writer is not None:
            self.writer.close()
            self.files = self.writer.files

    def abort(self):
        if self.writer is not None:
            self.writer.abort()

    def provenance(self):
        return dict(mode=self.mode, **metadata_provenance(self.catalog))


def build_site_bag_index(directories, out, *, require_labels=True):
    """Create lightweight coordinate-sorted locations; never deserialize read tensors."""
    _, pq = require_arrow()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE.txt').write_text('Site index is incomplete.\n')
    database = sqlite3.connect(str(out / 'sort.sqlite'))
    database.execute('CREATE TABLE sites (row_id TEXT PRIMARY KEY, chrom TEXT, start INTEGER, payload TEXT)')
    sources, fields, counts = [], list(INDEX_FIELDS), Counter()
    fields += LOCATION_FIELDS
    fields += [field.name for field in metadata_fields()]
    fields = list(dict.fromkeys(fields))
    target = None
    sampling_policy = None
    source_assemblies = set()
    sources_without_assembly = 0
    try:
        for directory in directories:
            directory = Path(directory).resolve()
            run = completed_run(directory)
            if run['schema'] not in SITE_SCHEMAS:
                raise ValueError('Expected common per-site Parquet')
            assembly = run.get('assembly')
            if assembly is None:
                sources_without_assembly += 1
            elif not isinstance(assembly, str) or not assembly.strip():
                raise ValueError('Feature assembly must be a nonempty identifier')
            else:
                source_assemblies.add(assembly)
            if len(source_assemblies) > 1 or (source_assemblies and sources_without_assembly):
                raise ValueError('Do not mix feature assemblies or declared and unknown assemblies in one index')
            policy = (run['max_reads'], run['sampling_seed'])
            if sampling_policy is not None and sampling_policy != policy:
                raise ValueError('Mixed read sampling policies in the same index')
            sampling_policy = policy
            sources.append(dict(path=str(directory), run_sha256=sha256(directory / 'run.json')))
            for item in run['parquet_files']['sites']:
                path = directory / 'sites' / item['path']
                if path.stat().st_size != item['size_bytes']:
                    raise ValueError('Site Parquet changed')
                parquet = pq.ParquetFile(path)
                excluded = {'sequence_tokens', 'sequence_repeat_mask', 'read_features', 'read_inputs', 'raw_reads'}
                metadata_columns = [name for name in parquet.schema_arrow.names if name not in excluded]
                fields = list(dict.fromkeys(fields + metadata_columns))
                for group in range(parquet.num_row_groups):
                    for offset, row in enumerate(parquet.read_row_group(group, columns=metadata_columns).to_pylist()):
                        if target is not None and target != row['modification']:
                            raise ValueError('Site index must have exactly one target')
                        target = row['modification']
                        counts['input_sites'] += 1
                        label = row['training_class']
                        if require_labels and label not in ('m6A', 'inosine', 'false_positive'):
                            raise ValueError('Training index cannot include unlabeled sites')
                        if require_labels:
                            counts['labeled_sites'] += 1
                        if row['retained_depth'] == 0:
                            counts['zero_read_sites'] += 1
                            if require_labels:
                                continue
                        row.update(label=str(('m6A', 'inosine', 'false_positive').index(label)) if label is not None else '',
                                   features_dir=os.path.relpath(directory, out),
                                   site_parquet=str(path.relative_to(directory)), site_row_group=group,
                                   site_row_offset=offset)
                        database.execute('INSERT INTO sites VALUES (?,?,?,?)',
                                         (row['row_id'], row['chrom'], row['start'], json.dumps(row)))
                        counts['indexed_sites'] += 1
        database.commit()
        with (out / 'sites.tsv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fields, delimiter='\t')
            writer.writeheader()
            for payload, in database.execute('SELECT payload FROM sites ORDER BY chrom,start,row_id'):
                row = json.loads(payload)
                for name, value in row.items():
                    if isinstance(value, list):
                        row[name] = ';'.join(value)
                writer.writerow(row)
        report = dict(status='complete', schema='site_bag_index_v3', target=target,
                      require_labels=require_labels, feature_sources=sources, counts=dict(counts),
                      max_reads=sampling_policy[0], sampling_seed=sampling_policy[1],
                      index_sha256=sha256(out / 'sites.tsv'))
        if source_assemblies:
            report.update(assembly=next(iter(source_assemblies)), coordinates='BED0')
        (out / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as error:
        (out / 'FAILED.txt').write_text(str(error))
        raise
    finally:
        database.close()
        (out / 'sort.sqlite').unlink(missing_ok=True)
