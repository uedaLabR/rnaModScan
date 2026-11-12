"""Normalize explicitly mapped positive-site lists into a portable BED catalog.

This is an import adapter, not an assay-specific significance filter. Every
input row must already be a selected positive site for the declared target.
"""
import csv
import json
import os
from pathlib import Path
import re
import tempfile

from utils.io.bed import open_text
from utils.references.catalog import MODIFICATIONS, ReferenceCatalog, sha256

TOP_KEYS = {'schema_version', 'assembly', 'species', 'sources'}
SOURCE_KEYS = {
    'id', 'kind', 'cell', 'modification', 'assembly', 'species', 'path', 'format',
    'coordinates', 'columns', 'source_url', 'citation', 'database', 'version',
    'method', 'metadata', 'original_assembly', 'conversion_provenance',
    'evidence_type', 'source_cell', 'cross_cell_mapping', 'evidence_bucket_note',
}
SOURCE_IMPORT_KEYS = {'path', 'format', 'coordinates', 'columns'}
SITE_FIELDS = {'chrom', 'start', 'end', 'position', 'strand'}
COORDINATES = {'BED0', 'one_based_position', 'one_based_closed'}


def required_text(values, key, label):
    value = values.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f'{label}: set a nonempty {key}')
    return value


def check_keys(values, allowed, label):
    if not isinstance(values, dict):
        raise ValueError(f'{label} must be a mapping')
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f'{label}: unknown fields {sorted(unknown)}')


def read_manifest(path):
    import yaml
    try:
        config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    except yaml.YAMLError as error:
        raise ValueError(f'Invalid reference manifest: {path}: {error}') from error
    check_keys(config, TOP_KEYS, 'manifest')
    if config.get('schema_version', 1) != 1:
        raise ValueError('Reference manifest schema_version must be 1')
    required_text(config, 'assembly', 'manifest')
    if 'species' in config:
        required_text(config, 'species', 'manifest')
    if not isinstance(config.get('sources'), list) or not config['sources']:
        raise ValueError('Reference manifest needs at least one positive-site source')
    return config


def source_settings(source, config):
    check_keys(source, SOURCE_KEYS, 'source')
    source = dict(source)
    name = required_text(source, 'id', 'source')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}', name):
        raise ValueError('Source IDs must be simple names without spaces or path separators')
    if name in {'short_read_any', 'database_any'}:
        raise ValueError('Reserved source ID')
    if source.get('kind') not in {'short_read', 'database'}:
        raise ValueError(f'{name}: kind must be short_read or database')
    cell = required_text(source, 'cell', name)
    if source['kind'] == 'short_read' and cell == '*':
        raise ValueError(f'{name}: short_read sources need an explicit cell or tissue')
    if source.get('modification') not in MODIFICATIONS:
        raise ValueError(f'{name}: modification must be one of {MODIFICATIONS}')
    required_text(source, 'path', name)
    if source.get('format') not in {'bed', 'tsv', 'csv'}:
        raise ValueError(f'{name}: set format to bed, tsv or csv')
    if source.get('coordinates') not in COORDINATES:
        raise ValueError(f'{name}: explicitly set coordinates to {sorted(COORDINATES)}')
    if source.get('assembly', config['assembly']) != config['assembly']:
        raise ValueError(f'{name}: source assembly differs from the manifest; no liftover is performed')
    source['assembly'] = config['assembly']
    if 'species' in source:
        required_text(source, 'species', name)
    if config.get('species') and source.get('species', config['species']) != config['species']:
        raise ValueError(f'{name}: source species differs from the manifest')
    if config.get('species'):
        source['species'] = config['species']
    if not source.get('source_url') and not source.get('citation'):
        raise ValueError(f'{name}: provide source_url or citation, including for unpublished assays')
    for field in ('source_url', 'citation', 'database', 'version', 'method', 'evidence_type'):
        if field in source:
            required_text(source, field, name)
    if source['kind'] == 'database':
        required_text(source, 'database', name)
    if 'metadata' in source and not isinstance(source['metadata'], dict):
        raise ValueError(f'{name}: metadata must be a mapping')
    if source['format'] == 'bed':
        if source['coordinates'] != 'BED0':
            raise ValueError(f'{name}: bed inputs use BED0; map one-based tables as tsv or csv')
        source.setdefault('columns', dict(chrom=1, start=2, end=3, strand=6))
    columns = source.get('columns')
    check_keys(columns, SITE_FIELDS, f'{name} columns')
    required = {'chrom', 'strand', 'position'} if source['coordinates'] == 'one_based_position' else {'chrom', 'strand', 'start', 'end'}
    if set(columns) != required:
        raise ValueError(f'{name}: columns must map exactly {sorted(required)}')
    for column in columns.values():
        valid = (type(column) is int and column >= 1) if source['format'] == 'bed' else (isinstance(column, str) and bool(column.strip()))
        if not valid:
            raise ValueError(f'{name}: use 1-based column numbers for BED or exact header names for TSV/CSV')
    if len(set(columns.values())) != len(columns):
        raise ValueError(f'{name}: site fields must use distinct input columns')
    return source


def integer(value, context):
    value = value.strip()
    if not re.fullmatch(r'[+-]?\d+', value):
        raise ValueError(f'{context}: expected an integer coordinate, got {value!r}')
    return int(value)


def site_key(values, coordinates, context):
    chrom, strand = values['chrom'].strip(), values['strand'].strip()
    if not chrom or any(c.isspace() for c in chrom):
        raise ValueError(f'{context}: chromosome name must be nonempty and contain no whitespace')
    if strand not in {'+', '-'}:
        raise ValueError(f'{context}: an explicit + or - strand is required')
    if coordinates == 'one_based_position':
        position = integer(values['position'], context)
        start, end = position-1, position
    else:
        start, end = integer(values['start'], context), integer(values['end'], context)
        if coordinates == 'one_based_closed':
            start -= 1
    if start < 0 or end != start+1:
        raise ValueError(f'{context}: require a single-base site with a nonnegative BED0 start')
    return chrom, start, strand


def read_sites(path, source):
    sites, rows = set(), 0
    columns = source['columns']
    with open_text(path) as handle:
        if source['format'] == 'bed':
            for line_number, line in enumerate(handle, 1):
                if not line.strip() or line.startswith(('#', 'track', 'browser')):
                    continue
                fields = line.split()
                context = f'{path}:{line_number}'
                if len(fields) < max(columns.values()):
                    raise ValueError(f'{context}: missing mapped BED column')
                values = {name: fields[index-1] for name, index in columns.items()}
                sites.add(site_key(values, source['coordinates'], context)); rows += 1
        else:
            lines = (line for line in handle if line.strip() and not line.startswith('#'))
            reader = csv.DictReader(lines, delimiter='\t' if source['format'] == 'tsv' else ',')
            headers = reader.fieldnames
            if not headers or len(set(headers)) != len(headers):
                raise ValueError(f'{path}: require a header with distinct column names')
            if set(columns.values()) - set(headers):
                raise ValueError(f'{path}: mapped columns are missing from the header')
            try:
                for record in reader:
                    context = f'{path}:data row {rows+1}'
                    if None in record or any(record[name] is None for name in columns.values()):
                        raise ValueError(f'{context}: inconsistent column count')
                    values = {name: record[column] for name, column in columns.items()}
                    sites.add(site_key(values, source['coordinates'], context)); rows += 1
            except csv.Error as error:
                raise ValueError(f'{path}: invalid delimited table: {error}') from error
    if not rows:
        raise ValueError(f'{path}: no positive-site records')
    return sites, rows


def build_catalog(manifest, output):
    """Validate in a private staging directory; reserve output exclusively.

    Existing destinations are never replaced. Validation errors create no
    destination. The catalog is moved last, after every runtime dependency.
    """
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output exists: {output}; choose a new reference directory')
    config = read_manifest(manifest)
    sources = [source_settings(source, config) for source in config['sources']]
    ids = [source['id'] for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate source IDs')
    species = {source['species'] for source in sources if source.get('species')}
    if len(species) > 1:
        raise ValueError('Do not mix species in one reference catalog')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.rnamodscan-references-', dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage/'sites').mkdir(); (stage/'provenance').mkdir()
        imported, reports = [], []
        manifest_sha = sha256(manifest)
        for index, source in enumerate(sources, 1):
            path = Path(source['path']).expanduser()
            if not path.is_absolute():
                path = manifest.parent/path
            path = path.resolve()
            input_sha = sha256(path)
            sites, rows = read_sites(path, source)
            if sha256(path) != input_sha:
                raise ValueError(f'Input changed during import: {path}')
            safe_id = re.sub(r'[^A-Za-z0-9_.-]', '_', source['id'])
            stem = f'{index:04d}-{safe_id}'
            bed, provenance = Path('sites')/(stem+'.bed'), Path('provenance')/(stem+'.json')
            with (stage/bed).open('w', encoding='utf-8', newline='') as handle:
                for chrom, start, strand in sorted(sites):
                    handle.write(f'{chrom}\t{start}\t{start+1}\t{source["id"]}\t0\t{strand}\n')
            record = dict(schema='positive_reference_import_v1', source=source,
                          input_sha256=input_sha, manifest_sha256=manifest_sha,
                          input_rows=rows, unique_sites=len(sites), duplicate_rows=rows-len(sites),
                          normalized_coordinates='BED0', assembly=config['assembly'],
                          policy='Every input row is an author-selected positive site; absent overlap is unknown, not negative. No liftover or chromosome renaming.')
            (stage/provenance).write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
            entry = {key:value for key,value in source.items() if key not in SOURCE_IMPORT_KEYS}
            entry.update(path=bed.as_posix(), coordinates='BED0', sha256=sha256(stage/bed),
                         provenance=provenance.as_posix(), unique_sites=len(sites))
            imported.append(entry)
            reports.append(dict(id=source['id'], modification=source['modification'], cell=source['cell'],
                                input_rows=rows, unique_sites=len(sites), duplicate_rows=rows-len(sites)))
        if sha256(manifest) != manifest_sha:
            raise ValueError('Manifest changed during import')
        catalog = dict(assembly=config['assembly'], sources=imported)
        if config.get('species') or species:
            catalog['species'] = config.get('species') or next(iter(species))
        (stage/'catalog.json').write_text(json.dumps(catalog, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        checked = ReferenceCatalog(stage/'catalog.json', for_training=True)
        assert len(checked.sources) == len(imported)
        report = dict(status='complete', schema='positive_reference_import_v1', assembly=config['assembly'],
                      catalog=str(output/'catalog.json'), sources=reports,
                      label_policy='positive-site evidence only; negative training labels require separate controls')
        (stage/'import_run.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        files = sorted(p for p in stage.rglob('*') if p.is_file())
        (stage/'SHA256SUMS').write_text(''.join(sha256(p)+'  '+p.relative_to(stage).as_posix()+'\n' for p in files), encoding='utf-8')
        output.mkdir()  # Exclusive; do not overwrite even an empty existing directory.
        for name in ('sites', 'provenance', 'import_run.json', 'SHA256SUMS', 'catalog.json'):
            os.rename(stage/name, output/name)
    return report


def check_catalog(path, *, for_training=False):
    catalog = ReferenceCatalog(path, for_training=for_training)
    return dict(status='complete', assembly=catalog.assembly,
                purpose='training' if for_training else 'prediction',
                sources=[dict(id=s['id'], modification=s['modification'], cell=s['cell'],
                              status=s['status'], unique_sites=s.get('unique_sites')) for s in catalog.sources])
