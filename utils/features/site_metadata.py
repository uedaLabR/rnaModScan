"""Reference support and saved joint labels for the common per-site dataset.

Support is reporting metadata, never an NN input or a new label assignment.
Short-read support uses the sample's explicitly configured reference_cell; no
cell name is inferred. Source IDs retain the exact experimental/DB provenance.
"""
from utils.adenine import CLASSES, MODIFICATIONS
from utils.references.catalog import sha256

TRAINING_FIELDS = ('training_label_m6A', 'training_label_inosine',
                   'training_label_false_positive')
SUPPORT_KINDS = ('short_read', 'rmbase')


def metadata_fields():
    """Arrow fields to append to a site schema; inference labels remain null."""
    from utils.features.parquet import require_arrow
    pa, _ = require_arrow()
    fields = [pa.field('training_class', pa.string())]
    fields.extend(pa.field(name, pa.bool_()) for name in TRAINING_FIELDS)
    for modification in MODIFICATIONS:
        for kind in SUPPORT_KINDS:
            fields.append(pa.field(f'support_{kind}_{modification}', pa.bool_()))
            fields.append(pa.field(f'support_{kind}_sources_{modification}', pa.list_(pa.string())))
    return fields


def saved_training_labels(row, sample, mode):
    """Project one existing joint label to nullable one-hot fields.

    The original label_m6A/label_inosine fields are binary assay-label audit
    columns. They must not be overwritten by the new joint-class one-hot flags.
    """
    if mode == 'inference':
        return dict(training_class=None, **dict.fromkeys(TRAINING_FIELDS))
    if mode != 'training':
        raise ValueError('mode must be training or inference')
    if row.get('modification') != 'adenine':
        raise ValueError('Training requires an existing joint adenine site')
    label = str(row.get('label', ''))
    if label not in ('0', '1', '2') or row.get('label_status', 'final') != 'final':
        raise ValueError('Training requires a saved final joint label; missing support is not a negative label')
    original = [str(row.get('label_' + modification, '')) for modification in MODIFICATIONS]
    if original == ['1', '1']:
        raise ValueError('Dual-positive sites must be excluded and audited before extraction')
    if (label == '2' and sample['condition'] == 'WT'
            and any('label_' + modification in row for modification in MODIFICATIONS)
            and original != ['0', '0']):
        raise ValueError('A WT negative with an unknown partner modification is not a joint false positive')
    training_class = CLASSES[int(label)]
    return dict(training_class=training_class,
                **{name: index == int(label) for index, name in enumerate(TRAINING_FIELDS)})


def support_sources(catalog, reference_cell, modification, kind, chrom, start, strand):
    """Return None when no applicable ready source exists, otherwise an overlap flag."""
    if catalog is None:
        return None, []
    applicable = catalog.applicable(reference_cell, modification)
    if kind == 'short_read':
        sources = [source for source in applicable
                   if source['kind'] == 'short_read' and source['cell'] == reference_cell
                   and source.get('source_cell', source['cell']) == reference_cell]
    else:
        sources = [source for source in applicable
                   if source['kind'] == 'database' and source.get('database') == 'RMBase'
                   and str(source.get('version')) == '3.0']
    ready_ids = {source['id'] for source in sources if source.get('status', 'ready') == 'ready'}
    if not ready_ids:
        return None, []
    matches = catalog.matches(reference_cell, modification, chrom, start, strand)
    matched_ids = [source_id for source_id in matches if source_id in ready_ids]
    return bool(matched_ids), matched_ids


def annotate_site_metadata(row, sample, catalog, mode):
    """Create support/label columns for one site without changing its saved label.

    reference_cell is the explicit cell-to-reference mapping. Current catalogs
    describe cell lines, not matched donors, samples, or biological replicates.
    """
    for name in ('sample_id', 'cell', 'reference_cell', 'condition'):
        if not sample.get(name):
            raise ValueError('Set explicit sample_id, cell, reference_cell and condition')
        if row.get(name) not in (None, '', sample[name]):
            raise ValueError('Site and sample metadata disagree: ' + name)
    if sample['condition'] not in ('WT', 'IVT'):
        raise ValueError('condition must be WT or IVT')
    chrom, strand = row['chrom'], row['strand']
    start = int(row['start'])
    if not chrom or start < 0 or strand not in ('+', '-'):
        raise ValueError('Use GRCh38 BED0 coordinates and an explicit strand')
    if row.get('end') not in (None, '') and int(row['end']) != start + 1:
        raise ValueError('Site intervals must cover one base')
    result = saved_training_labels(row, sample, mode)
    for modification in MODIFICATIONS:
        for kind in SUPPORT_KINDS:
            support, source_ids = support_sources(catalog, sample['reference_cell'], modification,
                                                   kind, chrom, start, strand)
            result[f'support_{kind}_{modification}'] = support
            result[f'support_{kind}_sources_{modification}'] = source_ids
    return result


def metadata_provenance(catalog):
    """Keep source assembly, cell scope, file digests and provenance in run.json."""
    if catalog is None:
        return dict(catalog=None, short_read_scope='explicit_reference_cell', sources=[])
    return dict(catalog=str(catalog.path), catalog_sha256=sha256(catalog.path),
                short_read_scope='explicit_reference_cell; source_cell must agree; no replicate matching implied',
                rmbase_scope='RMBase 3.0, applicable exact cell or wildcard',
                support_interpretation='overlap metadata only; absent/unavailable support never assigns a negative label',
                sources=[dict(source) for source in catalog.sources])
