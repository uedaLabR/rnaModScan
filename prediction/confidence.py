"""Audit modkit's confidence-passing counts without changing candidate AF."""

BASE_FIELDS = ('valid_coverage', 'failed_calls', 'different_base_reads', 'no_calls', 'af_pass')
AUDIT_FIELDS = BASE_FIELDS + tuple(
    field + '_' + target for target in ('m6A', 'inosine') for field in BASE_FIELDS
)


def confidence_fields(row):
    """A zero confidence-passing denominator is undefined, rather than AF zero."""
    result = {field: row[field] for field in BASE_FIELDS if field != 'af_pass'}
    valid = row['valid_coverage']
    result['af_pass'] = row['modified_reads'] / valid if valid else ''
    return result


def combine_confidence(row, rows_by_target):
    """Keep each target's confidence counts; a missing record remains blank."""
    for target in ('m6A', 'inosine'):
        source = rows_by_target.get(target, {})
        for field in BASE_FIELDS:
            row[field + '_' + target] = source.get(field, '')
    return row
