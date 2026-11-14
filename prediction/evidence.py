"""Independent retention flags, with compatibility for earlier decision tables."""

from prediction.motifs import POLICY_FLAG_FIELDS

FLAG_FIELDS = ('flag_db', 'flag_short_read', 'flag_nn', 'flag_drach', 'flag_mettl16') + POLICY_FLAG_FIELDS


def evidence_values(row):
    """The absent new motif flag in an older four-flag table means disabled."""
    optional = {'flag_mettl16', *POLICY_FLAG_FIELDS}
    values = [row.get(name, '0') if name in optional else row.get(name) for name in FLAG_FIELDS]
    if any(value not in ('0', '1') for value in values):
        raise ValueError('Evidence flags must be binary')
    if row.get('modification') != 'm6A' and (values[3] == '1' or values[4] == '1'):
        raise ValueError('DRACH and METTL16 motif retention require an original m6A candidate')
    for name, value in zip(POLICY_FLAG_FIELDS, values[5:]):
        target = 'm5C' if name in ('flag_nsun6', 'flag_nsun2') else 'pseU'
        if value == '1' and row.get('modification') != target:
            raise ValueError('Motif retention flag disagrees with the original target')
    if sum(map(int, values[3:])) > 1:
        raise ValueError('Sequence motif flags are mutually exclusive')
    return values
