"""Independent retention flags, with compatibility for earlier decision tables."""

FLAG_FIELDS = ('flag_db', 'flag_short_read', 'flag_nn', 'flag_drach', 'flag_mettl16')


def evidence_values(row):
    """The absent new motif flag in an older four-flag table means disabled."""
    values = [row.get(name, '0') if name == 'flag_mettl16' else row.get(name) for name in FLAG_FIELDS]
    if any(value not in ('0', '1') for value in values):
        raise ValueError('Evidence flags must be binary')
    if row.get('modification') != 'm6A' and (values[3] == '1' or values[4] == '1'):
        raise ValueError('DRACH and METTL16 motif retention require an original m6A candidate')
    return values
