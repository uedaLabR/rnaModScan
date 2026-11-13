"""Validate reporting-only reference motif evidence for retention policies."""
from utils.references.genome import candidate_nonamer, mettl16


def mettl16_site(row):
    """Use a routed nine-base proof or the saved centered reference window."""
    routed = row.get('mettl16_motif') or ''
    saved = candidate_nonamer(row.get('reference_sequence'))
    if routed and saved and routed.upper().replace('U', 'T') != saved:
        raise ValueError('Routed METTL16 motif disagrees with the saved reference context')
    return mettl16(routed or saved)
