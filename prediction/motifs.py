"""Validate reporting-only reference motif evidence for retention policies."""
from utils.references.genome import candidate_nonamer, mettl16


# These are sequence-only reporting rules, not enzyme/structure assignments.
# Code, target, config switch, independent TSV flag, DNA pattern, target index.
POLICY_MOTIFS = (
    (3, 'm5C', 'm5c_nsun6_keep', 'flag_nsun6', 'CTCCA', 0),
    (4, 'm5C', 'm5c_nsun2_keep', 'flag_nsun2', 'CNGGG', 0),
    (5, 'pseU', 'pseu_trub1_keep', 'flag_trub1', 'GTTCNANNC', 2),
    (6, 'pseU', 'pseu_pus7_keep', 'flag_pus7', 'TGTAR', 2),
)
POLICY_FLAG_FIELDS = tuple(rule[3] for rule in POLICY_MOTIFS)
MOTIF_ENCODING = {
    '0': 'none', '1': 'm6A DRACH', '2': 'm6A canonical METTL16 UAC[m6A]GAGAA',
    '3': 'm5C NSUN6-associated [m5C]UCCA',
    '4': 'm5C NSUN2-associated [m5C]NGGG',
    '5': 'pseU TRUB1-associated GU[Psi]CNANNC',
    '6': 'pseU PUS7-associated UG[Psi]AR',
}
_BASES = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'N': 'ACGT', 'R': 'AG'}


def motif_keep_code(drach, mettl16, nsun6=0, nsun2=0, trub1=0, pus7=0):
    """Stable numeric code for mutually exclusive sequence-retention rules."""
    values = (drach, mettl16, nsun6, nsun2, trub1, pus7)
    if any(value not in (0, 1) for value in values):
        raise ValueError('Motif retention flags must be binary')
    if sum(values) > 1:
        raise ValueError('Sequence motif flags are mutually exclusive')
    return sum(code*value for code, value in enumerate(values, 1))


def motif_matches(sequence, pattern):
    """Match concrete reference bases; unknown/padded N is never evidence."""
    if not isinstance(sequence, str):
        return False
    sequence = sequence.upper().replace('U', 'T')
    return len(sequence) == len(pattern) and all(base in _BASES[allowed]
                                               for base, allowed in zip(sequence, pattern))


def routed_policy_motif(genome, row, filtering):
    """Read each candidate-aligned window without scanning or creating sites."""
    for code, target, option, flag, pattern, index in POLICY_MOTIFS:
        if row['modification'] != target or not filtering.get(option, False):
            continue
        sequence = genome.context(row['chrom'], int(row['start']), row['strand'],
                                  left=index, right=len(pattern)-index-1)
        if motif_matches(sequence, pattern):
            return code, sequence
    return 0, ''


def policy_motif_flags(row):
    """Validate routed motif proof again before emitting final evidence."""
    value = row.get('policy_motif_code')
    code = 0 if value in (None, '') else int(value)
    flags = dict.fromkeys(POLICY_FLAG_FIELDS, 0)
    if code:
        rule = next((r for r in POLICY_MOTIFS if r[0] == code), None)
        if rule is None or row['modification'] != rule[1] or not motif_matches(row.get('policy_motif_sequence'), rule[4]):
            raise ValueError('Policy motif code requires the correct target and candidate-aligned sequence')
        flags[rule[3]] = 1
    return flags


def mettl16_site(row):
    """Use a routed nine-base proof or the saved centered reference window."""
    routed = row.get('mettl16_motif') or ''
    saved = candidate_nonamer(row.get('reference_sequence'))
    if routed and saved and routed.upper().replace('U', 'T') != saved:
        raise ValueError('Routed METTL16 motif disagrees with the saved reference context')
    return mettl16(routed or saved)
