"""Binary repeat annotation from an explicitly softmasked reference FASTA."""

COMPLEMENT = str.maketrans('ACGTNacgtn', 'TGCANtgcan')


def softmasked_window(fasta, chrom, start, strand, window):
    """Return uppercase sequence and lowercase-acgt flags in RNA 5′→3′ order."""
    if window < 3 or window % 2 != 1 or strand not in ('+', '-'):
        raise ValueError('Use an odd sequence window and an explicit strand')
    length = fasta.get_reference_length(chrom)
    if not 0 <= start < length:
        raise ValueError('Site outside softmasked FASTA')
    flank = window // 2
    sequence = ('N' * max(0, flank - start)
                + fasta.fetch(chrom, max(0, start - flank), min(length, start + flank + 1))
                + 'N' * max(0, start + flank + 1 - length))
    if strand == '-':
        sequence = sequence.translate(COMPLEMENT)[::-1]
    # Unknown N/n and artificial chromosome-edge padding are not repeat evidence.
    return sequence.upper(), [base in 'acgt' for base in sequence]

