"""Indexed genome and strand-oriented motif helpers."""
from pathlib import Path

class IndexedFasta:
    """Read uncompressed, samtools-faidx-indexed FASTA without loading the genome."""
    def __init__(self, path):
        self.path = Path(path)
        if self.path.suffix in ('.gz', '.bgz'):
            raise ValueError('Use an uncompressed GRCh38 FASTA plus .fai')
        self.index = {}
        with Path(str(path)+'.fai').open() as h:
            for line in h:
                c = line.split()
                self.index[c[0]] = tuple(map(int, c[1:5]))
        self.handle = self.path.open('rb')
        self.cache = None
        self.chunk_size = 65536

    def close(self):
        self.handle.close()

    def motif(self, chrom, start, strand):
        return self.context(chrom, start, strand, left=2, right=2)

    def context(self, chrom, start, strand, *, left, right):
        """Return an RNA-oriented reference window with asymmetric flanks."""
        if strand not in ('+', '-') or min(left, right) < 0:
            raise ValueError('Set an explicit strand and nonnegative reference flanks')
        if chrom not in self.index:
            raise ValueError(f'FASTA contig missing: {chrom}; coordinate names must agree')
        length, offset, bases, width = self.index[chrom]
        if not 0 <= start < length:
            raise ValueError(f'Position outside FASTA: {chrom}:{start}')
        before, after = (left, right) if strand == '+' else (right, left)
        lo, hi = max(0, start-before), min(length, start+after+1)
        if self.cache is None or self.cache[0] != chrom or not (self.cache[1] <= lo and hi <= self.cache[2]):
            block_lo = lo // self.chunk_size * self.chunk_size
            block_hi = min(length, block_lo + self.chunk_size + before + after)
            first = offset + block_lo//bases*width + block_lo%bases
            last = offset + (block_hi-1)//bases*width + (block_hi-1)%bases
            self.handle.seek(first)
            block = self.handle.read(last-first+1).replace(b'\n', b'').replace(b'\r', b'').decode().upper()
            if len(block) != block_hi-block_lo:
                raise ValueError('FASTA/.fai mismatch')
            self.cache = (chrom, block_lo, block_hi, block)
        seq = self.cache[3][lo-self.cache[1]:hi-self.cache[1]]
        seq = 'N'*max(0, before-start) + seq + 'N'*max(0, start+after+1-length)
        return seq if strand == '+' else seq.translate(str.maketrans('ACGTN','TGCAN'))[::-1]


def drach(seq):
    return (len(seq)==5 and seq[0] in 'AGT' and seq[1] in 'AG' and
            seq[2]=='A' and seq[3]=='C' and seq[4] in 'ACT')


def mettl16(seq):
    """Match the canonical candidate-aligned UAC[m6A]GAGAA nonamer only."""
    return isinstance(seq, str) and seq.upper().replace('U', 'T') == 'TACAGAGAA'


def candidate_nonamer(sequence):
    """Extract candidate -3/+5 from an odd-length, candidate-centered window."""
    if not isinstance(sequence, str) or len(sequence) < 11 or len(sequence) % 2 != 1:
        return ''
    center = len(sequence) // 2
    return sequence[center-3:center+6].upper().replace('U', 'T')


def mettl16_context(sequence):
    return mettl16(candidate_nonamer(sequence))



