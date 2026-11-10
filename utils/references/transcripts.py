"""Map spliced transcript BED0 positions to the input genome, without liftover."""
from bisect import bisect_right
from dataclasses import dataclass

from utils.features.position import annotation_attributes
from utils.io.bed import open_text


@dataclass
class TranscriptCoordinates:
    chrom: str
    strand: str
    exons: list

    def __post_init__(self):
        self.exons = sorted(set(self.exons), reverse=self.strand == '-')
        if self.strand not in ('+', '-'):
            raise ValueError('Transcript strand must be explicit')
        genomic = sorted(self.exons)
        if any(lo < 0 or hi <= lo for lo, hi in genomic):
            raise ValueError('Invalid exon coordinates')
        if any(left[1] > right[0] for left, right in zip(genomic, genomic[1:])):
            raise ValueError('Overlapping exons in one transcript')
        self.offsets = []
        self.length = 0
        for lo, hi in self.exons:
            self.offsets.append(self.length)
            self.length += hi - lo

    def genomic_position(self, position):
        """Position zero is the transcript's RNA 5-prime end on either strand."""
        if not 0 <= position < self.length:
            raise ValueError('Position outside transcript')
        index = bisect_right(self.offsets, position) - 1
        lo, hi = self.exons[index]
        within = position - self.offsets[index]
        return lo + within if self.strand == '+' else hi - 1 - within

    def junction_context(self, position, flank=2):
        """Whether the reference k-mer crosses an exon junction."""
        return any(position - flank < offset <= position + flank for offset in self.offsets[1:])


def load_transcript_coordinates(annotation, transcript_ids):
    """Read GENCODE GTF/GFF3 exons for exact versioned IDs needed by inference."""
    exons, identities = {}, {}
    with open_text(annotation) as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 9 or fields[2] != 'exon':
                continue
            attrs = annotation_attributes(fields[8])
            ids = [attrs['transcript_id']] if 'transcript_id' in attrs else attrs.get('Parent', '').split(',')
            for transcript_id in ids:
                if transcript_id not in transcript_ids:
                    continue
                identity = fields[0], fields[6]
                if transcript_id in identities and identities[transcript_id] != identity:
                    raise ValueError('Transcript has inconsistent chromosome/strand: ' + transcript_id)
                identities[transcript_id] = identity
                exons.setdefault(transcript_id, []).append((int(fields[3]) - 1, int(fields[4])))
    return {name: TranscriptCoordinates(*identities[name], intervals) for name, intervals in exons.items()}
