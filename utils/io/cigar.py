"""A compact CIGAR index, built once per alignment and reused at every site.

Coordinates are BED0 on the reference and BAM-stored query coordinates. An
intron is one segment, irrespective of its length; no full aligned-pair array
or reference-span dictionary is constructed.
"""
from bisect import bisect_left, bisect_right


class CigarIndex:
    def __init__(self, read):
        self.reference_starts, self.reference_segments = [], []
        self.query_starts, self.query_segments = [], []
        self.insertions, self.junctions = {}, []
        query, reference = 0, read.reference_start
        for operation, length in read.cigartuples or ():
            if length < 1:
                raise ValueError('CIGAR lengths must be positive')
            if operation in (0, 7, 8):
                self.reference_starts.append(reference)
                self.reference_segments.append((reference, reference + length, query, 'base'))
                self.query_starts.append(query)
                self.query_segments.append((query, query + length, reference))
                reference += length
                query += length
            elif operation in (2, 3):
                self.reference_starts.append(reference)
                state = 'deletion' if operation == 2 else 'refskip'
                self.reference_segments.append((reference, reference + length, None, state))
                reference += length
                if operation == 3:
                    self.junctions.append(query)
            elif operation == 1:
                anchor = reference - 1
                self.insertions[anchor] = self.insertions.get(anchor, 0) + length
                query += length
            elif operation == 4:
                query += length
            elif operation not in (5, 6):
                raise ValueError(f'Unsupported CIGAR operation {operation}')
        self.insertion_anchors = sorted(self.insertions)
        self.reference_end, self.query_length = reference, query

    def reference_to_query(self, position):
        """Return (stored query index, alignment state); D/N have no query base."""
        index = bisect_right(self.reference_starts, position) - 1
        if index >= 0:
            start, end, query, state = self.reference_segments[index]
            if position < end:
                return (query + position - start if query is not None else None), state
        return None, 'uncovered'

    def query_to_reference(self, position):
        """Return BED0 reference index, or None for inserted/soft-clipped bases."""
        index = bisect_right(self.query_starts, position) - 1
        if index >= 0:
            start, end, reference = self.query_segments[index]
            if position < end:
                return reference + position - start
        return None

    def window(self, lo, hi):
        """Expand only the requested short reference window, including D/N states."""
        mapping = {}
        index = max(0, bisect_right(self.reference_starts, lo) - 1)
        while index < len(self.reference_segments):
            start, end, query, state = self.reference_segments[index]
            if start >= hi:
                break
            for position in range(max(lo, start), min(hi, end)):
                mapping[position] = (query + position - start if query is not None else None), state
            index += 1
        first = bisect_left(self.insertion_anchors, lo - 1)
        last = bisect_left(self.insertion_anchors, hi)
        insertions = {anchor: self.insertions[anchor] for anchor in self.insertion_anchors[first:last]}
        return mapping, insertions
