"""Reusable sparse alignment index, including giant introns and query gaps."""
import importlib.util
import unittest
from types import SimpleNamespace

from utils.io.cigar import CigarIndex


class CigarIndexTest(unittest.TestCase):
    def test_reference_and_query_coordinates_keep_indel_and_splice_distinct(self):
        read = SimpleNamespace(reference_start=100,
            cigartuples=[(4, 2), (0, 4), (3, 100), (0, 4), (1, 1), (0, 2), (2, 2), (0, 3), (4, 1)])
        index = CigarIndex(read)
        self.assertEqual(index.reference_to_query(103), (5, 'base'))
        self.assertEqual(index.reference_to_query(104), (None, 'refskip'))
        self.assertEqual(index.reference_to_query(204), (6, 'base'))
        self.assertEqual(index.reference_to_query(208), (11, 'base'))
        self.assertEqual(index.reference_to_query(210), (None, 'deletion'))
        self.assertEqual(index.reference_to_query(215), (None, 'uncovered'))
        for query, reference in [(0, None), (1, None), (2, 100), (5, 103), (6, 204),
                                  (10, None), (11, 208), (15, 214), (16, None), (17, None)]:
            self.assertEqual(index.query_to_reference(query), reference)
        mapping, insertions = index.window(208, 213)
        self.assertEqual(mapping[208], (11, 'base'))
        self.assertEqual(mapping[210], (None, 'deletion'))
        self.assertEqual(insertions, {207: 1})  # Reverse-oriented low window edge.
        self.assertEqual(index.junctions, [6])

    def test_cigar_is_read_once_and_long_introns_are_not_expanded(self):
        class Read:
            reference_start = 10
            accesses = 0
            @property
            def cigartuples(self):
                self.accesses += 1
                return [(0, 5), (3, 100_000_000), (0, 5)]
        read = Read()
        index = CigarIndex(read)
        for position in (10, 12, 100_000_015, 100_000_019):
            query, state = index.reference_to_query(position)
            self.assertEqual(state, 'base')
            self.assertEqual(index.query_to_reference(query), position)
        mapping, _ = index.window(100_000_014, 100_000_017)
        self.assertEqual(mapping[100_000_014], (None, 'refskip'))
        self.assertEqual(len(mapping), 3)
        self.assertEqual(len(index.reference_segments), 3)
        self.assertEqual(read.accesses, 1)

    @unittest.skipUnless(importlib.util.find_spec('pysam'), 'pysam not installed')
    def test_query_mapping_matches_pysam_for_both_alignment_strands(self):
        import pysam
        for reverse in (False, True):
            read = pysam.AlignedSegment()
            read.reference_start = 10
            read.flag = 16 if reverse else 0
            read.query_sequence = 'A' * 16
            read.cigarstring = '2S4M1I3M2D20N5M1S'
            index = CigarIndex(read)
            expected = read.get_reference_positions(full_length=True)
            self.assertEqual([index.query_to_reference(q) for q in range(read.query_length)], expected)


if __name__ == '__main__':
    unittest.main()
