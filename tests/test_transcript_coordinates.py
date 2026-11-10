"""Exact BED0 orientation, splice junctions and versioned GENCODE identities."""
from pathlib import Path
import tempfile
import unittest

from utils.references.transcripts import TranscriptCoordinates, load_transcript_coordinates


class TranscriptCoordinatesTest(unittest.TestCase):
    def test_plus_and_minus_exon_boundaries(self):
        plus = TranscriptCoordinates('chr1', '+', [(10, 13), (20, 24)])
        minus = TranscriptCoordinates('chr1', '-', [(10, 13), (20, 24)])
        self.assertEqual([plus.genomic_position(i) for i in range(7)], [10, 11, 12, 20, 21, 22, 23])
        self.assertEqual([minus.genomic_position(i) for i in range(7)], [23, 22, 21, 20, 12, 11, 10])
        self.assertTrue(plus.junction_context(3))
        self.assertFalse(plus.junction_context(0))
        self.assertTrue(minus.junction_context(4))
        for position in (-1, 7):
            with self.assertRaises(ValueError):
                minus.genomic_position(position)

    def test_gff3_and_gtf_keep_exact_versions_and_ignore_other_transcripts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'annotation.gff3'
            path.write_text('chr1\ttest\texon\t11\t13\t.\t+\t.\tParent=tx.1;transcript_id=tx.1\n'
                            'chr1\ttest\texon\t21\t24\t.\t+\t.\tParent=tx.1\n'
                            'chr2\ttest\texon\t31\t34\t.\t-\t.\ttranscript_id "tx.2";\n')
            result = load_transcript_coordinates(path, {'tx.1', 'tx.2', 'tx'})
            self.assertEqual(set(result), {'tx.1', 'tx.2'})
            self.assertEqual(result['tx.1'].genomic_position(3), 20)
            self.assertEqual(result['tx.2'].genomic_position(0), 33)

    def test_overlapping_exons_fail(self):
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            TranscriptCoordinates('chr1', '+', [(10, 14), (13, 17)])


if __name__ == '__main__':
    unittest.main()
