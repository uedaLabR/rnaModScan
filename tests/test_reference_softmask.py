from pathlib import Path
import tempfile
import unittest

import pysam

from utils.references.softmask import softmasked_window


class ReferenceSoftmaskTest(unittest.TestCase):
    def test_case_reverse_strand_unknown_and_chromosome_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'genome.fa'
            path.write_text('>chr1\naCgTACgtNnAAA\n')
            pysam.faidx(str(path))
            with pysam.FastaFile(str(path)) as reference:
                self.assertEqual(softmasked_window(reference, 'chr1', 2, '+', 5),
                                 ('ACGTA', [True, False, True, False, False]))
                self.assertEqual(softmasked_window(reference, 'chr1', 2, '-', 5),
                                 ('TACGT', [False, False, True, False, True]))
                self.assertEqual(softmasked_window(reference, 'chr1', 0, '+', 5),
                                 ('NNACG', [False, False, True, False, True]))
                self.assertEqual(softmasked_window(reference, 'chr1', 9, '+', 5)[1],
                                 [True, False, False, False, False])


if __name__ == '__main__':
    unittest.main()
