"""Final-only decoding, shared alignment work, exact sampling and memory release."""
from array import array
from collections import Counter
import csv
import gc
from io import StringIO
from pathlib import Path
import tempfile
import unittest
import weakref
from unittest.mock import patch

import pysam

from test_bam_reservoir import MemorySiteSink
from utils.features import bam as features


class GuardedRead:
    """Fail if sampling touches sequence/quality before final reservoirs exist."""
    def __init__(self, read):
        self.read = read
        self.allow_features = False
        self.sequence_calls = self.quality_calls = 0

    def __getattr__(self, name):
        return getattr(self.read, name)

    @property
    def query_sequence(self):
        if not self.allow_features:
            raise AssertionError('Sequence converted during sampling')
        self.sequence_calls += 1
        return self.read.query_sequence

    @property
    def query_qualities(self):
        if not self.allow_features:
            raise AssertionError('Quality converted during sampling')
        self.quality_calls += 1
        return self.read.query_qualities


class DeferredReadFeaturesTest(unittest.TestCase):
    def test_only_final_read_union_is_decoded_once_and_contexts_are_released(self):
        with tempfile.TemporaryDirectory() as temporary:
            fasta_path = Path(temporary) / 'reference.fa'
            fasta_path.write_text('>chr1\n' + 'A' * 100 + '\n')
            pysam.faidx(str(fasta_path))
            header = pysam.AlignmentHeader.from_references(['chr1'], [100])
            reads = []
            for index in range(1000):
                read = pysam.AlignedSegment(header)
                read.query_name, read.query_sequence = str(index), 'A' * 30
                read.reference_id, read.reference_start, read.flag = 0, 0, 0
                read.cigarstring, read.mapping_quality = '30M', 60
                read.query_qualities = array('B', [30] * 30)
                read.set_tag('MM', 'A+a?,10,1;A+17596?,10,1;')
                read.set_tag('ML', array('B', [index % 256] * 4))
                reads.append(GuardedRead(read))
            sites = [dict(row_id=str(center), sample_id='sample', chrom='chr1', start=center,
                          end=center+1, strand='+', modification='adenine') for center in (10, 12)]
            pending, counts, channels = {}, Counter(), Counter()
            contexts = weakref.WeakSet()
            constructor = features.DeferredRead

            def track_context(read, coordinates):
                context = constructor(read, coordinates)
                contexts.add(context)
                return context

            with pysam.FastaFile(str(fasta_path)) as fasta, \
                    patch.object(features, 'DeferredRead', side_effect=track_context), \
                    patch.object(features, 'CigarIndex', wraps=features.CigarIndex) as cigars, \
                    patch.object(features, 'decode_mods', wraps=features.decode_mods) as mods, \
                    patch.object(features, 'encode_read', wraps=features.encode_read) as encoded:
                for read in reads:
                    features.collect_read_features(read, sites, [10, 12], pending,
                                                   9, counts, 300, 42)
                self.assertEqual(cigars.call_count, 1000)
                self.assertEqual(mods.call_count, 0)
                self.assertEqual(encoded.call_count, 0)
                selected = {context.read.query_name for bag in pending.values() for context in bag['reads']}
                self.assertGreater(len(selected), 300)  # The two sites have independent reservoirs.
                self.assertLess(len(selected), 600)  # Some alignments are shared by both sites.
                for read in reads:
                    read.allow_features = True
                    if read.query_name not in selected:
                        # Unselected MM tags must never be parsed, even if malformed.
                        read.read.set_tag('MM', 'invalid;')
                sink = MemorySiteSink()
                for index, site in enumerate(sites):
                    bag = pending.pop(('chr1', index))
                    features.write_site_features(site, bag, fasta, 9, None, None,
                        csv.writer(StringIO()), counts, sink, channels=channels)
                    self.assertEqual(bag['reads'], [])
                self.assertEqual(cigars.call_count, 1000)
                decoded_ids = [call.args[0].query_name for call in mods.call_args_list]
                self.assertEqual(set(decoded_ids), selected)
                self.assertEqual(len(decoded_ids), len(selected))
                self.assertEqual(encoded.call_count, 600)
                self.assertEqual(counts['encoded_read_site_windows'], 600)
                self.assertEqual(counts['retained_read_site_pairs'], 600)
                self.assertEqual(counts['eligible_read_site_pairs'], 2000)
                self.assertGreater(counts['reservoir_admissions'], 600)
                self.assertTrue(channels['m6A'] > 0)
                for read in reads:
                    expected = int(read.query_name in selected)
                    self.assertEqual(read.sequence_calls, expected)
                    self.assertEqual(read.quality_calls, expected)
            gc.collect()
            self.assertEqual(len(contexts), 0)


if __name__ == '__main__':
    unittest.main()
