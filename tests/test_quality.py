from array import array
from collections import Counter
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from utils.features.bam import MODIFICATIONS
from utils.quality.stats import (
    BASES, PROBABILITY_UNITS, collect_read_distributions, estimate_quality,
    histogram_quantile, sample_read,
)


class QualityHelpersTest(unittest.TestCase):
    def test_empirical_quantiles_keep_ties_and_empty_unknown(self):
        self.assertIsNone(histogram_quantile([0, 0, 0], .1))
        self.assertEqual(histogram_quantile([0, 9, 1], .1), 1)
        self.assertEqual(histogram_quantile([0, 9, 1], .9), 1)
        self.assertEqual(histogram_quantile([0, 9, 1], 1), 2)
        self.assertEqual(histogram_quantile([0, 9, 1], 0), 1)

    def test_name_sampling_is_reproducible_and_order_independent(self):
        names = [f'read{i}' for i in range(200)]
        first = {name for name in names if sample_read(name, .2, 42)}
        second = {name for name in reversed(names) if sample_read(name, .2, 42)}
        self.assertEqual(first, second)
        self.assertTrue(10 < len(first) < 90)
        self.assertNotEqual(first, {name for name in names if sample_read(name, .2, 43)})
        self.assertTrue(all(sample_read(name, 1, 42) for name in names))


@unittest.skipUnless(importlib.util.find_spec('pysam'), 'pysam is not installed')
class QualityDistributionTest(unittest.TestCase):
    def read(self, sequence, mm, ml=None, *, reverse=False, cigar=None):
        import pysam
        header = pysam.AlignmentHeader.from_references(['chr1'], [1000])
        read = pysam.AlignedSegment(header)
        read.query_name = 'synthetic'
        stored = sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1] if reverse else sequence
        read.query_sequence = stored
        read.query_qualities = array('B', [30] * len(sequence))
        read.flag = 16 if reverse else 0
        read.reference_id, read.reference_start, read.mapping_quality = 0, 100, 60
        read.cigarstring = cigar or f'{len(sequence)}M'
        read.set_tag('MM', mm)
        read.set_tag('MN', len(sequence))
        if ml is not None:
            read.set_tag('ML', array('B', ml))
        return read

    def collect(self, read, only_aligned=True):
        raw = {modification: [0] * 256 for modification in MODIFICATIONS}
        confidence = {base: [0] * 513 for base in BASES}
        winners = {(base, 'canonical'): [0] * 513 for base in BASES}
        from utils.features.bam import CANONICAL
        for modification in MODIFICATIONS:
            winners[CANONICAL[modification], modification] = [0] * 513
        counts = Counter()
        collect_read_distributions(read, raw, confidence, winners, counts, only_aligned)
        return raw, confidence, winners, counts

    def test_all_eight_channels_and_canonical_confidence(self):
        mm = ('A+a?,0;A+17596?,0;A+69426?,0;C+m?,0;C+19228?,0;'
              'G+19229?,0;T+17802?,0;T+19227?,0;')
        scores = [128, 64, 32, 64, 64, 0, 200, 30]
        read = self.read('ACGT', mm, scores)
        original = read.to_string()
        raw, confidence, winners, counts = self.collect(read)
        self.assertTrue(all(sum(histogram) == 1 for histogram in raw.values()))
        self.assertEqual(confidence['A'][257], 1)
        self.assertEqual(confidence['C'][254], 1)
        self.assertEqual(confidence['G'][511], 1)
        self.assertEqual(confidence['T'][401], 1)
        self.assertEqual(winners['A', 'm6A'][257], 1)
        self.assertEqual(winners['C', 'canonical'][254], 1)
        self.assertEqual(counts['observed_ml_values'], 8)
        self.assertEqual(counts['confidence_positions'], 4)
        self.assertEqual(read.to_string(), original)

    def test_unknown_and_inferred_canonical_are_not_measured_ml(self):
        adenine_channels = 'A+17596.;A+69426.;'
        raw, confidence, _, counts = self.collect(self.read('AAA', 'A+a?,0;' + adenine_channels, [220]))
        self.assertEqual(sum(raw['m6A']), 1)
        self.assertEqual(sum(confidence['A']), 1)
        self.assertEqual(counts['positions_with_unknown_probabilities'], 2)
        raw, confidence, _, counts = self.collect(self.read('AAA', 'A+a.,0;' + adenine_channels, [220]))
        self.assertEqual(sum(raw['m6A']), 1)
        self.assertEqual(confidence['A'][512], 2)
        raw, confidence, _, counts = self.collect(self.read('AAA', 'A+a;' + adenine_channels))
        self.assertEqual(sum(raw['m6A']), 0)
        self.assertEqual(confidence['A'][512], 3)
        raw, confidence, _, counts = self.collect(self.read('AAA', 'A+a?,0;' + adenine_channels))
        self.assertEqual(sum(raw['m6A']), 0)
        self.assertEqual(sum(confidence['A']), 0)
        self.assertEqual(counts['explicit_unknown_scores'], 1)

    def test_invalid_probability_sum_and_unknown_mod_do_not_fabricate_canonical(self):
        read = self.read('AAA', 'A+a?,0,0,0;A+17596?,0,0,0;A+69426.;', [200, 128, 0, 200, 128, 0])
        raw, confidence, _, counts = self.collect(read)
        self.assertEqual(sum(raw['m6A']), 3)
        self.assertEqual(counts['positions_with_probability_sum_above_one'], 1)
        self.assertEqual(counts['positions_with_quantization_clamp'], 1)
        self.assertEqual(confidence['A'][257], 1)
        self.assertEqual(confidence['A'][510], 1)
        read = self.read('A', 'A+az?,0;', [100, 20])
        raw, confidence, _, counts = self.collect(read)
        self.assertEqual(raw['m6A'][100], 1)
        self.assertEqual(sum(confidence['A']), 0)
        self.assertEqual(counts['positions_with_unsupported_channels'], 1)

    def test_reverse_orientation_and_softclips_use_stored_cigar_once(self):
        from unittest.mock import patch
        from utils.io.cigar import CigarIndex
        mm = 'A+a?,0,0,0,0;A+17596.;A+69426.;'
        scores = [10, 20, 30, 40]
        # Original AAAA maps to stored TTTT; the last original base is soft-clipped.
        read = self.read('AAAA', mm, scores, reverse=True, cigar='1S3M')
        with patch('utils.quality.stats.CigarIndex', wraps=CigarIndex) as coordinates:
            raw, confidence, _, counts = self.collect(read)
            self.assertEqual(coordinates.call_count, 1)
        self.assertEqual([raw['m6A'][score] for score in scores], [1, 1, 1, 0])
        self.assertEqual(sum(confidence['A']), 3)
        raw, confidence, _, counts = self.collect(read, only_aligned=False)
        self.assertEqual(sum(raw['m6A']), 4)
        self.assertEqual(sum(confidence['A']), 4)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib is not installed')
    def test_end_to_end_exports_histograms_thresholds_and_provenance(self):
        import pysam
        from utils.references.catalog import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'reads.bam'
            read = self.read('AAAA', 'A+a?,0,0,0,0;A+17596.;A+69426.;', [10, 20, 30, 40])
            with pysam.AlignmentFile(str(path), 'wb', header=read.header) as bam:
                bam.write(read)
            before = sha256(path)
            report = estimate_quality(path, root / 'quality', percentile=.5)
            self.assertEqual(report['counts']['records_scanned'], 1)
            self.assertEqual(report['counts']['confidence_positions'], 4)
            self.assertEqual(report['filter_thresholds']['A'], (512 - 61) / 512)
            self.assertIsNone(report['filter_thresholds']['C'])
            self.assertEqual(report['raw_channel_quantiles']['m6A'], 20.5 / 256)
            self.assertEqual(report['inherited_modification_thresholds']['m6A'], report['filter_thresholds']['A'])
            self.assertIsNone(report['inherited_modification_thresholds']['inosine'])
            self.assertNotIn('modification_thresholds', report)
            for filename in ('raw_ml_histograms.tsv', 'confidence_histograms.tsv',
                             'winner_histograms.tsv', 'thresholds.json', 'run.json', 'histograms.png'):
                self.assertTrue((root / 'quality' / filename).is_file())
            self.assertFalse((root / 'quality/INCOMPLETE.txt').exists())
            saved = json.loads((root / 'quality/thresholds.json').read_text())
            self.assertEqual(saved['schema'], 'modification_quality_distributions_v1')
            self.assertEqual(sha256(path), before)
            with self.assertRaises(FileExistsError):
                estimate_quality(path, root / 'quality')

    def test_bad_tags_leave_failed_output_and_invalid_options_fail_early(self):
        import pysam
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'reads.bam'
            read = self.read('AAAA', 'A+a?,0;', [20])
            read.set_tag('MN', 5)
            with pysam.AlignmentFile(str(path), 'wb', header=read.header) as bam:
                bam.write(read)
            with self.assertRaisesRegex(ValueError, 'MN/SEQ'):
                estimate_quality(path, root / 'bad')
            self.assertTrue((root / 'bad/FAILED.txt').exists())
            self.assertTrue((root / 'bad/INCOMPLETE.txt').exists())
            self.assertFalse((root / 'bad/run.json').exists())
            with self.assertRaises(ValueError):
                estimate_quality(path, root / 'options', sample_fraction=0)
            self.assertFalse((root / 'options').exists())

    def test_incomplete_same_base_channels_do_not_imply_canonical(self):
        raw, confidence, winners, counts = self.collect(self.read('AAA', 'A+a.,0;', [220]))
        self.assertEqual(raw['m6A'][220], 1)
        self.assertEqual(sum(confidence['A']), 0)
        self.assertEqual(sum(winners['A', 'canonical']), 0)
        self.assertEqual(counts['positions_with_incomplete_channels'], 3)
