"""Bounded per-site sampling before feature encoding; real coordinates, no padding."""
from array import array
from collections import Counter
from copy import deepcopy
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.features.bam import (decode_mods, extract_bam_features, new_site_bag,
                                reservoir_slot)
from utils.io.cigar import CigarIndex

AVAILABLE = all(importlib.util.find_spec(name) for name in ('pysam', 'pyarrow'))


class MemorySiteSink:
    def __init__(self):
        self.records = {}
        self.files = []
        self.closed = False

    def append_site(self, site, reads, reference_sequence, eligible_depth,
                    central_deletions, central_refskips, *, sequence_repeat_mask):
        self.records[site['row_id']] = dict(site=deepcopy(site), reads=deepcopy(reads),
            reference_sequence=reference_sequence, eligible_depth=eligible_depth,
            central_deletions=central_deletions, central_refskips=central_refskips)

    def close(self):
        self.closed = True

    def abort(self):
        pass


class ReservoirSamplingTest(unittest.TestCase):
    def sample(self, count, limit, seed=42, **metadata):
        site = dict(sample_id='sample', chrom='chr1', start=12, strand='+', **metadata)
        bag = new_site_bag(site, seed)
        for value in range(count):
            slot = reservoir_slot(bag, limit)
            if slot is None:
                continue
            if slot == len(bag['reads']):
                bag['reads'].append(value)
            else:
                bag['reads'][slot] = value
            self.assertLessEqual(len(bag['reads']), limit)
        self.assertEqual(bag['eligible_depth'], count)
        return bag['reads']

    def test_cap_reproducibility_no_duplicates_and_no_label_dependence(self):
        selected = self.sample(10000, 300)
        self.assertEqual(len(set(selected)), 300)
        self.assertEqual(selected, self.sample(10000, 300, label='2'))
        self.assertNotEqual(selected, self.sample(10000, 300, seed=43))
        for count in (0, 1, 10, 299, 300):
            self.assertEqual(self.sample(count, 300), list(range(count)))

    def test_early_and_late_reads_have_similar_inclusion_frequency(self):
        counts = Counter()
        for seed in range(1000):
            counts.update(self.sample(30, 3, seed))
        # Expected inclusion count is 100/1,000 trials for every read, including
        # the earliest and latest. Fixed seeds make this check reproducible.
        self.assertEqual(set(counts), set(range(30)))
        self.assertTrue(all(60 <= count <= 145 for count in counts.values()), counts)


@unittest.skipUnless(AVAILABLE, 'pysam/pyarrow required')
class BamReservoirTest(unittest.TestCase):
    def fixture(self, root):
        import pysam
        fasta = root / 'ref.fa'
        fasta.write_text('>chr1\n' + 'A' * 100 + '\n')
        pysam.faidx(str(fasta))
        header = pysam.AlignmentHeader.from_references(['chr1'], [100])
        bam = root / 'reads.bam'
        with pysam.AlignmentFile(str(bam), 'wb', header=header) as handle:
            for index in range(600):
                read = pysam.AlignedSegment(header)
                read.query_name = f'forward_{index}'
                read.query_sequence = 'A' * 10
                read.reference_id, read.reference_start = 0, 10
                read.flag, read.mapping_quality, read.cigarstring = 0, 60, '10M'
                read.query_qualities = array('B', [30] * 10)
                read.set_tag('MM', 'A+a?,2;')
                read.set_tag('ML', array('B', [index % 256]))
                handle.write(read)
            for index in range(10):
                read = pysam.AlignedSegment(header)
                read.query_name = f'reverse_{index}'
                read.query_sequence = 'T' * 10
                read.reference_id, read.reference_start = 0, 10
                read.flag, read.mapping_quality, read.cigarstring = 16, 60, '10M'
                read.query_qualities = array('B', [30] * 10)
                handle.write(read)
            for cigar in ('2M2D6M', '2M2N6M'):
                read = pysam.AlignedSegment(header)
                read.query_name = cigar
                read.query_sequence = 'A' * 8
                read.reference_id, read.reference_start = 0, 10
                read.flag, read.mapping_quality, read.cigarstring = 0, 60, cigar
                handle.write(read)
        pysam.index(str(bam))
        sites = root / 'sites.tsv'
        with sites.open('w', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['row_id', 'sample_id', 'chrom', 'start', 'end', 'strand', 'modification'])
            writer.writerow(['plus', 'sample', 'chr1', 12, 13, '+', 'adenine'])
            writer.writerow(['minus', 'sample', 'chr1', 12, 13, '-', 'adenine'])
            writer.writerow(['empty', 'sample', 'chr1', 50, 51, '+', 'adenine'])
        return sites, bam, fasta

    def test_sink_stores_only_real_retained_reads_and_full_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sites, bam, fasta = self.fixture(root)
            sink = MemorySiteSink()
            with patch('utils.features.bam.CigarIndex', wraps=CigarIndex) as coordinates, \
                    patch('utils.features.bam.decode_mods', wraps=decode_mods) as mods:
                report = extract_bam_features(sites, 'sample', bam, fasta, root / 'out',
                    flank=2, max_reads=300, sampling_seed=42, site_sink=sink)
            self.assertEqual(coordinates.call_count, 612)
            self.assertLess(mods.call_count, 610)  # Rejected reads are not MM-decoded.
            self.assertEqual(report['counts']['peak_retained_reads_per_site'], 300)
            self.assertEqual(report['counts']['eligible_read_site_pairs'], 610)
            self.assertEqual(report['counts']['retained_read_site_pairs'], 310)
            self.assertEqual(report['schema'], 'site_bags_parquet_v3')
            self.assertEqual(report['max_reads'], 300)
            self.assertEqual(report['sampling_seed'], 42)
            self.assertTrue(sink.closed)
            plus = sink.records['plus']
            self.assertEqual(plus['eligible_depth'], 600)
            self.assertEqual(len(plus['reads']), 300)
            self.assertEqual((plus['central_deletions'], plus['central_refskips']), (1, 1))
            self.assertEqual(sink.records['minus']['eligible_depth'], 10)
            self.assertEqual(len(sink.records['minus']['reads']), 10)
            self.assertEqual(sink.records['empty']['reads'], [])
            self.assertFalse((root / 'out/reads').exists())
            self.assertFalse((root / 'out/sites').exists())
            self.assertIsInstance(plus['reads'][0]['mm_skip_modes'], dict)
            again = MemorySiteSink()
            extract_bam_features(sites, 'sample', bam, fasta, root / 'again',
                flank=2, max_reads=300, sampling_seed=42, site_sink=again)
            self.assertEqual(plus, again.records['plus'])

    def test_sampling_matches_single_scan_with_independent_groups(self):
        from tests.test_grouped_bam_features import GroupedBamFeaturesTest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sites, bam, fasta = GroupedBamFeaturesTest().write_fixture(root)
            full, grouped = MemorySiteSink(), MemorySiteSink()
            extract_bam_features(sites, 'sample', bam, fasta, root / 'full',
                                 flank=2, max_reads=1, site_sink=full)
            with patch('utils.features.bam.CigarIndex', wraps=CigarIndex) as coordinates:
                extract_bam_features(sites, 'sample', bam, fasta, root / 'grouped',
                    flank=2, max_reads=1, site_sink=grouped, candidate_group_gap=10000, read_batch_size=2)
            self.assertEqual(coordinates.call_count, 9)
            self.assertEqual(full.records, grouped.records)
