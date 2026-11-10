"""Independent grouped scans retain legacy features and release alignment contexts."""
from array import array
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import weakref
from unittest.mock import patch

from utils.features.bam import decode_mods, extract_bam_features
from utils.io.cigar import CigarIndex

AVAILABLE = all(importlib.util.find_spec(name) for name in ('pysam', 'pyarrow'))


@unittest.skipUnless(AVAILABLE, 'pysam/pyarrow required')
class GroupedBamFeaturesTest(unittest.TestCase):
    def write_fixture(self, root):
        import pysam
        fasta = root / 'reference.fa'
        fasta.write_text('>chr1\n' + 'A' * 160100 + '\n')
        pysam.faidx(str(fasta))
        header = pysam.AlignmentHeader.from_references(['chr1'], [160100])

        def make_read(name, start, cigar, sequence, mm=None, ml=None, reverse=False):
            read = pysam.AlignedSegment(header)
            read.query_name, read.query_sequence = name, sequence
            read.reference_id, read.reference_start = 0, start
            read.flag, read.mapping_quality = 16 if reverse else 0, 60
            read.cigarstring = cigar
            read.query_qualities = array('B', [30] * len(sequence))
            if mm is not None:
                read.set_tag('MM', mm)
                read.set_tag('ML', array('B', ml))
            return read

        reads = [
            make_read('long_splice', 99, '3M59998N4M39996N6M', 'A' * 13,
                      'A+a?,1,2,3;A+17596?,1,2,3;', [201, 202, 203, 101, 102, 103]),
            make_read('nearby', 10099, '5M', 'A' * 5, 'A+a?,0;', [180]),
            make_read('irrelevant_gap', 30000, '5M', 'A' * 5),
            make_read('same_name', 60100, '6M', 'A' * 6, 'A+a?,1;A+17596?,1;', [190, 90]),
            make_read('same_name', 60100, '6M', 'A' * 6, 'A+a?,1;A+17596?,1;', [195, 95]),
            make_read('reverse', 60100, '6M', 'T' * 6, 'A+a?,4;A+17596?,4;', [210, 110], True),
            make_read('central_deletion', 60100, '1M1D3M', 'A' * 4),
        ]
        bam = root / 'reads.bam'
        with pysam.AlignmentFile(str(bam), 'wb', header=header) as handle:
            for read in sorted(reads, key=lambda value: value.reference_start):
                handle.write(read)
        pysam.index(str(bam))
        sites = root / 'sites.tsv'
        with sites.open('w', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['row_id', 'sample_id', 'chrom', 'start', 'end', 'strand', 'modification'])
            for position in (100, 10099, 20099, 60101, 100101, 150000):
                for target in ('m6A', 'inosine'):
                    writer.writerow([f'{position}:+:{target}', 'sample', 'chr1', position, position+1, '+', target])
                if position == 60101:
                    writer.writerow([f'{position}:-:m6A', 'sample', 'chr1', position, position+1, '-', 'm6A'])
        return sites, bam, fasta

    def read_bags(self, directory):
        import pyarrow.parquet as pq
        from tests.raw_feature_fixture import RawReadAudit
        report = json.loads((directory / 'run.json').read_text())
        store = RawReadAudit(directory)
        result = {}
        for entry in report['parquet_files']['sites']:
            path = directory / 'sites' / entry['path']
            for site in pq.ParquetFile(path).read().to_pylist():
                features = store.rows(site['read_offset'], site['retained_depth'], site['row_id']) if site['retained_depth'] else []
                site.pop('read_offset')
                result[site['row_id']] = (site, features)
        return result

    def test_gap_boundary_ignores_strand_and_target(self):
        from utils.features.grouping import candidate_groups
        sites = [dict(chrom='chr1', start=position, strand=strand, modification=target)
                 for position, strand, target in [
                     (100, '+', 'm6A'), (100, '-', 'inosine'), (10099, '+', 'm6A'),
                     (20099, '+', 'inosine'), (20099, '-', 'm6A'), (30098, '+', 'm6A')]]
        self.assertEqual(candidate_groups(sites, 10000), [(0, 3), (3, 6)])

    def test_span_limit_splits_a_single_dense_group_and_bounds_merged_fetches(self):
        from utils.features.grouping import candidate_groups, candidate_group_batches
        sites = [dict(start=position) for position in (0, 10, 24, 25, 40, 50)]
        groups = candidate_groups(sites, 10000, max_fetch_span=25)
        self.assertEqual(groups, [(0, 3), (3, 5), (5, 6)])
        self.assertEqual(list(candidate_group_batches(sites, groups, 16, 25)),
                         [(0, 3, 1), (3, 5, 1), (5, 6, 1)])
        for invalid in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, 'max_fetch_span'):
                candidate_groups([], 10000, max_fetch_span=invalid)

    def test_grouped_matches_scan_and_recomputes_cross_group_reads(self):
        import pysam
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sites, bam, fasta = self.write_fixture(root)
            legacy = extract_bam_features(sites, 'sample', bam, fasta, root / 'legacy',
                flank=2, partition_by_locus=True, parquet_batch_rows=2, parquet_file_rows=3)
            fetch_calls = []
            original_alignment_file = pysam.AlignmentFile

            class IndexedFetchOnly:
                def __init__(self, handle):
                    self.handle = handle

                def __enter__(self):
                    return self

                def __exit__(self, *arguments):
                    self.handle.close()

                def __getattr__(self, name):
                    return getattr(self.handle, name)

                def fetch(self, *arguments, **keywords):
                    if keywords.get('until_eof'):
                        raise AssertionError('Grouped mode must use indexed region fetches')
                    fetch_calls.append((arguments, keywords))
                    return self.handle.fetch(*arguments, **keywords)

            def open_bam(*arguments, **keywords):
                return IndexedFetchOnly(original_alignment_file(*arguments, **keywords))

            with patch('pysam.AlignmentFile', side_effect=open_bam), \
                    patch('utils.features.bam.CigarIndex', wraps=CigarIndex) as coordinate_calls, \
                    patch('utils.features.bam.decode_mods', wraps=decode_mods) as decode_calls:
                grouped = extract_bam_features(sites, 'sample', bam, fasta, root / 'grouped',
                    flank=2, partition_by_locus=True, parquet_batch_rows=2, parquet_file_rows=3,
                    candidate_group_gap=10000, read_batch_size=2)
            # Re-fetching the spanning alignment is allowed; decoding stays shared within a group.
            self.assertEqual(coordinate_calls.call_count, 9)
            self.assertEqual(decode_calls.call_count, 7)
            self.assertEqual(grouped['counts']['coordinate_index_builds'], 9)
            self.assertEqual(grouped['counts']['modification_decodes'], 7)
            self.assertEqual(legacy['counts']['coordinate_index_builds'], 6)
            self.assertEqual(legacy['counts']['modification_decodes'], 5)
            self.assertEqual(len(fetch_calls), 5)
            expected, actual = self.read_bags(root / 'legacy'), self.read_bags(root / 'grouped')
            self.assertEqual(set(expected), set(actual))
            for row_id, (expected_site, expected_reads) in expected.items():
                site, reads = actual[row_id]
                self.assertEqual(site, expected_site)
                self.assertCountEqual(reads, expected_reads)
            site, reads = actual['60101:+:m6A']
            self.assertEqual(site['central_deletions'], 1)
            self.assertEqual(site['retained_depth'], 3)
            repeated_names = [row for row in reads if row['read_id'] == 'same_name']
            self.assertEqual(len(repeated_names), 2)
            self.assertEqual({row['ml_byte'][2][0] for row in repeated_names}, {190, 195})
            self.assertEqual(actual['20099:+:m6A'][0]['central_refskips'], 1)
            self.assertEqual(actual['150000:+:m6A'][0]['retained_depth'], 0)

    def test_intronic_group_does_not_prevent_later_exon_reprocessing(self):
        import pysam
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / 'reference.fa'
            fasta.write_text('>chr1\n' + 'A' * 60000 + '\n')
            pysam.faidx(str(fasta))
            bam = root / 'reads.bam'
            header = pysam.AlignmentHeader.from_references(['chr1'], [60000])
            read = pysam.AlignedSegment(header)
            read.query_name, read.query_sequence = 'intron_then_exon', 'A' * 10
            read.flag, read.reference_id, read.reference_start = 0, 0, 10
            read.mapping_quality, read.cigarstring = 60, '5M50000N5M'
            read.query_qualities = array('B', [30] * 10)
            read.set_tag('MM', 'A+a?,6;')
            read.set_tag('ML', array('B', [222]))
            with pysam.AlignmentFile(str(bam), 'wb', header=header) as handle:
                handle.write(read)
            pysam.index(str(bam))
            sites = root / 'sites.tsv'
            sites.write_text('row_id\tsample_id\tchrom\tstart\tend\tstrand\tmodification\n'
                             'intron\tsample\tchr1\t1000\t1001\t+\tm6A\n'
                             'exon\tsample\tchr1\t50016\t50017\t+\tm6A\n')
            with patch('utils.features.bam.CigarIndex', wraps=CigarIndex) as coordinates, \
                    patch('utils.features.bam.decode_mods', wraps=decode_mods) as modifications:
                extract_bam_features(sites, 'sample', bam, fasta, root / 'features',
                    flank=2, candidate_group_gap=10000, read_batch_size=1)
            self.assertEqual(coordinates.call_count, 2)
            self.assertEqual(modifications.call_count, 1)
            bags = self.read_bags(root / 'features')
            self.assertEqual(bags['intron'][0]['central_refskips'], 1)
            self.assertEqual(bags['intron'][0]['retained_depth'], 0)
            self.assertEqual(bags['exon'][0]['retained_depth'], 1)
            self.assertEqual(bags['exon'][1][0]['ml_byte'][2][0], 222)

    def test_one_fetch_merges_both_strands_and_groups_without_encoding_gap_reads(self):
        from tests.test_bam_reservoir import MemorySiteSink
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sites, bam, fasta = self.write_fixture(root)
            for limit in (None, 1):
                with self.subTest(max_reads=limit):
                    expected, actual = MemorySiteSink(), MemorySiteSink()
                    extract_bam_features(sites, 'sample', bam, fasta, root / f'scan_{limit}',
                                         flank=2, max_reads=limit, site_sink=expected)
                    with patch('utils.features.bam.CigarIndex', wraps=CigarIndex) as coordinates, \
                            patch('utils.features.bam.decode_mods', wraps=decode_mods) as modifications:
                        report = extract_bam_features(sites, 'sample', bam, fasta, root / f'fetch_{limit}',
                            flank=2, max_reads=limit, site_sink=actual, candidate_group_gap=10000,
                            group_batch_size=16, read_batch_size=2)
                    self.assertEqual(report['counts']['bam_fetches'], 1)
                    self.assertEqual(report['counts']['candidate_groups_completed'], 5)
                    self.assertEqual(report['counts']['bam_records_fetched'], 7)
                    self.assertEqual(coordinates.call_count, 6)
                    names = [call.args[0].query_name for call in coordinates.call_args_list]
                    self.assertNotIn('irrelevant_gap', names)
                    if limit is None:
                        self.assertEqual(modifications.call_count, 5)
                    self.assertEqual(expected.records, actual.records)

    def test_read_and_cigar_contexts_are_released_before_next_batch_fetch(self):
        import pysam
        from utils.features.bam import DeferredRead
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sites, bam, fasta = self.write_fixture(root)
            contexts, coordinates = weakref.WeakSet(), weakref.WeakSet()
            original_open = pysam.AlignmentFile
            fetches = []

            def track_context(read, cigar):
                context = DeferredRead(read, cigar)
                contexts.add(context)
                return context

            def track_coordinates(read):
                index = CigarIndex(read)
                coordinates.add(index)
                return index

            class CheckedBam:
                def __init__(self, handle):
                    self.handle = handle

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.handle.close()

                def __getattr__(self, name):
                    return getattr(self.handle, name)

                def fetch(inner, *args, **kwargs):
                    self.assertEqual(len(contexts), 0)
                    self.assertEqual(len(coordinates), 0)
                    fetches.append(args)
                    return inner.handle.fetch(*args, **kwargs)

            with patch('pysam.AlignmentFile', side_effect=lambda *a, **k: CheckedBam(original_open(*a, **k))), \
                    patch('utils.features.bam.DeferredRead', new=track_context), \
                    patch('utils.features.bam.CigarIndex', new=track_coordinates):
                extract_bam_features(sites, 'sample', bam, fasta, root / 'features',
                    flank=2, candidate_group_gap=10000, group_batch_size=2, read_batch_size=2)
            self.assertEqual(len(fetches), 3)
            self.assertEqual(len(contexts), 0)
            self.assertEqual(len(coordinates), 0)

    def test_span_safety_limit_preserves_all_sites_and_sampled_features(self):
        from tests.test_bam_reservoir import MemorySiteSink
        import pysam
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sites, bam, fasta = self.write_fixture(root)
            expected, actual = MemorySiteSink(), MemorySiteSink()
            extract_bam_features(sites, 'sample', bam, fasta, root / 'scan',
                                 flank=2, max_reads=1, site_sink=expected)
            original_open = pysam.AlignmentFile
            fetches = []

            class BoundedBam:
                def __init__(inner, handle):
                    inner.handle = handle

                def __enter__(inner):
                    return inner

                def __exit__(inner, *args):
                    inner.handle.close()

                def __getattr__(inner, name):
                    return getattr(inner.handle, name)

                def fetch(inner, chrom, start, end):
                    self.assertLessEqual(end - start, 5000)
                    fetches.append((chrom, start, end))
                    return inner.handle.fetch(chrom, start, end)

            with patch('pysam.AlignmentFile', side_effect=lambda *a, **k: BoundedBam(original_open(*a, **k))):
                report = extract_bam_features(sites, 'sample', bam, fasta, root / 'bounded',
                    flank=2, max_reads=1, site_sink=actual, candidate_group_gap=10000,
                    group_batch_size=16, max_fetch_span=5000, read_batch_size=2)
            self.assertEqual(len(fetches), 6)
            self.assertEqual(report['counts']['candidate_groups_completed'], 6)
            self.assertEqual(actual.records, expected.records)


if __name__ == '__main__':
    unittest.main()
