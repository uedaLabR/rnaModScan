import csv
import json
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest.mock import patch

try:
    import pysam
    import pyarrow.parquet as pq
except ImportError:
    pysam = None

from utils.features.bam import MODIFICATIONS, decode_mods, extract_bam_features, read_map
from utils.features.parquet import iter_feature_batches, ShardedParquetWriter, feature_schema


@unittest.skipIf(pysam is None, 'pysam/pyarrow not installed')
class BamFeaturesTest(unittest.TestCase):
    def make_read(self, header, name, pos, seq, cigar, mm=None, ml=None, reverse=False):
        read = pysam.AlignedSegment(header)
        read.query_name, read.query_sequence = name, seq
        read.flag, read.reference_id, read.reference_start = 16 if reverse else 0, 0, pos
        read.mapping_quality, read.cigarstring = 60, cigar
        read.query_qualities = array('B', [30] * len(seq))
        if mm is not None:
            read.set_tag('MM', mm)
            read.set_tag('MN', len(seq))
        if ml is not None:
            read.set_tag('ML', array('B', ml))
        return read

    def test_reverse_mod_positions_and_all_eight_channels(self):
        header = pysam.AlignmentHeader.from_references(['chr1'], [100])
        codes = 'A+a?,0;A+17596?,0;C+m?,0;T+17802?,0;A+69426?,0;C+19228?,0;G+19229?,0;T+19227?,0;'
        for reverse in (False, True):
            # Original sequence ACGT -> BAM stored ACGT after reverse complement too.
            read = self.make_read(header, 'r', 10, 'ACGT', '4M', codes, [0, 11, 22, 33, 44, 55, 66, 255], reverse)
            mods, modes = decode_mods(read)
            self.assertEqual(set(modes), set(MODIFICATIONS))
            aq = 3 if reverse else 0
            tq = 0 if reverse else 3
            self.assertEqual(mods[aq, 'm6A'], 0)
            self.assertEqual(mods[aq, 'inosine'], 11)
            self.assertEqual(mods[tq, '2OmeU'], 255)

    def test_cigar_softclips_indels_and_splice(self):
        header = pysam.AlignmentHeader.from_references(['chr1'], [100])
        read = self.make_read(header, 'r', 10, 'A' * 11, '2S3M1I2M2D4N3M')
        mapping, ins = read_map(read, 0, 40)
        self.assertEqual(mapping[10], (2, 'base'))
        self.assertEqual(mapping[13], (6, 'base'))
        self.assertEqual(ins[12], 1)
        self.assertEqual(mapping[15][1], 'deletion')
        self.assertEqual(mapping[17][1], 'refskip')
        self.assertEqual(mapping[21], (8, 'base'))

    def test_end_to_end_masks_orientation_and_no_label_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fa = root / 'ref.fa'
            fa.write_text('>chr1\n' + 'A' * 100 + '\n')
            pysam.faidx(str(fa))
            header = pysam.AlignmentHeader.from_references(['chr1'], [100])
            reads = [self.make_read(header, f'f{i}', 10, 'A' * 10, '10M',
                                    'A+a?,2;', [0 if i == 0 else 200]) for i in range(6)]
            reads.append(self.make_read(header, 'reverse', 10, 'A' * 10, '10M', 'T+17802?,7;', [250], True))
            reads.append(self.make_read(header, 'deletion', 10, 'A' * 8, '2M2D6M'))
            reads.append(self.make_read(header, 'splice', 10, 'A' * 8, '2M2N6M'))
            secondary = self.make_read(header, 'secondary', 10, 'A' * 10, '10M')
            secondary.flag = 256
            reads.append(secondary)
            bam = root / 'reads.bam'
            with pysam.AlignmentFile(str(bam), 'wb', header=header) as handle:
                for read in reads:
                    handle.write(read)
            pysam.index(str(bam))
            sites = root / 'sites.tsv'
            fields = ['row_id', 'sample_id', 'chrom', 'start', 'end', 'strand', 'modification', 'label', 'database_sources']
            with sites.open('w', newline='') as handle:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow(fields)
                writer.writerow(['plus', 'WT', 'chr1', 12, 13, '+', 'm6A', 1, 'RMBase'])
                writer.writerow(['minus', 'WT', 'chr1', 12, 13, '-', 'pseU', 0, ''])
                writer.writerow(['missing', 'WT', 'chr1', 50, 51, '+', 'm6A', '', ''])
            report = extract_bam_features(sites, 'WT', bam, fa, root / 'all', flank=2,
                                           parquet_batch_rows=2, parquet_file_rows=3)
            bags = {r['row_id']: r for r in pq.read_table(str(root / 'all/sites')).to_pylist()}
            read_rows = [r for batch in iter_feature_batches(root / 'all/reads') for r in batch.to_pylist()]
            for site in bags.values():
                site['reads'] = [r for r in read_rows if r['row_id'] == site['row_id']]
            self.assertEqual(len(report['parquet_files']['reads']), 3)
            self.assertEqual(sum(f['rows'] for f in report['parquet_files']['reads']), 7)
            self.assertIsNone(report['max_reads'])
            self.assertEqual(report['policies']['sampling'], 'all eligible reads; no subsampling')
            self.assertEqual(report['schema'], 'read_features_parquet_v3')
            self.assertEqual(pq.read_schema(str(root / 'all/reads/part-000000.parquet')).metadata[b'window_flank'], b'2')
            self.assertEqual(pq.ParquetFile(str(root / 'all/reads/part-000000.parquet')).metadata.row_group(0).column(0).compression, 'ZSTD')
            plus, minus = bags['plus'], bags['minus']
            self.assertEqual(plus['eligible_depth'], 6)
            self.assertEqual(plus['central_deletions'], 1)
            self.assertEqual(plus['central_refskips'], 1)
            self.assertNotIn('label', plus)
            self.assertNotIn('database_sources', plus)
            r = next(r for r in plus['reads'] if r['read_id'] == 'f0')
            self.assertEqual(r['ml_byte'][2][0], 0)
            self.assertIsNone(r['splice_distance_before'])
            self.assertIsNone(r['splice_distance_after'])
            self.assertTrue(r['modification_observed'][2][0])
            self.assertIsNone(r['ml_byte'][1][0])
            self.assertFalse(r['modification_observed'][1][0])
            self.assertIsNone(r['ml_byte'][2][2])
            self.assertFalse(r['modification_applicable'][2][2])
            self.assertEqual(minus['reference_sequence'], 'TTTTT')
            self.assertEqual(minus['reads'][0]['ml_byte'][2][3], 250)
            self.assertEqual(minus['reads'][0]['base'][2], 'T')
            self.assertEqual(bags['missing']['retained_depth'], 0)
            batches = list(iter_feature_batches(root / 'all/reads', columns=['row_id', 'ml_byte'], batch_size=1))
            self.assertTrue(all(b.schema.names == ['row_id', 'ml_byte'] and b.num_rows == 1 for b in batches))
            (root / 'all/INCOMPLETE.txt').write_text('incomplete')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                list(iter_feature_batches(root / 'all/reads'))

    def test_no_reads_still_has_typed_empty_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = ShardedParquetWriter(root / 'reads', feature_schema(2, MODIFICATIONS), batch_rows=2, file_rows=3)
            writer.close()
            table = pq.read_table(str(root / 'reads'))
            self.assertEqual(table.num_rows, 0)
            self.assertIn('ml_byte', table.schema.names)

    def test_invalid_modification_tags_fail(self):
        header = pysam.AlignmentHeader.from_references(['chr1'], [100])
        read = self.make_read(header, 'r', 10, 'AAAA', '4M', 'A+a?,0;', [100])
        read.set_tag('MN', 5)
        with self.assertRaisesRegex(ValueError, 'MN/SEQ'):
            decode_mods(read)

    def test_empty_numeric_groups_preserve_declarations_without_phantom_calls(self):
        header = pysam.AlignmentHeader.from_references(['chr1'], [100])
        mm = 'A+a?,0;A+17596.;A+69426?;'
        for reverse in (False, True):
            sequence = 'TTT' if reverse else 'AAA'
            read = self.make_read(header, 'empty_channels', 10, sequence, '3M', mm, [220], reverse)
            original = read.to_string()
            modifications, modes = decode_mods(read)
            query = 2 if reverse else 0
            self.assertEqual(modifications, {(query, 'm6A'): 220})
            self.assertEqual(modes, {'m6A': '?', 'inosine': '.', '2OmeA': '?'})
            self.assertEqual(read.to_string(), original)
        read = self.make_read(header, 'no_calls', 10, 'AAA', '3M', 'A+17596.;A+69426?;')
        modifications, modes = decode_mods(read)
        self.assertEqual(modifications, {})
        self.assertEqual(modes, {'inosine': '.', '2OmeA': '?'})

    def test_one_read_conversion_serves_distant_sites_and_multiple_targets(self):
        from utils.io.cigar import CigarIndex
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fa = root / 'ref.fa'
            fa.write_text('>chr1\n' + 'A' * 150100 + '\n')
            pysam.faidx(str(fa))
            header = pysam.AlignmentHeader.from_references(['chr1'], [150100])
            long_read = self.make_read(header, 'long_splice', 10, 'A' * 10, '5M100000N5M',
                                      'A+a?,2,4;A+17596?,2,4;', [220, 230, 120, 130])
            short_read = self.make_read(header, 'downstream', 100015, 'A' * 5, '5M',
                                       'A+a?,2;A+17596?,2;', [221, 121])
            bam = root / 'reads.bam'
            with pysam.AlignmentFile(str(bam), 'wb', header=header) as handle:
                handle.write(long_read)
                handle.write(short_read)
            pysam.index(str(bam))
            sites = root / 'sites.tsv'
            with sites.open('w', newline='') as handle:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow(['row_id', 'sample_id', 'chrom', 'start', 'end', 'strand', 'modification'])
                for start in (12, 100017, 150000):
                    for target in ('m6A', 'inosine'):
                        writer.writerow([f'{start}:{target}', 'WT', 'chr1', start, start+1, '+', target])
            with patch('utils.features.bam.CigarIndex', wraps=CigarIndex) as coordinate_calls, \
                    patch('utils.features.bam.decode_mods', wraps=decode_mods) as decode_calls:
                run = extract_bam_features(sites, 'WT', bam, fa, root / 'features', flank=2)
            self.assertEqual(coordinate_calls.call_count, 2)
            self.assertEqual(decode_calls.call_count, 2)
            self.assertEqual(run['bam_passes'], 1)
            self.assertEqual(run['counts']['coordinate_index_builds'], 2)
            self.assertEqual(run['counts']['modification_decodes'], 2)
            self.assertEqual(run['counts']['bam_records_scanned'], 2)
            self.assertEqual(run['extraction_strategy'], 'one_bam_pass_all_targets')
            result = pq.read_table(str(root / 'features/sites')).to_pylist()
            self.assertEqual([row['retained_depth'] for row in result], [1, 1, 2, 2, 0, 0])
            self.assertEqual([row['read_offset'] for row in result], [0, 1, 2, 4, 6, 6])
            reads = pq.read_table(str(root / 'features/reads')).to_pylist()
            self.assertEqual([row['row_id'] for row in reads], ['12:m6A', '12:inosine',
                '100017:m6A', '100017:m6A', '100017:inosine', '100017:inosine'])


if __name__ == '__main__':
    unittest.main()
