"""Synthetic splice junctions: query positions, MM/ML, missing states and strands."""
import csv
import tempfile
import unittest
from array import array
from pathlib import Path

try:
    import pysam
except ImportError:
    pysam = None

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from utils.features.bam import (decode_mods, encode_read, extract_bam_features, read_map,
                                     read_splice_junctions, splice_distances)


@unittest.skipIf(pysam is None, 'pysam not installed')
class SpliceFeaturesTest(unittest.TestCase):
    def make_read(self, reverse=False, reference_id=0):
        header = pysam.AlignmentHeader.from_references(['chrP', 'chrR'], [500, 500])
        read = pysam.AlignedSegment(header)
        read.query_name = 'reverse' if reverse else 'forward'
        read.query_sequence = ('T' if reverse else 'A') * 17
        read.flag = 16 if reverse else 0
        read.reference_id, read.reference_start = reference_id, 100
        read.mapping_quality = 60
        read.cigarstring = '2S4M100N4M1I2M2D3M1S'
        read.query_qualities = array('B', range(10, 27))
        # Genomic position 204 maps to stored query position 6. On reverse
        # alignments the original read position is 17-1-6 = 10.
        read.set_tag('MM', f'A+a?,{10 if reverse else 6};')
        read.set_tag('ML', array('B', [231]))
        read.set_tag('MN', 17)
        return read

    def test_splice_and_deletion_advance_reference_without_consuming_query(self):
        for reverse in (False, True):
            read = self.make_read(reverse)
            mapping, insertion = read_map(read, 100, 215)
            self.assertEqual(mapping[100], (2, 'base'))  # Leading soft clip consumes query.
            self.assertEqual(mapping[103], (5, 'base'))
            self.assertEqual(mapping[104], (None, 'refskip'))
            self.assertEqual(mapping[203], (None, 'refskip'))
            self.assertEqual(mapping[204], (6, 'base'))  # 100N does not shift query/MM positions.
            self.assertEqual(mapping[208], (11, 'base'))  # Insertion consumes one query base.
            self.assertEqual(insertion[207], 1)
            self.assertEqual(mapping[210], (None, 'deletion'))
            self.assertEqual(mapping[211], (None, 'deletion'))
            self.assertEqual(mapping[212], (13, 'base'))
            self.assertEqual(mapping[214], (15, 'base'))
            self.assertNotIn(215, mapping)  # Trailing soft clip has no reference base.

    def test_mm_ml_after_splice_and_rna_orientation(self):
        for reverse in (False, True):
            read = self.make_read(reverse)
            mapping, insertion = read_map(read, 202, 207)
            mods, modes = decode_mods(read)
            self.assertEqual(mods[6, 'm6A'], 231)
            site = dict(chrom='chrP', start=204, strand='-' if reverse else '+')
            result = encode_read(read, site, 2, mapping, insertion, mods, modes, 'AAAAA')
            expected = (['base', 'base', 'base', 'refskip', 'refskip'] if reverse else
                        ['refskip', 'refskip', 'base', 'base', 'base'])
            self.assertEqual(result['alignment_state'], expected)
            self.assertEqual(result['base'][2], 'A')
            self.assertEqual(result['base_quality'][2], 16)
            self.assertEqual(result['ml_byte'][2][0], 231)
            self.assertTrue(result['modification_observed'][2][0])
            self.assertEqual((result['splice_distance_before'], result['splice_distance_after']),
                             (None, 0) if reverse else (0, None))
            for i, state in enumerate(expected):
                if state == 'refskip':
                    self.assertIsNone(result['base'][i])
                    self.assertIsNone(result['base_quality'][i])
                    self.assertIsNone(result['mismatch'][i])
                    self.assertTrue(all(value is None for value in result['ml_byte'][i]))
                    self.assertFalse(any(result['modification_observed'][i]))

    def test_junction_distances_use_full_read_and_exclude_introns_and_deletions(self):
        for reverse in (False, True):
            read = self.make_read(reverse)
            read.query_sequence = ('T' if reverse else 'A') * 39
            read.query_qualities = array('B', [30] * 39)
            read.cigarstring = '3S10M2I100N5M3D5M70N12M2S'
            junctions = read_splice_junctions(read)
            self.assertEqual(junctions, [15, 25])
            expected = {109: (None, 2), 210: (0, 9), 218: (5, 4), 293: (0, None), 299: (6, None)}
            for start, distances in expected.items():
                # The junction for 299 lies well outside this +/-2 genome window.
                mapping, insertion = read_map(read, start - 2, start + 3)
                feature = encode_read(read, dict(start=start, strand='-' if reverse else '+'),
                                      2, mapping, insertion, {}, {}, 'AAAAA', junctions)
                if reverse:
                    distances = tuple(reversed(distances))
                self.assertEqual((feature['splice_distance_before'], feature['splice_distance_after']), distances)
            self.assertEqual(splice_distances(junctions, None, reverse), (None, None))

    def test_unspliced_read_does_not_claim_a_junction_at_read_ends(self):
        read = self.make_read()
        read.cigarstring = '2S4M1I2M2D7M1S'
        junctions = read_splice_junctions(read)
        self.assertEqual(junctions, [])
        self.assertEqual(splice_distances(junctions, 6), (None, None))

    def test_reverse_insertion_at_low_coordinate_window_boundary_is_retained(self):
        read = self.make_read(reverse=True)
        # An insertion between 207/208 follows RNA-oriented base 208. This is
        # the low-coordinate edge of a [208,213) feature window.
        mapping, insertion = read_map(read, 208, 213)
        mods, modes = decode_mods(read)
        result = encode_read(read, dict(start=210, strand='-'), 2,
                             mapping, insertion, mods, modes, 'AAAAA')
        self.assertEqual(result['insertion_length_after'][-1], 1)

    @unittest.skipIf(pq is None, 'pyarrow not installed')
    def test_bam_to_parquet_junctions_and_skipped_candidate_centers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / 'genome.fa'
            fasta.write_text('>chrP\n' + 'A' * 500 + '\n>chrR\n' + 'T' * 500 + '\n')
            pysam.faidx(str(fasta))
            bam = root / 'spliced.bam'
            forward = self.make_read()
            reverse = self.make_read(True, 1)
            with pysam.AlignmentFile(str(bam), 'wb', header=forward.header) as handle:
                handle.write(forward)
                handle.write(reverse)
            pysam.index(str(bam))
            sites = root / 'sites.tsv'
            fields = ['row_id', 'sample_id', 'chrom', 'start', 'end', 'strand', 'modification']
            with sites.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fields, delimiter='\t')
                writer.writeheader()
                for chrom, strand in [('chrP', '+'), ('chrR', '-')]:
                    for start in (150, 204, 210):
                        writer.writerow(dict(row_id=f'{chrom}:{start}', sample_id='WT', chrom=chrom,
                                             start=start, end=start+1, strand=strand, modification='m6A'))
            report = extract_bam_features(sites, 'WT', bam, fasta, root / 'features', flank=2)
            self.assertEqual(report['schema'], 'read_features_parquet_v3')
            metadata = {r['row_id']: r for r in pq.read_table(str(root / 'features/sites')).to_pylist()}
            reads = {r['row_id']: r for r in pq.read_table(str(root / 'features/reads')).to_pylist()}
            for chrom in ('chrP', 'chrR'):
                self.assertEqual(metadata[f'{chrom}:204']['retained_depth'], 1)
                self.assertEqual(reads[f'{chrom}:204']['ml_byte'][2][0], 231)
                self.assertEqual(metadata[f'{chrom}:150']['retained_depth'], 0)
                self.assertEqual(metadata[f'{chrom}:150']['central_refskips'], 1)
                self.assertEqual(metadata[f'{chrom}:210']['retained_depth'], 0)
                self.assertEqual(metadata[f'{chrom}:210']['central_deletions'], 1)
                self.assertNotIn(f'{chrom}:150', reads)
                self.assertNotIn(f'{chrom}:210', reads)
            self.assertEqual(reads['chrP:204']['splice_distance_before'], 0)
            self.assertIsNone(reads['chrP:204']['splice_distance_after'])
            self.assertIsNone(reads['chrR:204']['splice_distance_before'])
            self.assertEqual(reads['chrR:204']['splice_distance_after'], 0)


class SpliceInputsTest(unittest.TestCase):
    def raw_row(self):
        return dict(base=['A'], mm_skip_modes=['?'] * 8, ml_byte=[[None] * 8],
                    modification_observed=[[False] * 8], modification_applicable=[[False] * 8],
                    base_quality=[30], mismatch=[0], alignment_state=['base'],
                    insertion_length_after=[0], mapq=60,
                    splice_distance_before=0, splice_distance_after=None)

    def test_adjacent_junction_and_unobserved_side_use_distinct_values(self):
        from utils.nn.inputs import CHANNEL_NAMES, encode_read_row
        values = dict(zip(CHANNEL_NAMES, encode_read_row(self.raw_row())[0]))
        self.assertEqual(values['splice_before_log_distance'], 0)
        self.assertEqual(values['splice_after_log_distance'], 1)
        self.assertNotIn('splice_before_observed', values)
        self.assertNotIn('splice_after_observed', values)
        self.assertEqual(len(values), 18)

    def test_distance_transform_caps_large_distances_and_rejects_old_rows(self):
        from utils.nn.inputs import CHANNEL_NAMES, encode_read_row
        row = self.raw_row()
        row['splice_distance_before'] = 1001
        values = dict(zip(CHANNEL_NAMES, encode_read_row(row)[0]))
        self.assertEqual(values['splice_before_log_distance'], 1)
        row['splice_distance_before'] = -1
        with self.assertRaisesRegex(ValueError, 'Splice distance'):
            encode_read_row(row)
        del row['splice_distance_before']
        with self.assertRaises(KeyError):
            encode_read_row(row)


if __name__ == '__main__':
    unittest.main()
