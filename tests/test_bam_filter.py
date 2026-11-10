from array import array
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prediction.bam_recalibration import load_rejected_sites, modification_entries, write_filtered_bam, zero_rejected_modifications


@unittest.skipUnless(importlib.util.find_spec('pysam'), 'pysam not installed')
class BamFilterTest(unittest.TestCase):
    def read(self, name='read', reverse=False):
        import pysam
        sequence = 'AAAATTACCCGG'
        read = pysam.AlignedSegment()
        read.query_name = name
        read.query_sequence = sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1] if reverse else sequence
        read.flag = 16 if reverse else 0
        read.reference_id = 0
        read.reference_start = 100
        read.mapping_quality = 60
        read.cigarstring = '2S4M2I100N4M'
        read.query_qualities = array('B', [30] * 12)
        read.set_tag('MM', 'A+a?,0,0,0,0,0;C+mh?,0,0,0;T+17802?,0,0;A+17596?,0,0,0,0,0;C-m?,1;G+19229?,0,0;')
        scores = array('B', range(10, 31))
        scores[2] = 0
        read.set_tag('ML', scores)
        read.set_tag('MN', 12)
        read.set_tag('ZZ', 'must stay unchanged')
        return read

    def predictions(self, root, rows):
        directory = root / 'predictions'
        directory.mkdir()
        fields = ['sample_id', 'chrom', 'start', 'end', 'strand', 'modification',
                  'prediction_status', 'keep_score', 'nn_keep', 'rescued', 'final_keep']
        with (directory / 'all_predictions.tsv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fields, delimiter='\t')
            writer.writeheader()
            for position, strand, target, policy in rows:
                row = dict(sample_id='sample', chrom='chr1', start=position, end=position + 1,
                           strand=strand, modification=target, prediction_status='predicted',
                           keep_score=.1, nn_keep=0, rescued=0, final_keep=0)
                if policy == 'rescued':
                    row.update(rescued=1, final_keep=1)
                elif policy == 'kept':
                    row.update(keep_score=.9, nn_keep=1, final_keep=1)
                elif policy == 'unknown':
                    row.update(prediction_status='no_reads', keep_score='', nn_keep='')
                elif policy == 'other_sample':
                    row.update(sample_id='other')
                writer.writerow(row)
        (directory / 'run.json').write_text(json.dumps(dict(status='complete')))
        return directory

    def test_raw_ml_order_multicode_numeric_strand_and_unaligned_bases(self):
        read = self.read()
        original_ml, original_mm, original_qual = list(read.get_tag('ML')), read.get_tag('MM'), read.query_qualities
        original_sequence = read.query_sequence
        keys = [(100, 'm6A'), (101, 'm6A'), (204, 'm5C'), (102, 'pseU'), (100, 'inosine')]
        rejected = {('chr1', p, '+', target): dict(observed=0, changed=0, already_zero=0) for p, target in keys}
        from utils.io.cigar import CigarIndex
        with patch('prediction.bam_recalibration.CigarIndex', wraps=CigarIndex) as mapping:
            observed, changed = zero_rejected_modifications(read, rejected, 'chr1')
            self.assertEqual(mapping.call_count, 1)
        expected = list(original_ml)
        for offset in (2, 3, 7, 11, 15):
            expected[offset] = 0
        self.assertEqual(list(read.get_tag('ML')), expected)
        self.assertEqual((observed, changed), (5, 4))
        self.assertEqual(rejected['chr1', 100, '+', 'm6A'], dict(observed=1, changed=0, already_zero=1))
        self.assertEqual(read.get_tag('MM'), original_mm)
        self.assertEqual(read.query_sequence, original_sequence)
        self.assertEqual(read.query_qualities, original_qual)
        self.assertEqual(read.get_tag('MN'), 12)
        self.assertEqual(read.get_tag('ZZ'), 'must stay unchanged')
        # Unknown h, C-m on the opposite MM strand and 2OmeG keep their raw bytes.
        self.assertEqual([expected[i] for i in (6, 8, 10, 18, 19, 20)],
                         [original_ml[i] for i in (6, 8, 10, 18, 19, 20)])

    def test_reverse_alignment_uses_original_mm_query_positions(self):
        read = self.read(reverse=True)
        original_ml = list(read.get_tag('ML'))
        rejected = {('chr1', p, '-', target): dict(observed=0, changed=0, already_zero=0)
                    for p, target in [(205, 'm6A'), (101, 'm5C'), (205, 'inosine')]}
        observed, changed = zero_rejected_modifications(read, rejected, 'chr1')
        expected = list(original_ml)
        for offset in (2, 7, 15):
            expected[offset] = 0
        self.assertEqual((observed, changed), (3, 2))
        self.assertEqual(list(read.get_tag('ML')), expected)
        opposite = {('chr1', 205, '+', 'm6A'): dict(observed=0, changed=0, already_zero=0)}
        self.assertEqual(zero_rejected_modifications(read, opposite, 'chr1'), (0, 0))

    def test_prediction_selection_preserves_rescue_missing_and_other_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = self.predictions(root, [(100, '+', 'm6A', 'rejected'),
                (101, '+', 'm6A', 'rescued'), (102, '+', 'pseU', 'unknown'),
                (204, '+', 'm5C', 'kept'), (205, '+', 'm5C', 'other_sample')])
            rejected, _, counts = load_rejected_sites([directory], 'sample')
            self.assertEqual(set(rejected), {('chr1', 100, '+', 'm6A')})
            self.assertEqual(counts['rejected_sites'], 1)
            self.assertEqual(counts['unpredicted_sites_preserved'], 1)
            self.assertEqual(counts['retained_sites_preserved'], 2)
            self.assertEqual(counts['other_sample_rows'], 1)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_rejected_sites([directory, directory], 'sample')

    def test_tags_aliases_and_invalid_ml_layout(self):
        read = self.read()
        mm, ml = read.get_tag('MM'), read.get_tag('ML')
        read.set_tag('MM', None)
        read.set_tag('ML', None)
        read.set_tag('Mm', mm)
        read.set_tag('Ml', ml)
        rejected = {('chr1', 101, '+', 'm6A'): dict(observed=0, changed=0, already_zero=0)}
        self.assertEqual(zero_rejected_modifications(read, rejected, 'chr1'), (1, 1))
        self.assertFalse(read.has_tag('ML'))
        self.assertEqual(read.get_tag('Ml')[3], 0)
        with self.assertRaisesRegex(ValueError, 'entry counts differ'):
            modification_entries(read, 'Mm', 1)
        read.set_tag('MN', 11)
        with self.assertRaisesRegex(ValueError, 'MN/SEQ'):
            modification_entries(read, 'Mm', len(ml))
        read.set_tag('MN', 12)
        read.set_tag('Mm', 'A+a,99;')
        with self.assertRaisesRegex(ValueError, 'outside SEQ'):
            modification_entries(read, 'Mm', 1)

    def test_complete_bam_export_keeps_records_and_source_and_writes_index(self):
        import pysam
        from utils.references.catalog import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bam = root / 'input.bam'
            reads = [self.read('forward'), self.read('reverse', reverse=True), self.read('secondary')]
            reads[2].flag |= 256
            unaligned = pysam.AlignedSegment()
            unaligned.query_name, unaligned.query_sequence, unaligned.flag = 'unmapped', 'AA', 4
            reads.append(unaligned)
            with pysam.AlignmentFile(str(bam), 'wb', header=dict(HD=dict(VN='1.6', SO='coordinate'),
                                                SQ=[dict(SN='chr1', LN=1000)])) as handle:
                for read in reads:
                    handle.write(read)
            source_hash = sha256(bam)
            directory = self.predictions(root, [(101, '+', 'm6A', 'rejected'),
                (101, '-', 'm5C', 'rejected'), (102, '+', 'pseU', 'unknown'), (204, '+', 'm5C', 'rescued')])
            result = write_filtered_bam(bam, [directory], 'sample', root / 'out', threads=1)
            self.assertEqual(result['counts']['records_written'], 4)
            self.assertEqual(result['counts']['records_changed'], 3)
            self.assertEqual(result['counts']['ml_entries_changed_to_zero'], 3)
            self.assertEqual(sha256(bam), source_hash)
            with pysam.AlignmentFile(str(root / 'out/filtered.bam'), 'rb', require_index=True) as handle:
                output = list(handle.fetch(until_eof=True))
                self.assertEqual(handle.header.to_dict()['HD']['SO'], 'coordinate')
            self.assertEqual([r.query_name for r in output], [r.query_name for r in reads])
            for before, after in zip(reads, output):
                self.assertEqual(before.query_sequence, after.query_sequence)
                self.assertEqual(before.query_qualities, after.query_qualities)
                self.assertEqual(before.flag, after.flag)
                self.assertEqual(before.cigarstring, after.cigarstring)
                if before.has_tag('MM'):
                    self.assertEqual(before.get_tag('MM'), after.get_tag('MM'))
                    self.assertEqual(before.get_tag('MN'), after.get_tag('MN'))
            self.assertEqual(output[0].get_tag('ML')[3], 0)
            self.assertEqual(output[0].get_tag('ML')[7], reads[0].get_tag('ML')[7])  # Rescued.
            self.assertEqual(output[0].get_tag('ML')[11], reads[0].get_tag('ML')[11])  # No score.
            self.assertEqual(output[1].get_tag('ML')[7], 0)
            self.assertTrue((root / 'out/filtered.bam.bai').is_file())
            self.assertTrue((root / 'out/site_audit.tsv').is_file())
            self.assertFalse((root / 'out/INCOMPLETE.txt').exists())
            with self.assertRaises(FileExistsError):
                write_filtered_bam(bam, [directory], 'sample', root / 'out', threads=1)

    def test_bad_mm_prevents_completed_bam(self):
        import pysam
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bam = root / 'input.bam'
            read = self.read()
            read.set_tag('MM', 'A+a,99;')
            with pysam.AlignmentFile(str(bam), 'wb', header=dict(HD=dict(SO='coordinate'),
                                                SQ=[dict(SN='chr1', LN=1000)])) as handle:
                handle.write(read)
            directory = self.predictions(root, [(101, '+', 'm6A', 'rejected')])
            with self.assertRaisesRegex(ValueError, 'outside SEQ'):
                write_filtered_bam(bam, [directory], 'sample', root / 'out', threads=1)
            self.assertTrue((root / 'out/FAILED.txt').exists())
            self.assertTrue((root / 'out/INCOMPLETE.txt').exists())
            self.assertFalse((root / 'out/run.json').exists())

    def test_empty_candidate_tables_preserve_the_entire_bam(self):
        import pysam
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bam = root / 'input.bam'
            read = self.read()
            with pysam.AlignmentFile(str(bam), 'wb', header=dict(HD=dict(SO='coordinate'),
                                                SQ=[dict(SN='chr1', LN=1000)])) as handle:
                handle.write(read)
            directory = self.predictions(root, [])
            report = write_filtered_bam(bam, [directory], 'sample', root / 'out', threads=1)
            self.assertEqual(report['counts']['rejected_sites'], 0)
            self.assertEqual(report['counts']['records_changed'], 0)
            with pysam.AlignmentFile(str(root / 'out/filtered.bam'), 'rb', require_index=True) as handle:
                output = next(handle.fetch(until_eof=True))
            self.assertEqual(output.get_tag('ML'), read.get_tag('ML'))
            self.assertEqual(output.get_tag('MM'), read.get_tag('MM'))


if __name__ == '__main__':
    unittest.main()
