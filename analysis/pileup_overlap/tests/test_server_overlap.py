import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import test_call_summary as support
from rnamodscan.analysis.call_summary import summarize_calls
from utils.pileup.provenance import read_pileup_log
from utils.references.genome import IndexedFasta


class ServerOverlapTests(unittest.TestCase):
    def make_log(self, root):
        p = root / 'pileup.log'
        p.write_text('[DEBUG] command line: modkit pileup input.bam output.bed --ref hg38.fa\n'
                     '[INFO] setting threshold 0.5917969 (qual: 151) for base A, percentile 10, 100 total probabilites\n'
                     '[INFO] setting threshold 0.6035156 (qual: 154) for base C, percentile 10, 100 total probabilites\n'
                     '[INFO] setting threshold 0.7324219 (qual: 187) for base G, percentile 10, 100 total probabilites\n'
                     '[INFO] setting threshold 0.6191406 (qual: 158) for base T, percentile 10, 100 total probabilites\n'
                     '[INFO] Done, processed 5 rows.\n')
        return p

    def test_actual_threshold_metadata_and_four_part_partition(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fa, manifest, samples, thresholds = support.CallSummaryTests().fixture(root)
            Path(samples[0]['bedmethyl']).write_text(
                support.pileup(0) + support.pileup(10) + support.pileup(20)
                + support.pileup(30, strand='-') + support.pileup(40, modified=0))
            (root / 'ref.bed').write_text('chr1\t10\t11\ta\t0\t+\nchr1\t30\t31\tb\t0\t-\n')
            (root / 'db.bed').write_text('chr1\t10\t11\ta\t0\t+\nchr1\t20\t21\tb\t0\t+\n')
            config = json.loads(manifest.read_text())
            next(s for s in config['sources'] if s['id'] == 'DB')['path'] = 'db.bed'
            manifest.write_text(json.dumps(config))
            for sample in samples:
                sample['pileup_log'] = self.make_log(root)
            out = root / 'out'
            report = summarize_calls(samples, thresholds[:1], fa, manifest, out,
                                     write_called_sites=False, progress_every=2)
            self.assertIsNone(report['upstream_min_ml'])
            self.assertEqual(report['upstream_by_sample']['WT']['filter_mode'], 'sampled_per_base')
            self.assertEqual(report['upstream_by_sample']['WT']['base_thresholds']['A']['logged_quality'], 151)
            self.assertTrue(all(a['matches_log_row_count'] for a in report['input_audits']))
            self.assertEqual(report['status'], 'complete')
            self.assertFalse((out / 'called_sites.tsv').exists())
            self.assertFalse((out / 'called_sites.tsv.gz').exists())
            self.assertFalse((out / 'INCOMPLETE.txt').exists())
            parts = support.read_table(out / 'reference_overlap_summary.tsv')
            row = next(r for r in parts if r['sample_id'] == 'WT' and r['modification'] == 'm6A' and r['motif_group'] == 'all')
            self.assertEqual([row[k] for k in ('call_sites', 'both', 'short_read_only', 'database_only', 'neither')],
                             ['4', '1', '1', '1', '1'])
            missing = next(r for r in parts if r['modification'] == 'm5C')
            self.assertEqual(missing['neither'], '')
            self.assertEqual(missing['reference_status'], 'incomplete_reference')

    def test_conflicting_fixed_ml_is_refused_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fa, manifest, samples, thresholds = support.CallSummaryTests().fixture(root)
            samples[0]['pileup_log'] = self.make_log(root)
            with self.assertRaisesRegex(ValueError, 'set upstream_min_ml=None'):
                summarize_calls(samples, thresholds, fa, manifest, root / 'out', 125)
            self.assertFalse((root / 'out').exists())
            self.assertEqual(read_pileup_log(samples[0]['pileup_log'])['completed_rows'], 5)

    def test_optional_gzip_site_table(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fa, manifest, samples, thresholds = support.CallSummaryTests().fixture(root)
            out = root / 'out'
            summarize_calls(samples[:1], thresholds[:1], fa, manifest, out, compress_called_sites=True)
            with gzip.open(out / 'called_sites.tsv.gz', 'rt') as f:
                rows = list(csv.DictReader(f, delimiter='\t'))
            self.assertEqual(len(rows), 3)
            self.assertEqual(next(r for r in rows if r['start'] == '30')['motif5'], 'AGACT')

    def test_cached_fasta_boundaries_and_contig_switch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fa = root / 'ref.fa'
            sequences = {'chr1': 'ACGTAGACTTACGTACGA', 'chr2': 'TTAGTCTACGATGCAAAA'}
            content = b''
            fai = []
            for chrom, seq in sequences.items():
                header = ('>' + chrom + '\n').encode()
                offset = len(content) + len(header)
                content += header + ''.join(seq[i:i+6] + '\n' for i in range(0, len(seq), 6)).encode()
                fai.append(f'{chrom}\t{len(seq)}\t{offset}\t6\t7\n')
            fa.write_bytes(content)
            Path(str(fa) + '.fai').write_text(''.join(fai))
            fasta = IndexedFasta(fa)
            fasta.chunk_size = 8
            try:
                for chrom in ['chr1', 'chr2', 'chr1']:
                    seq = sequences[chrom]
                    for pos in range(len(seq)):
                        expected = ('NN' + seq + 'NN')[pos:pos+5]
                        self.assertEqual(fasta.motif(chrom, pos, '+'), expected)
                        self.assertEqual(fasta.motif(chrom, pos, '-'), expected.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1])
            finally:
                fasta.close()
