"""Run the public command through real CPU/GPU-worker/CPU/BAM stages on tiny data."""
from array import array
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from utils.references.catalog import sha256

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES_AVAILABLE = all(importlib.util.find_spec(name) for name in ('pysam', 'pyarrow', 'tensorflow'))


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, 'BAM, Parquet and TensorFlow dependencies required')
class PublicPredictionEndToEndTest(unittest.TestCase):
    def test_complete_prediction_with_isolated_inference_and_recalibrated_bam(self):
        self.complete_prediction('hg38')

    def test_mouse_prediction_with_cpu_and_recalibrated_bam(self):
        self.complete_prediction('mm10')

    def test_other_genome_prediction_without_db_or_short_read(self):
        self.complete_prediction('other')

    def complete_prediction(self, genome):
        import numpy as np
        import pysam
        import tensorflow as tf
        from test_inference import FinalBundleTest
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fasta = directory / 'reference.fa'
            fasta.write_text('>chr1\n' + 'A' * 50 + 'T' * 50 + '\n')
            pysam.faidx(str(fasta))
            bam = directory / 'reads.bam'
            with pysam.AlignmentFile(str(bam), 'wb', header=dict(
                    HD=dict(SO='coordinate'), SQ=[dict(SN='chr1', LN=100)])) as handle:
                for strand, start in [('+', 1), ('-', 51)]:
                    for index in range(10):
                        read = pysam.AlignedSegment(handle.header)
                        read.query_name = ('plus' if strand == '+' else 'minus') + str(index)
                        read.query_sequence = ('A' if strand == '+' else 'T') * 19
                        read.query_qualities = array('B', [30] * 19)
                        read.flag = 0 if strand == '+' else 16
                        read.reference_id, read.reference_start = 0, start
                        read.cigarstring, read.mapping_quality = '19M', 60
                        read.set_tag('MM', 'A+a?,9;A+17596?,9;')
                        read.set_tag('ML', array('B', [220 if index < 3 else 10, 44]))
                        handle.write(read)
            pysam.index(str(bam))
            input_sha = sha256(bam)
            lines = []
            for position, strand in [(10, '+'), (60, '-')]:
                for code, modified, other in [('a', 3, 2), ('17596', 2, 3)]:
                    lines.append(f'chr1 {position} {position+1} {code} 10 {strand} {position} '
                                 f'{position+1} 0,0,0 10 {modified*10} {modified} 5 {other} 0 0 0 0\n')
            (directory / 'calls.bed').write_text(''.join(lines))
            (directory / 'pileup.log').write_text('setting threshold 0.7 (qual: 179) for base A, percentile 0.1\n')
            gtf = directory / 'genes.gtf'
            gtf.write_text('##genome-build GRCh38\n'
                          'chr1\ttest\texon\t1\t50\t.\t+\t.\tgene_id "g_plus"; transcript_id "t_plus"; gene_name "PLUS";\n'
                          'chr1\ttest\tCDS\t1\t50\t.\t+\t0\tgene_id "g_plus"; transcript_id "t_plus"; gene_name "PLUS";\n'
                          'chr1\ttest\texon\t51\t100\t.\t-\t.\tgene_id "g_minus"; transcript_id "t_minus"; gene_name "MINUS";\n'
                          'chr1\ttest\tUTR\t51\t100\t.\t-\t.\tgene_id "g_minus"; transcript_id "t_minus"; gene_name "MINUS";\n')
            assembly = 'mm10' if genome == 'mm10' else 'GRCh38'
            (directory / 'catalog.json').write_text(json.dumps(dict(assembly=assembly, sources=[])))
            model_directory = directory / 'models/adenine'
            run = FinalBundleTest().save_bundle(model_directory, dict(gtf=dict(sha256=sha256(gtf))))
            # A constant score tests pipeline decisions deterministically, not NN performance.
            model_path = model_directory / 'model.keras'
            model = tf.keras.models.load_model(model_path, compile=False)
            output_layer = model.get_layer('class_scores')
            kernel, bias = output_layer.get_weights()
            output_layer.set_weights([np.zeros_like(kernel), np.log(np.array([.2, .1, .7], dtype=bias.dtype))])
            model.save(model_path)
            run.update(threshold=.75, model_sha256=sha256(model_path))
            (model_directory / 'run.json').write_text(json.dumps(run))
            tf.keras.backend.clear_session()
            config = yaml.safe_load((ROOT / 'prediction/config.yaml').read_text())
            config.update(output_dir='output', targets=['m6A', 'inosine'], model_root='models')
            config['nn']['task'] = 'adenine_three_class'
            config['sample'].update(sample_id='synthetic', cell='HEK293T', reference_cell='HEK293T',
                                    bam='reads.bam', bedmethyl='calls.bed', pileup_log='pileup.log')
            config['references'].update(fasta='reference.fa', gtf='genes.gtf', catalog='catalog.json')
            config_path = directory / 'pipeline.yaml'
            config_path.write_text(yaml.safe_dump(config))
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', TF_CPP_MIN_LOG_LEVEL='3',
                               TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2',
                               MPLCONFIGDIR=str(directory / 'matplotlib'))
            result = subprocess.run([sys.executable, '-B', '-m', 'rnamodscan', 'predict', '--config',
                                     str(config_path), '--genome', genome, '--recalibrate', '--device', 'cpu', '--cpu-threads', '2'], cwd=ROOT, env=environment,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + '\n' + result.stderr)
            output = directory / 'output'
            runtime = json.loads((output / 'scores/adenine/run.json').read_text())['runtime']
            self.assertEqual((runtime['requested_device'], runtime['device']), ('cpu', 'cpu'))
            self.assertEqual(runtime['gpu_devices'], [])
            self.assertEqual(runtime['cpu_threads'], 2)
            for strand in ('plus', 'minus'):
                prepared = output / 'nn_inputs/adenine/chrom=chr1' / ('strand=' + strand) / 'inputs'
                self.assertTrue(list(prepared.glob('*.parquet')))
                self.assertTrue((output / 'scores/adenine/chrom=chr1' / ('strand=' + strand) / 'scores.parquet').is_file())
            with (output / 'predictions/m6A/all_predictions.tsv').open() as handle:
                rows = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(len(rows), 2)
            by_strand = {row['strand']: row for row in rows}
            self.assertEqual(by_strand['+']['gene_names'], 'PLUS')
            self.assertEqual(by_strand['+']['regions'], 'CDS')
            self.assertEqual(by_strand['-']['gene_names'], 'MINUS')
            self.assertEqual(by_strand['-']['regions'], 'UTR')
            for row in rows:
                self.assertAlmostEqual(float(row['keep_score']), .2, places=6)
                self.assertEqual(row['nn_keep'], '0')
                self.assertEqual(row['final_keep'], '0')
                self.assertEqual(float(row['af']), .3)
                self.assertEqual(row['total_reads'], '10')
                self.assertEqual(row['modified_reads'], '3')
                self.assertEqual(row['call_count_m6A'], '3')
                self.assertEqual(row['call_count_inosine'], '2')
            with (output / 'predictions/inosine/all_predictions.tsv').open() as handle:
                self.assertEqual(list(csv.DictReader(handle, delimiter='\t')), [])
            self.assertEqual((output / 'predictions/bedmethyl/filtered.bed').read_text(), '')
            with pysam.AlignmentFile(str(output / 'bam/filtered.bam'), 'rb') as handle:
                reads = list(handle.fetch(until_eof=True))
            self.assertEqual(len(reads), 20)
            for read in reads:
                self.assertEqual(list(read.get_tag('ML')), [0, 44])  # Below-threshold inosine was never a candidate.
                self.assertEqual(read.get_tag('MM'), 'A+a?,9;A+17596?,9;')
                self.assertEqual(list(read.query_qualities), [30] * 19)
            self.assertTrue((output / 'bam/filtered.bam.bai').is_file())
            self.assertEqual(sha256(bam), input_sha)
            self.assertFalse(list(output.rglob('INCOMPLETE.txt')))
            self.assertFalse(list(output.rglob('FAILED.txt')))


if __name__ == '__main__':
    unittest.main()
