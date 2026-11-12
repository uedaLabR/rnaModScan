import contextlib
import io
import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from rnamodscan.cli import main
from utils.config import build_sample_pairs, describe_stage, load_config

ROOT = Path(__file__).resolve().parents[1]


class WorkflowCliTest(unittest.TestCase):
    def prediction_config(self, directory, **changes):
        values = yaml.safe_load((ROOT / 'prediction/config.yaml').read_text())
        values.update(output_dir='runs/analysis', **changes)
        path = directory / 'prediction.yaml'
        path.write_text(yaml.safe_dump(values))
        return path

    def test_relative_paths_and_samples_are_normalized_without_data_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = self.prediction_config(directory)
            config = load_config(path, 'predict')
            self.assertEqual(config['output_dir'], directory / 'runs/analysis')
            self.assertIsNone(config['sample']['bam'])
            self.assertEqual(config['references']['catalog'], (ROOT / 'data/references/hg38/catalog.json').resolve())
            report = describe_stage(config, 'predict', 'features')
            self.assertIn('analysis_sample.bam', report['unset_inputs'])
            self.assertEqual(report['output'], directory / 'runs/analysis/features')
            self.assertFalse(report['output'].exists())

    def test_fetch_span_limit_is_configurable_and_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self.prediction_config(directory, features=dict(max_fetch_span=5000000))
            self.assertEqual(load_config(path, 'predict')['features']['max_fetch_span'], 5000000)
            for invalid in (0, -1, True, None, 1.5):
                path = self.prediction_config(directory, features=dict(max_fetch_span=invalid))
                with self.assertRaisesRegex(ValueError, 'max_fetch_span'):
                    load_config(path, 'predict')

    def test_training_pair_mapping_does_not_infer_cell_identity(self):
        config = load_config(ROOT / 'training/config.yaml', 'train')
        pairs = build_sample_pairs(config['samples'])
        self.assertEqual(pairs[0]['cell'], 'HEK293T')
        self.assertEqual(pairs[0]['reference_cell'], 'HEK293T')
        self.assertEqual(pairs[0]['wt_sample'], 'HEK293T_WT')
        self.assertEqual(pairs[0]['ivt_bedmethyl'], config['samples'][1]['bedmethyl'])
        missing = [dict(sample, bedmethyl=None) for sample in config['samples']]
        self.assertIsNone(build_sample_pairs(missing)[0]['ivt_bedmethyl'])
        changed = dict(config['samples'][1], reference_cell='HEK293')
        with self.assertRaisesRegex(ValueError, 'reference_cell disagree'):
            build_sample_pairs([config['samples'][0], changed])

    def test_unknown_keys_and_wrong_mode_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = self.prediction_config(directory, output_directory='typo')
            with self.assertRaisesRegex(ValueError, 'Unknown top-level'):
                load_config(path, 'predict')
            path = self.prediction_config(directory, mode='train')
            with self.assertRaisesRegex(ValueError, 'mode does not match'):
                load_config(path, 'predict')

    def test_dry_run_and_cli_do_not_import_tensorflow_or_pysam(self):
        for command, stage in (('quality', 'quality'), ('predict', 'all')):
            with self.subTest(command=command):
                arguments = [command, '--config', 'prediction/config.yaml', '--dry-run']
                code = ('import sys; from rnamodscan.cli import main; '
                        f'main({arguments!r}); '
                        'assert "tensorflow" not in sys.modules; assert "pysam" not in sys.modules')
                result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertTrue(report['dry_run'])
                self.assertEqual(report['stage'], stage)
                self.assertIn('analysis_sample.bam', report['unset_inputs'])

    def test_dispatch_passes_the_loaded_config_and_stage(self):
        with patch('prediction.pipeline.run_prediction_stage', return_value={'status': 'not executed'}) as run:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                result = main(['predict', '--stage', 'bam', '--config', str(ROOT / 'prediction/config.yaml')])
            self.assertEqual(result, 0)
            config, stage = run.call_args.args
            self.assertEqual(stage, 'bam')
            self.assertEqual(config['mode'], 'predict')
            self.assertEqual(json.loads(output.getvalue())['status'], 'not executed')

    def test_missing_bam_has_a_stage_specific_error(self):
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as exit_result:
                main(['predict', '--stage', 'bam', '--config', str(ROOT / 'prediction/config.yaml')])
        self.assertEqual(exit_result.exception.code, 2)
        self.assertIn('Set sample.bam', errors.getvalue())

    def test_invalid_stage_is_rejected_by_cli(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                main(['train', '--stage', 'unknown', '--config', str(ROOT / 'training/config.yaml')])
        self.assertEqual(result.exception.code, 2)

    @unittest.skipUnless(importlib.util.find_spec('pysam') and importlib.util.find_spec('pyarrow'),
                         'BAM/Parquet dependencies are not installed')
    def test_prediction_cli_connects_candidates_features_and_annotation(self):
        import csv
        from array import array
        import pysam
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fasta = directory / 'reference.fa'
            fasta.write_text('>chr1\n' + 'A' * 100 + '\n')
            pysam.faidx(str(fasta))
            bam = directory / 'reads.bam'
            with pysam.AlignmentFile(str(bam), 'wb', header=dict(HD=dict(SO='coordinate'),
                    SQ=[dict(SN='chr1', LN=100)])) as handle:
                read = pysam.AlignedSegment(handle.header)
                read.query_name = 'synthetic_read'
                read.query_sequence = 'A' * 19
                read.query_qualities = array('B', [30] * 19)
                read.flag, read.reference_id, read.reference_start = 0, 0, 1
                read.cigarstring, read.mapping_quality = '19M', 60
                read.set_tag('MM', 'A+a?,9;')
                read.set_tag('ML', array('B', [220]))
                handle.write(read)
            pysam.index(str(bam))
            (directory / 'calls.bed').write_text('chr1 10 11 a 10 + 10 11 0,0,0 10 30 3 7 0 0 0 0 0\n')
            (directory / 'pileup.log').write_text('setting threshold 0.7 (qual: 179) for base A, percentile 0.1\n')
            (directory / 'reference.gtf').write_text('##genome-build GRCh38\n'
                'chr1\ttest\texon\t1\t100\t.\t+\t.\tgene_id "g"; transcript_id "t";\n'
                'chr1\ttest\tCDS\t1\t100\t.\t+\t0\tgene_id "g"; transcript_id "t";\n')
            values = yaml.safe_load((ROOT / 'prediction/config.yaml').read_text())
            values['output_dir'] = 'output'
            values['targets'] = ['m6A', 'inosine']
            values['nn']['task'] = 'adenine_three_class'
            values['model_root'] = 'models'
            from tests.test_inference import input_schema
            from utils.references.catalog import sha256
            model_directory = directory / 'models/adenine'
            model_directory.mkdir(parents=True)
            (model_directory / 'model.keras').write_bytes(b'CPU preparation must not open model weights')
            model_run = dict(status='complete', role='final_all_labeled_sites', target='adenine',
                model_file='model.keras', model_sha256=sha256(model_directory / 'model.keras'), threshold=.5,
                model=dict(target='adenine', window=19, read_channels=18, max_reads=300, sampling_seed=42),
                input_schema=input_schema())
            (model_directory / 'run.json').write_text(json.dumps(model_run))
            (directory / 'catalog.json').write_text(json.dumps(dict(assembly='GRCh38', sources=[])))
            values['sample'].update(sample_id='synthetic', bam='reads.bam',
                                    bedmethyl='calls.bed', pileup_log='pileup.log')
            values['references'].update(fasta='reference.fa', gtf='reference.gtf', catalog='catalog.json')
            config = directory / 'pipeline.yaml'
            config.write_text(yaml.safe_dump(values))
            for stage in ('candidates', 'features', 'annotation'):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(['predict', '--stage', stage, '--config', str(config)]), 0)
            index = directory / 'output/features/index/adenine/sites.tsv'
            with index.open() as handle:
                row = next(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(row['row_id'], 'synthetic|chr1:10:+:adenine')
            self.assertEqual(row['retained_depth'], '1')
            self.assertEqual(row['label'], '')
            with (directory / 'output/annotation/sites.tsv').open() as handle:
                row = next(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(row['region_cds'], '1')
            self.assertTrue((directory / 'output/features/samples/synthetic/sites').is_dir())


if __name__ == '__main__':
    unittest.main()
