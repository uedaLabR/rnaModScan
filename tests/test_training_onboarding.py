"""Fresh training setup is portable, read-only checked and never starts jobs."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from training.cli import main
from training.onboarding import check_training, initialize, planned_stages
from utils.config import load_config


class TrainingOnboardingTest(unittest.TestCase):
    def test_blank_sample_cell_is_rejected_even_with_an_explicit_reference_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.yaml'
            with self.assertRaisesRegex(ValueError, 'cell'):
                initialize(path, cell='   ', reference_cell='leaf')
            self.assertFalse(path.exists())

    def make_config(self, root, **options):
        path = root/'configs/train.yaml'
        initialize(path, cell='leaf', reference_cell='leaf', catalog=root/'catalog.json',
                   input_dir=root/'inputs', run_dir=root/'run', **options)
        return path

    def ready_files(self, root, *, omit_inosine=False, assembly='TAIR10', source_cell='leaf'):
        path = self.make_config(root)
        config = load_config(path, 'train')
        sources = []
        for target in ['m6A', 'inosine']:
            if omit_inosine and target == 'inosine':
                continue
            bed = root/(target+'.bed')
            bed.write_text('chr1\t0\t1\tpositive\t0\t+\n')
            sources.append(dict(id=target, assembly=assembly, coordinates='BED0',
                                kind='short_read', cell=source_cell, modification=target, path=bed.name))
        (root/'catalog.json').write_text(json.dumps(dict(assembly=assembly, sources=sources)))
        for sample in config['samples']:
            for key in ['bam', 'bedmethyl', 'pileup_log']:
                sample[key].parent.mkdir(parents=True, exist_ok=True)
                sample[key].write_text('placeholder for metadata-only test')
            Path(str(sample['bam'])+'.bai').write_text('index fixture')
        fasta = config['references']['fasta']
        fasta.write_text('>chr1\nAaaa\n')
        Path(str(fasta)+'.fai').write_text('chr1\t4\t6\t4\t5\n')
        return path, config

    def test_initialize_writes_portable_paths_and_explicit_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.make_config(root)
            config = load_config(path, 'train')
            self.assertEqual(config['references']['catalog'], root/'catalog.json')
            self.assertEqual(config['output_dir'], root/'run')
            self.assertIsNone(config['selection']['source_dir'])
            self.assertTrue(config['selection']['enabled'])
            self.assertIsNone(config['references']['gtf'])
            self.assertEqual([s['condition'] for s in config['samples']], ['WT', 'IVT'])
            self.assertEqual({s['reference_cell'] for s in config['samples']}, {'leaf'})
            self.assertEqual(config['nn']['learning_rate'], 0.0003)
            self.assertFalse((root/'run').exists())
            raw = yaml.safe_load(path.read_text())
            self.assertFalse(Path(raw['references']['catalog']).is_absolute())
            self.assertEqual(raw['targets'], ['m6A', 'inosine'])

    def test_initialize_keeps_existing_configuration_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.make_config(root)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                initialize(path)
            self.assertEqual(path.read_bytes(), original)
            (root/'existing').mkdir()
            (root/'existing/data').write_text('keep')
            with self.assertRaises(FileExistsError):
                initialize(root/'different.yaml', run_dir=root/'existing')
            self.assertFalse((root/'different.yaml').exists())
            self.assertEqual((root/'existing/data').read_text(), 'keep')

    def test_invalid_identity_does_not_create_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'train.yaml'
            for options in [dict(cell=' '), dict(wt_id='same', ivt_id='same'), dict(wt_id='WT sample')]:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    initialize(path, **options)
                self.assertFalse(path.exists())

    def test_metadata_check_accepts_explicit_other_assembly_without_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config = self.ready_files(root)
            report = check_training(config, planned_stages(config), skip_dependencies=True)
            self.assertEqual(report['status'], 'ready', report['errors'])
            self.assertEqual(report['catalog_assembly'], 'TAIR10')
            self.assertEqual(report['skipped_optional_stages'], ['annotation'])
            self.assertTrue(report['read_only'])
            self.assertFalse(report['jobs_started'])
            self.assertFalse((root/'run').exists())

    def test_missing_inputs_are_reported_together_without_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.make_config(root)
            config = load_config(path, 'train')
            report = check_training(config, planned_stages(config), skip_dependencies=True)
            self.assertEqual(report['status'], 'not_ready')
            settings = {error['setting'] for error in report['errors']}
            self.assertIn('samples.sample_WT.bam', settings)
            self.assertIn('samples.sample_IVT.bedmethyl', settings)
            self.assertIn('references.fasta', settings)
            self.assertIn('references.catalog', settings)
            self.assertFalse((root/'run').exists())

    def test_one_positive_modification_is_not_enough_for_joint_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config = self.ready_files(root, omit_inosine=True)
            report = check_training(config, planned_stages(config), skip_dependencies=True)
            self.assertEqual(report['status'], 'not_ready')
            self.assertTrue(any('inosine' in e['message'] for e in report['errors']))
            # Building labels alone may legitimately leave one modification unknown.
            report = check_training(config, ('sites',), skip_dependencies=True)
            self.assertEqual(report['status'], 'ready')
            self.assertTrue(any('inosine' in warning for warning in report['warnings']))

    def test_reference_cell_is_not_silently_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config = self.ready_files(root, source_cell='different_cell')
            report = check_training(config, planned_stages(config), skip_dependencies=True)
            self.assertEqual(report['status'], 'not_ready')
            self.assertTrue(any('leaf' in e['message'] and 'positive-reference' in e['message'] for e in report['errors']))

    def test_catalog_assembly_and_bed_coordinate_mismatches_fail_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config = self.ready_files(root)
            manifest = json.loads((root/'catalog.json').read_text())
            manifest['sources'][0]['assembly'] = 'different'
            manifest['sources'][1]['coordinates'] = '1-based'
            (root/'catalog.json').write_text(json.dumps(manifest))
            report = check_training(config, planned_stages(config), skip_dependencies=True)
            self.assertEqual(report['status'], 'not_ready')
            self.assertEqual(sum('declared assembly and BED0' in e['message'] for e in report['errors']), 2)

    def test_existing_stage_output_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config = self.ready_files(root)
            (root/'run/sites').mkdir(parents=True)
            original = root/'run/sites/original.tsv'
            original.write_text('keep labels')
            report = check_training(config, planned_stages(config), skip_dependencies=True)
            self.assertEqual(report['status'], 'not_ready')
            self.assertTrue(any(e['setting']=='output_dir' for e in report['errors']))
            self.assertEqual(original.read_text(), 'keep labels')

    def test_all_pipeline_requires_all_folds_before_any_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config = self.ready_files(root)
            config['nn']['folds'] = [0]
            report = check_training(config, planned_stages(config), skip_dependencies=True)
            self.assertTrue(any(e['setting']=='nn.folds' for e in report['errors']))
            self.assertFalse((root/'run').exists())

    def test_init_and_check_commands_keep_exit_status_and_json_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'train.yaml'
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(['init', '--output', str(path), '--cell', 'leaf']), 0)
            self.assertEqual(json.loads(output.getvalue())['status'], 'created')
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(['check', '--config', str(path), '--skip-dependencies']), 2)
            self.assertFalse(json.loads(output.getvalue())['jobs_started'])

    def test_legacy_dry_run_remains_lightweight_and_skips_optional_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_config(Path(tmp))
            with patch('training.onboarding.check_training') as preflight, contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(['--config', str(path), '--dry-run']), 0)
            preflight.assert_not_called()
            report = json.loads(output.getvalue())
            self.assertEqual(report['planned_stages'], ['sites','select','features','split','fit','final'])
            self.assertEqual(report['skipped_optional_stages'], ['annotation'])

    def test_public_init_and_dry_run_do_not_import_neural_or_bam_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'train.yaml'
            code = ('import sys; from rnamodscan.cli import main; '
                    f'main(["train","init","--output",{str(path)!r}]); '
                    f'main(["train","--config",{str(path)!r},"--dry-run"]); '
                    'assert "tensorflow" not in sys.modules; assert "pysam" not in sys.modules')
            result = subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,
                                    cwd=Path(__file__).resolve().parents[1])
            self.assertEqual(result.returncode,0,result.stderr)

    def test_all_run_fails_before_pipeline_when_preflight_reports_missing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_config(Path(tmp))
            report = dict(status='not_ready', errors=[dict(message='missing inputs')])
            with patch('training.onboarding.check_training', return_value=report), \
                    patch('training.pipeline.run_training_stage') as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--config', str(path)]), 2)
            run.assert_not_called()

    def test_individual_legacy_stage_keeps_dispatch_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_config(Path(tmp))
            with patch('training.pipeline.run_training_stage', return_value=dict(status='mocked')) as run, \
                    patch('training.onboarding.check_training') as preflight, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--config', str(path), '--stage', 'sites']), 0)
            self.assertEqual(run.call_args.args[1], 'sites')
            preflight.assert_not_called()

    def test_dependency_failures_are_visible_without_running_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config = self.ready_files(root)
            def importer(name):
                if name == 'tensorflow':
                    raise ImportError('missing NN runtime')
                return type('Available', (), {'__version__':'fixture'})()
            with patch('training.onboarding.importlib.import_module', side_effect=importer):
                report = check_training(config, planned_stages(config))
            self.assertTrue(any(e['setting']=='dependencies.tensorflow' for e in report['errors']))
            self.assertFalse((root/'run').exists())

    @unittest.skipUnless(importlib.util.find_spec('pysam') and importlib.util.find_spec('pyarrow'), 'BAM dependencies unavailable')
    def test_bam_header_must_match_fasta_index_without_scanning_reads(self):
        import pysam
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config = self.ready_files(root)
            for sample in config['samples']:
                with pysam.AlignmentFile(str(sample['bam']), 'wb',
                        header=dict(HD=dict(SO='coordinate'), SQ=[dict(SN='chr1', LN=5)])):
                    pass
                pysam.index(str(sample['bam']))
            (root/'run/sites').mkdir(parents=True)
            (root/'run/sites/run.json').write_text(json.dumps(dict(status='complete')))
            report = check_training(config, ('features',))
            self.assertEqual(report['status'], 'not_ready')
            mismatches = [error for error in report['errors'] if error['setting']=='sample_reference_match']
            self.assertEqual(len(mismatches), 2)
            self.assertTrue(all('does not match FASTA/FAI' in error['message'] for error in mismatches))
            self.assertFalse((root/'run/features').exists())


if __name__ == '__main__':
    unittest.main()
