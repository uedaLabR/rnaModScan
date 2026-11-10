"""Test the public prediction command and its input checks with synthetic paths."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from prediction.pipeline import (prediction_stages, run_prediction_pipeline,
                                 run_prediction_phase, run_isolated_inference)
from rnamodscan.cli import main
from utils.config import load_config, joint_adenine, neural_prediction_targets

ROOT = Path(__file__).resolve().parents[1]


class PredictionCliTest(unittest.TestCase):
    def write_config(self, directory, recalibrate=False):
        values = yaml.safe_load((ROOT / 'prediction/config.yaml').read_text())
        values.update(output_dir='output', targets=['m6A', 'inosine'], model_root='models',
                      recalibrate_bam=recalibrate)
        values['nn']['task'] = 'adenine_three_class'
        values['sample'].update(bam='reads.bam', bedmethyl='calls.bed', pileup_log='pileup.log')
        values['references'].update(fasta='genome.fa', gtf='genes.gtf', catalog='catalog.json')
        path = directory / 'config.yaml'
        path.write_text(yaml.safe_dump(values))
        return path

    def dry_run(self, path, *options):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = main(['predict', '--config', str(path), '--dry-run', *options])
        self.assertEqual(result, 0)
        return json.loads(output.getvalue())

    def touch_inputs(self, config):
        paths = [config['sample'][name] for name in ('bam', 'bedmethyl', 'pileup_log')]
        paths.extend(path for path in config['references'].values() if path is not None)
        paths.extend([Path(str(config['sample']['bam']) + '.bai'),
                      Path(str(config['references']['fasta']) + '.fai')])
        model = config['model_root'] / 'adenine'
        paths.extend(model / name for name in ('model.keras', 'run.json'))
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        config['references']['catalog'].write_text(json.dumps(dict(assembly='GRCh38', sources=[])))

    def test_complete_prediction_is_default_and_dry_run_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            report = self.dry_run(self.write_config(directory))
            self.assertEqual(report['planned_stages'], ['candidates', 'features', 'annotation', 'inputs', 'infer', 'filter'])
            self.assertFalse(report['recalibrate_bam'])
            self.assertFalse((directory / 'output').exists())

    def test_recalibration_options_override_yaml_in_both_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = self.write_config(directory)
            report = self.dry_run(path, '--recalibrate')
            self.assertEqual(report['planned_stages'][-1], 'bam')
            self.assertTrue(report['recalibrate_bam'])
            self.assertFalse(load_config(path)['recalibrate_bam'])
            path = self.write_config(directory, recalibrate=True)
            self.assertEqual(self.dry_run(path)['planned_stages'][-1], 'bam')
            self.assertEqual(self.dry_run(path, '--no-recalibrate')['planned_stages'][-1], 'filter')

    def test_cpu_and_gpu_phases_have_separate_inputs_in_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(Path(tmp))
            prepared = self.dry_run(path, '--stage', 'prepare')
            self.assertEqual(prepared['planned_stages'], ['candidates', 'features', 'annotation', 'inputs'])
            inference = self.dry_run(path, '--stage', 'infer')
            self.assertNotIn('references.gtf', inference['unset_inputs'])
            self.assertNotIn('analysis_sample.bam', inference['unset_inputs'])
            filtered = self.dry_run(path, '--stage', 'filter', '--recalibrate')
            self.assertEqual(filtered['planned_stages'], ['filter', 'bam'])

    def test_filtering_does_not_require_raw_features_or_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(self.write_config(Path(tmp)), 'predict')
            config['references']['catalog'].write_text(json.dumps(dict(assembly='GRCh38', sources=[])))
            with patch('prediction.pipeline.run_prediction_stage', return_value={}) as run:
                run_prediction_phase(config, 'filter', False)
            self.assertEqual([call.args[1] for call in run.call_args_list], ['filter'])

    def test_complete_run_inference_is_a_separate_process(self):
        config = load_config(ROOT / 'prediction/config.yaml', 'predict')
        response = type('Response', (), dict(returncode=0, stdout='{"m6A": {"status": "complete"}}'))()
        with patch('subprocess.run', return_value=response) as run:
            result = run_isolated_inference(config)
        self.assertEqual(result['m6A']['status'], 'complete')
        command = run.call_args.args[0]
        self.assertEqual(command[-2:], ['--stage', 'infer'])
        self.assertIn('prediction', command)

    def test_device_overrides_yaml_and_reaches_isolated_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(Path(tmp))
            report = self.dry_run(path, '--device', 'cpu', '--cpu-threads', '2')
            self.assertEqual((report['device'], report['cpu_threads']), ('cpu', 2))
            self.assertTrue(all(stage['device'] == 'cpu' for stage in report['stages']))
            self.assertEqual(load_config(path, 'predict')['nn']['device'], 'auto')
            with patch('prediction.pipeline.run_prediction_pipeline', return_value={}) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    main(['predict', '--config', str(path), '--device', 'cpu', '--cpu-threads', '2'])
            self.assertEqual(run.call_args.args[0]['nn']['device'], 'cpu')
            self.assertEqual(run.call_args.args[0]['nn']['cpu_threads'], 2)

    def test_cpu_worker_hides_gpu_and_preserves_thread_override(self):
        config = load_config(ROOT / 'prediction/config.yaml', 'predict')
        config['nn'].update(device='cpu', cpu_threads=2)
        response = type('Response', (), dict(returncode=0, stdout='{}'))()
        with patch('subprocess.run', return_value=response) as run:
            run_isolated_inference(config)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('--device') + 1], 'cpu')
        self.assertEqual(command[command.index('--cpu-threads') + 1], '2')
        self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '-1')

    def test_invalid_device_and_thread_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(Path(tmp))
            values = yaml.safe_load(path.read_text())
            for key, value in [('device', 'tpu'), ('cpu_threads', 0), ('cpu_threads', True)]:
                changed = dict(values, nn=dict(values['nn'], **{key: value}))
                path.write_text(yaml.safe_dump(changed))
                with self.assertRaisesRegex(ValueError, 'nn.' + key):
                    load_config(path, 'predict')

    def test_pipeline_checks_inputs_then_executes_each_stage_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(self.write_config(Path(tmp)), 'predict')
            self.touch_inputs(config)
            with patch('prediction.pipeline.run_prediction_stage', return_value={'status': 'complete'}) as run:
                report = run_prediction_pipeline(config, recalibrate=True)
            self.assertEqual([call.args[1] for call in run.call_args_list],
                             ['candidates', 'features', 'annotation', 'inputs', 'infer', 'filter', 'bam'])
            self.assertEqual(list(report['stages']), ['candidates', 'features', 'annotation', 'inputs', 'infer', 'filter', 'bam'])
            self.assertTrue(report['recalibrate_bam'])

    def test_missing_model_fails_before_any_output_or_stage_is_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(self.write_config(Path(tmp)), 'predict')
            self.touch_inputs(config)
            (config['model_root'] / 'adenine/model.keras').unlink()
            with patch('prediction.pipeline.run_prediction_stage') as run:
                with self.assertRaisesRegex(FileNotFoundError, 'adenine model bundle'):
                    run_prediction_pipeline(config)
            run.assert_not_called()
            self.assertFalse(config['output_dir'].exists())

    def test_existing_stage_output_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(self.write_config(Path(tmp)), 'predict')
            self.touch_inputs(config)
            (config['output_dir'] / 'features').mkdir(parents=True)
            with patch('prediction.pipeline.run_prediction_stage') as run:
                with self.assertRaisesRegex(FileExistsError, 'Use a new output directory'):
                    run_prediction_pipeline(config)
            run.assert_not_called()

    def test_annotation_is_optional_output_metadata(self):
        config = load_config(ROOT / 'prediction/config.yaml', 'predict')
        self.assertNotIn('annotation', prediction_stages(config))
        config['references']['gtf'] = Path('/metadata_only.gtf')
        self.assertIn('annotation', prediction_stages(config))

    def test_public_prediction_dispatches_complete_pipeline_with_option(self):
        with patch('prediction.pipeline.run_prediction_pipeline', return_value={'status': 'complete'}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['predict', '--config', str(ROOT / 'prediction/config.yaml'),
                                       '--recalibrate']), 0)
        config, recalibrate = run.call_args.args
        self.assertEqual(config['mode'], 'predict')
        self.assertTrue(recalibrate)

    def test_invalid_yaml_and_nonboolean_recalibration_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / 'broken.yaml'
            path.write_text('mode: [predict\n')
            with self.assertRaisesRegex(ValueError, 'Invalid YAML configuration'):
                load_config(path, 'predict')

            path = self.write_config(directory)
            values = yaml.safe_load(path.read_text())
            values['recalibrate_bam'] = 'false'
            path.write_text(yaml.safe_dump(values))
            with self.assertRaisesRegex(ValueError, 'recalibrate_bam must be true or false'):
                load_config(path, 'predict')

    def test_joint_defaults_reject_old_binary_and_unbounded_read_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(Path(tmp))
            values = yaml.safe_load(path.read_text())
            values['nn'].pop('task')
            values['nn'].pop('max_reads', None)
            path.write_text(yaml.safe_dump(values))
            loaded = load_config(path, 'predict')
            self.assertEqual((loaded['nn']['task'], loaded['nn']['max_reads']), ('adenine_three_class', 300))
            for field, value in [('task', 'binary'), ('max_reads', None)]:
                changed = dict(values, nn=dict(values['nn'], **{field: value}))
                path.write_text(yaml.safe_dump(changed))
                with self.assertRaises(ValueError):
                    load_config(path, 'predict')
            values['targets'] = ['m6A']
            path.write_text(yaml.safe_dump(values))
            with self.assertRaisesRegex(ValueError, 'both m6A and inosine'):
                load_config(path, 'predict')
            values['targets'] = ['m5C', 'pseU']
            path.write_text(yaml.safe_dump(values))
            loaded = load_config(path, 'predict')
            self.assertFalse(joint_adenine(loaded))
            self.assertEqual(neural_prediction_targets(loaded), [])


if __name__ == '__main__':
    unittest.main()
