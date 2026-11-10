"""Supervisor guards and stage control; no BAM extraction or NN inference."""
from contextlib import nullcontext, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from prediction.launcher import (main, prediction_plan, run_prediction,
                                 start_background_prediction)
from utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class PredictionLauncherTest(unittest.TestCase):
    def config(self, directory):
        data = yaml.safe_load((ROOT / 'prediction/config.yaml').read_text())
        data.update(output_dir=str(directory / 'result'), model_root=None,
                    model_directories={'adenine': str(directory / 'model')})
        data['sample'].update(sample_id='HeLa_WT', cell='HeLa', reference_cell='HeLa')
        data['nn'].update(device='gpu', batch_sites=16, cpu_threads=4)
        path = directory / 'config.yaml'
        path.write_text(yaml.safe_dump(data))
        return load_config(path, 'predict')

    def test_plan_keeps_model_config_and_default_no_bam(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp))
            plan = prediction_plan(config)
            self.assertEqual(plan['sample_id'], 'HeLa_WT')
            self.assertEqual(plan['device'], 'gpu')
            self.assertFalse(plan['recalibrate_bam'])
            self.assertEqual(plan['stages'], ['candidates', 'features', 'inputs', 'infer', 'filter'])
            self.assertFalse(config['output_dir'].exists())

    def test_cli_dry_run_opens_no_inputs_or_tensorflow(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp))
            text = StringIO()
            had_tensorflow = 'tensorflow' in sys.modules
            with patch('prediction.pipeline.validate_prediction_inputs', side_effect=AssertionError('inputs opened')), \
                 patch('prediction.pipeline.run_prediction_stage', side_effect=AssertionError('stage ran')), \
                 redirect_stdout(text):
                result = main(['--config', str(config['config_path']), '--background', '--dry-run'])
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(text.getvalue())['status'], 'planned')
            self.assertEqual('tensorflow' in sys.modules, had_tensorflow)
            self.assertFalse(config['output_dir'].exists())

    def test_existing_output_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp))
            config['output_dir'].mkdir()
            old = config['output_dir'] / 'old.txt'
            old.write_text('keep')
            with patch('prediction.pipeline.validate_prediction_inputs') as validate:
                for function in (run_prediction, start_background_prediction):
                    with self.assertRaises(FileExistsError):
                        function(config)
                validate.assert_not_called()
            self.assertEqual(old.read_text(), 'keep')

    def test_success_runs_all_stages_with_isolated_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp))
            with patch('prediction.pipeline.validate_prediction_inputs') as validate, \
                 patch('prediction.pipeline.run_prediction_stage', return_value={'status': 'complete'}) as stage, \
                 patch('prediction.launcher.execution_log', return_value=nullcontext()):
                result = run_prediction(config)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['completed_stages'], prediction_plan(config)['stages'])
            self.assertEqual([call.args[1] for call in stage.call_args_list], result['completed_stages'])
            self.assertTrue(all(call.kwargs['isolate_gpu'] for call in stage.call_args_list))
            validate.assert_called_once_with(config, result['stages'])
            out = config['output_dir']
            self.assertFalse((out / 'INCOMPLETE.txt').exists())
            self.assertEqual((out / 'config.snapshot.yaml').read_bytes(), config['config_path'].read_bytes())
            self.assertTrue((out / 'source_code_sha256.json').is_file())
            self.assertEqual(json.loads((out / 'run.json').read_text())['status'], 'complete')

    def test_failure_records_exact_stage_and_stops_later_stages(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp))
            with patch('prediction.pipeline.validate_prediction_inputs'), \
                 patch('prediction.pipeline.run_prediction_stage', side_effect=[{}, RuntimeError('extraction failed')]) as stage, \
                 patch('prediction.launcher.execution_log', return_value=nullcontext()):
                with self.assertRaisesRegex(RuntimeError, 'extraction failed'):
                    run_prediction(config)
            self.assertEqual(stage.call_count, 2)
            out = config['output_dir']
            run = json.loads((out / 'run.json').read_text())
            self.assertEqual((run['status'], run['current_stage']), ('failed', 'features'))
            self.assertEqual(run['completed_stages'], ['candidates'])
            self.assertTrue((out / 'FAILED.txt').exists())
            self.assertTrue((out / 'INCOMPLETE.txt').exists())

    def test_background_is_detached_and_launch_records_are_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp))
            with patch('prediction.pipeline.validate_prediction_inputs'), \
                 patch('prediction.launcher.subprocess.Popen', return_value=SimpleNamespace(pid=123)) as popen:
                report = start_background_prediction(config)
                self.assertEqual(report['pid'], 123)
                self.assertEqual(popen.call_args.kwargs['stdin'], subprocess.DEVNULL)
                self.assertTrue(popen.call_args.kwargs['start_new_session'])
                command = popen.call_args.args[0]
                self.assertIn('prediction.launcher', command)
                self.assertNotIn('--background', command)
                self.assertEqual(command[-1], str(config['config_path']))
                self.assertTrue(Path(report['launch_record']).is_file())
                self.assertFalse(config['output_dir'].exists())
                with self.assertRaises(FileExistsError):
                    start_background_prediction(config)
                self.assertEqual(popen.call_count, 1)


if __name__ == '__main__':
    unittest.main()
