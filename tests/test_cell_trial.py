"""Cell-only launcher contracts; no TensorFlow installation or real fit required."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from training.cell_trial import (PROJECT_ROOT, cell_trial_plan, main,
                                 run_cell_trial, start_background_cell_trial)


def configuration(root):
    path = root / 'config.yaml'
    path.write_text('mode: train\n')
    return dict(config_path=path, output_dir=root / 'outputs', mode='train', targets=['m6A', 'inosine'],
                samples=[dict(sample_id='HEK293T_WT')],
                cell_dataset=dict(training_cell='HEK293T', separate_cell='HeLa', seed=17),
                features=dict(flank=9), cv=dict(seed=42),
                nn=dict(task='adenine_three_class', epochs=20, patience=3, batch_sites=16,
                        learning_rate=0.0003, max_reads=300, sampling_seed=42,
                        target_settings=dict(adenine=dict(threshold=0.5))))


class CellTrialTests(unittest.TestCase):
    def test_plan_preserves_cell_seed_learning_rate_and_original_input_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = configuration(Path(tmp))
            plan = cell_trial_plan(config)
            self.assertEqual(plan['output_dir'], str(config['output_dir'] / 'cell_models/HEK293T'))
            self.assertEqual(plan['model_output'], str(Path(plan['output_dir']) / 'adenine'))
            self.assertEqual(plan['datasets_root'], str(config['output_dir'] / 'cell_datasets'))
            self.assertEqual(plan['settings']['seed'], 17)
            self.assertEqual(plan['settings']['learning_rate'], 0.0003)
            self.assertEqual(plan['settings']['training_cell'], 'HEK293T')
            self.assertEqual(plan['settings']['separate_cell'], 'HeLa')
            self.assertEqual(plan['settings']['batch_sites'], 16)
            for key, bad in [('training_cell', 'HeLa'), ('separate_cell', 'HEK293T')]:
                original = config['cell_dataset'][key]
                config['cell_dataset'][key] = bad
                with self.assertRaises(ValueError):
                    cell_trial_plan(config)
                config['cell_dataset'][key] = original

    def test_dry_run_does_not_import_tensorflow_spawn_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = configuration(Path(tmp))
            captured = io.StringIO()
            with patch('training.cell_trial.load_config', return_value=config), \
                    patch.dict(sys.modules, {'tensorflow': None}), contextlib.redirect_stdout(captured), \
                    patch('training.cell_trial.subprocess.Popen') as spawn:
                self.assertEqual(main(['--background', '--dry-run']), 0)
            self.assertEqual(json.loads(captured.getvalue())['status'], 'planned')
            spawn.assert_not_called()
            self.assertFalse(config['output_dir'].exists())

    def test_existing_root_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = configuration(root)
            out = root / 'preserve'
            out.mkdir()
            sentinel = out / 'previous_model'
            sentinel.write_text('unchanged')
            with patch('training.cell_trial.subprocess.Popen') as spawn:
                for operation in (run_cell_trial, start_background_cell_trial):
                    with self.assertRaises(FileExistsError):
                        operation(config, out)
            spawn.assert_not_called()
            self.assertEqual(sentinel.read_text(), 'unchanged')
            self.assertEqual(list(out.iterdir()), [sentinel])

    def test_background_is_detached_and_preserves_launch_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = configuration(root)
            out = root / 'trials/new'
            with patch('training.cell_trial.subprocess.Popen', return_value=types.SimpleNamespace(pid=1234)) as spawn:
                report = start_background_cell_trial(config, out)
            self.assertEqual(report['status'], 'starting')
            self.assertFalse(out.exists())
            command = spawn.call_args.args[0]
            self.assertEqual(command[:4], [sys.executable, '-u', '-m', 'training.cell_trial'])
            self.assertNotIn('--background', command)
            options = spawn.call_args.kwargs
            self.assertEqual(options['cwd'], PROJECT_ROOT)
            self.assertEqual(options['stdin'], subprocess.DEVNULL)
            self.assertEqual(options['stderr'], subprocess.STDOUT)
            self.assertTrue(options['start_new_session'])
            self.assertNotIn('env', options)
            self.assertEqual(json.loads(Path(report['launch_record']).read_text())['pid'], 1234)
            with self.assertRaises(FileExistsError):
                start_background_cell_trial(config, out)

    def test_gpu_failure_records_failure_before_any_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = configuration(Path(tmp))
            out = Path(tmp) / 'failed'
            cell_model = types.ModuleType('training.cell_model')
            cell_model.train_cell_model = Mock()
            with patch.dict(sys.modules, {'tensorflow': types.ModuleType('tensorflow'),
                                          'training.cell_model': cell_model}), \
                    patch('training.cell_trial.tensorflow_devices', return_value={'tensorflow_version': 'mock'}), \
                    patch('training.cell_trial.require_gpu_runtime', side_effect=RuntimeError('GPU unavailable')):
                with self.assertRaisesRegex(RuntimeError, 'GPU unavailable'):
                    run_cell_trial(config, out)
            cell_model.train_cell_model.assert_not_called()
            report = json.loads((out / 'launch_run.json').read_text())
            self.assertEqual((report['status'], report['phase']), ('failed', 'failed'))
            self.assertEqual(report['error_type'], 'RuntimeError')
            self.assertTrue((out / 'FAILED.txt').exists())
            self.assertTrue((out / 'INCOMPLETE.txt').exists())
            self.assertIn('GPU unavailable', (out / 'execution.log').read_text())

    def test_success_forwards_exact_settings_and_saves_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = configuration(Path(tmp))
            out = Path(tmp) / 'complete'
            gpu = Mock()

            def fit(index, datasets, samples, model_out, **settings):
                gpu.assert_called_once()
                plan = cell_trial_plan(config, out)
                self.assertEqual((index, datasets, samples, model_out),
                                 (plan['index_path'], plan['datasets_root'], config['samples'], plan['model_output']))
                self.assertEqual(settings, plan['settings'])
                return dict(status='complete', feature_set='sequence_reads_repeat18_v1', best_epoch=5)

            cell_model = types.ModuleType('training.cell_model')
            cell_model.train_cell_model = fit
            with patch.dict(sys.modules, {'tensorflow': types.ModuleType('tensorflow'),
                                          'training.cell_model': cell_model}), \
                    patch('training.cell_trial.tensorflow_devices', return_value={'tensorflow_version': 'mock'}), \
                    patch('training.cell_trial.require_gpu_runtime', side_effect=lambda tf: gpu() or {}) as probe:
                gpu.return_value = {}
                report = run_cell_trial(config, out)
            probe.assert_called_once()
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['best_epoch'], 5)
            self.assertFalse((out / 'INCOMPLETE.txt').exists())
            self.assertFalse((out / 'FAILED.txt').exists())
            self.assertEqual((out / 'config.snapshot.yaml').read_bytes(), config['config_path'].read_bytes())
            hashes = json.loads((out / 'source_code_sha256.json').read_text())
            self.assertIn('training/cell_trial.py', hashes)
            self.assertIn('training/cell_model.py', hashes)
            self.assertEqual(json.loads((out / 'resolved_settings.json').read_text())['settings']['seed'], 17)
            self.assertTrue((out / 'execution.log').is_file())


if __name__ == '__main__':
    unittest.main()
