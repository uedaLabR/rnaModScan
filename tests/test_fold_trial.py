"""Launch metadata and GPU requirements, without raw data or real training."""
import contextlib
import io
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import yaml

from training.fold_trial import PROJECT_ROOT, main, require_gpu_runtime, run_fold_trial, start_background_trial
from utils.config import load_config


def fake_tensorflow(has_gpu=True, operation_device='/device:GPU:0'):
    tf = types.ModuleType('tensorflow')
    tf.__version__ = 'synthetic'
    gpu = types.SimpleNamespace(name='/physical_device:GPU:0', device_type='GPU')
    tf.config = types.SimpleNamespace(
        list_physical_devices=lambda kind=None: [gpu] if has_gpu else [],
        experimental=types.SimpleNamespace(set_memory_growth=Mock(),
                                           get_device_details=lambda device: dict(device_name='synthetic')))
    tf.sysconfig = types.SimpleNamespace(get_build_info=lambda: dict(cuda_version='11.2', cudnn_version='8'))
    tf.test = types.SimpleNamespace(is_built_with_cuda=lambda: True)
    tf.device = lambda name: contextlib.nullcontext()
    tf.constant = lambda value: value
    tf.matmul = lambda left, right: types.SimpleNamespace(device=operation_device, numpy=lambda: [[1.]])
    return tf


class FoldTrialTest(unittest.TestCase):
    def config(self, root):
        path = root / 'config.yaml'
        values = dict(schema_version=1, mode='train', output_dir='outputs', targets=['m6A', 'inosine'],
                      samples=[dict(sample_id='HEK_' + condition, cell='HEK293T',
                                    reference_cell='HEK293T', condition=condition) for condition in ('WT', 'IVT')],
                      nn=dict(task='adenine_three_class', epochs=2, batch_sites=3))
        path.write_text(yaml.safe_dump(values))
        return load_config(path, 'train')

    def test_dry_run_needs_no_gpu_tensorflow_or_raw_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(root)
            captured = io.StringIO()
            with patch.dict(sys.modules, {'tensorflow': None}), contextlib.redirect_stdout(captured), \
                    patch('training.fold_trial.subprocess.Popen') as spawn:
                self.assertEqual(main(['--config', str(config['config_path']), '--background', '--dry-run']), 0)
            spawn.assert_not_called()
            result = json.loads(captured.getvalue())
            self.assertEqual((result['target'], result['fold']), ('adenine', 0))
            self.assertEqual(result['settings']['epochs'], 2)
            self.assertFalse(config['output_dir'].exists())

    def test_background_launch_detaches_once_and_records_pid_without_claiming_fit_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(root)
            out = root / 'trials' / 'isolated'
            with patch('training.fold_trial.subprocess.Popen', return_value=types.SimpleNamespace(pid=4321)) as spawn:
                report = start_background_trial(config, out)
            self.assertEqual(report['status'], 'starting')
            self.assertEqual(report['pid'], 4321)
            self.assertFalse(out.exists())  # The child owns the trial output directory.
            command = spawn.call_args.args[0]
            self.assertEqual(command[:4], [sys.executable, '-u', '-m', 'training.fold_trial'])
            self.assertNotIn('--background', command)
            self.assertEqual(command[command.index('--config') + 1], str(config['config_path']))
            self.assertEqual(command[command.index('--output-dir') + 1], str(out))
            options = spawn.call_args.kwargs
            self.assertEqual(options['cwd'], PROJECT_ROOT)
            self.assertEqual(options['stdin'], subprocess.DEVNULL)
            self.assertEqual(options['stderr'], subprocess.STDOUT)
            self.assertTrue(options['start_new_session'])
            self.assertNotIn('env', options)  # Inherit CUDA_VISIBLE_DEVICES and the caller's environment.
            self.assertEqual(json.loads(Path(report['launch_record']).read_text())['pid'], 4321)
            self.assertTrue(Path(report['launch_log']).is_file())
            with patch('training.fold_trial.subprocess.Popen') as second:
                with self.assertRaises(FileExistsError):
                    start_background_trial(config, out)
            second.assert_not_called()

    def test_missing_gpu_fails_before_fit_and_records_build_information(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(root)
            fit = types.ModuleType('training.fit')
            fit.train_fold = Mock()
            out = root / 'trial'
            with patch.dict(sys.modules, {'tensorflow': fake_tensorflow(False), 'training.fit': fit}):
                with self.assertRaisesRegex(RuntimeError, 'No CPU fallback'):
                    run_fold_trial(config, out)
            fit.train_fold.assert_not_called()
            report = json.loads((out / 'launch_run.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['tensorflow_build']['cuda_version'], '11.2')
            self.assertTrue((out / 'FAILED.txt').is_file())
            self.assertTrue((out / 'INCOMPLETE.txt').is_file())

    def test_gpu_probe_rejects_soft_placement_onto_cpu(self):
        tf = fake_tensorflow(operation_device='/device:CPU:0')
        with self.assertRaisesRegex(RuntimeError, 'refusing CPU fallback'):
            require_gpu_runtime(tf)
        tf.config.experimental.set_memory_growth.assert_called_once()

    def test_success_sets_growth_before_fit_preserves_settings_and_captures_native_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(root)
            tf = fake_tensorflow()
            fit = types.ModuleType('training.fit')
            def train(index, splits, out, target, **settings):
                tf.config.experimental.set_memory_growth.assert_called_once()
                self.assertEqual(target, 'adenine')
                self.assertTrue(str(splits).endswith('/splits/adenine/fold_0'))
                self.assertEqual(settings['epochs'], 2)
                self.assertEqual(settings['batch_sites'], 3)
                print('mock fit stdout')
                os.write(2, b'mock native stderr\n')
                return dict(feature_set='sequence_reads_repeat18_v1')
            fit.train_fold = train
            out = root / 'trial'
            with patch.dict(sys.modules, {'tensorflow': tf, 'training.fit': fit}):
                report = run_fold_trial(config, out)
            self.assertEqual(report['status'], 'complete')
            self.assertTrue(report['gpu_memory_growth'])
            self.assertIn('GPU:', report['gpu_probe_device'])
            self.assertEqual((out / 'config.snapshot.yaml').read_bytes(), config['config_path'].read_bytes())
            log = (out / 'execution.log').read_text()
            self.assertIn('mock fit stdout', log)
            self.assertIn('mock native stderr', log)
            self.assertFalse((out / 'INCOMPLETE.txt').exists())
            self.assertFalse((out / 'FAILED.txt').exists())


if __name__ == '__main__':
    unittest.main()
