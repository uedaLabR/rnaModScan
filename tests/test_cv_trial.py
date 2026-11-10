"""Validate detached fold scheduling without TensorFlow or real training."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from training.cv_trial import (check_comparison, run_cv_trial, save_config_snapshot,
                               start_trial, trial_plan)
from utils.config import load_config
from utils.references.catalog import sha256

ROOT = Path(__file__).resolve().parents[1]


class CvTrialTest(unittest.TestCase):
    def config(self):
        return load_config(ROOT / 'training/config_lr0003.yaml', 'train')

    def baseline(self, root):
        base = root / 'old/adenine/fold_0'
        for relative, data in [('run.json', '{"status": "complete"}'),
                               ('train/best.keras', 'preserved model bytes'),
                               ('train/history.csv', 'epoch,val_loss\n0,0.1\n'),
                               ('validation/metrics.json', '{"old": true}')]:
            path = base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data)
        original = ROOT / 'training/config.yaml'
        (base.parents[1] / 'launch_run.json').write_text(json.dumps(dict(
            config_path=str(original), config_sha256=sha256(original))))
        return base

    def test_snapshot_preserves_every_resolved_input_and_setting(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.config()
            snapshot = Path(tmp) / 'config.snapshot.yaml'
            save_config_snapshot(source, snapshot)
            reloaded = load_config(snapshot, 'train')
            expected = {k: v for k, v in source.items() if k != 'config_path'}
            self.assertEqual({k: v for k, v in reloaded.items() if k != 'config_path'}, expected)
            self.assertEqual(reloaded['nn']['learning_rate'], 0.0003)
            self.assertEqual(reloaded['output_dir'], ROOT / 'results/training_v10')

    def test_only_learning_rate_may_change_from_preserved_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = self.baseline(Path(tmp))
            config = self.config()
            with patch('training.cv_trial.check_inputs') as inputs:
                checked = check_comparison(config, baseline)
                self.assertEqual(checked['nn']['learning_rate'], 0.001)
                inputs.assert_called_once_with(config, ('fit',))
                config['nn']['batch_sites'] *= 2
                with self.assertRaisesRegex(ValueError, 'Only nn.learning_rate'):
                    check_comparison(config, baseline)

    def test_start_detaches_new_run_and_never_overwrites_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.baseline(root)
            before = {str(p): p.read_bytes() for p in baseline.rglob('*') if p.is_file()}
            original_config = (ROOT / 'training/config.yaml').read_bytes()
            out = root / 'new'
            with patch('training.cv_trial.check_inputs'), \
                    patch('training.validation_comparison.summarize_validation_folds'), \
                    patch('training.cv_trial.subprocess.Popen', return_value=types.SimpleNamespace(pid=999)) as popen:
                report = start_trial(self.config(), out, baseline, ['0', '1'])
                self.assertEqual(report['pid'], 999)
                self.assertEqual(set(report['folds']), {'0', '1', '2', '3', '4'})
                self.assertTrue(all(row['status'] == 'pending' for row in report['folds'].values()))
                self.assertTrue(popen.call_args.kwargs['start_new_session'])
                self.assertIn('--worker', popen.call_args.args[0])
                self.assertTrue((out / 'INCOMPLETE.txt').is_file())
                with self.assertRaises(FileExistsError):
                    start_trial(self.config(), out, baseline, ['0', '1'])
            self.assertEqual(before, {str(p): p.read_bytes() for p in baseline.rglob('*') if p.is_file()})
            self.assertEqual((ROOT / 'training/config.yaml').read_bytes(), original_config)

    def scheduled_run(self, root, failed_fold=None):
        out = root / 'new'
        (out / 'workers').mkdir(parents=True)
        (out / 'logs').mkdir()
        (out / 'INCOMPLETE.txt').write_text('Running')
        save_config_snapshot(self.config(), out / 'config.snapshot.yaml')
        preserved = root / 'preserved.txt'
        preserved.write_text('unchanged')
        (out / 'run.json').write_text(json.dumps(dict(
            status='starting', baseline_sha256={str(preserved): sha256(preserved)},
            folds={str(fold): dict(status='pending') for fold in range(5)})))
        active, launches, completed = {}, [], []
        testcase = self

        class Worker:
            def __init__(self, command, **kwargs):
                self.fold = int(command[command.index('--fold') + 1])
                self.gpu = kwargs['env']['CUDA_VISIBLE_DEVICES']
                self.out = Path(command[command.index('--output-dir') + 1])
                self.pid = 1000 + self.fold
                self.polls = 0
                testcase.assertNotIn(self.gpu, active)
                testcase.assertTrue(kwargs['start_new_session'])
                testcase.assertEqual(Path(command[command.index('--fold-output-dir') + 1]),
                                     out / 'adenine' / f'fold_{self.fold}')
                active[self.gpu] = self.fold
                launches.append((self.fold, self.gpu))
                testcase.assertLessEqual(len(active), 2)

            def poll(self):
                self.polls += 1
                if self.polls == 1:
                    return None
                self.out.mkdir(parents=True)
                failed = self.fold == failed_fold
                (self.out / 'launch_run.json').write_text(json.dumps(dict(
                    status='failed' if failed else 'complete', error='synthetic failure' if failed else None)))
                del active[self.gpu]
                completed.append(self.fold)
                return 7 if failed else 0

        with patch('training.cv_trial.subprocess.Popen', side_effect=Worker), \
                patch('training.cv_trial.time.sleep'), \
                patch('training.validation_comparison.summarize_validation_folds') as summarize:
            report = run_cv_trial(out, root / 'baseline', ['0', '1'])
        self.assertEqual([fold for fold, _ in launches], list(range(5)))
        self.assertEqual(sorted(completed), list(range(5)))
        self.assertFalse(active)
        self.assertGreaterEqual(summarize.call_count, 7)
        self.assertEqual(preserved.read_text(), 'unchanged')
        return out, report, launches

    def test_queue_runs_all_folds_with_at_most_one_worker_per_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, report, launches = self.scheduled_run(Path(tmp))
            self.assertEqual(report['status'], 'complete')
            self.assertTrue(all(row['status'] == 'complete' for row in report['folds'].values()))
            self.assertEqual(launches, [(0, '0'), (1, '1'), (2, '0'), (3, '1'), (4, '0')])
            self.assertFalse((out / 'INCOMPLETE.txt').exists())

    def test_failed_fold_does_not_skip_remaining_folds_or_report_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, report, _ = self.scheduled_run(Path(tmp), failed_fold=1)
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['folds']['1']['status'], 'failed')
            self.assertEqual(report['folds']['1']['returncode'], 7)
            self.assertEqual(report['folds']['4']['status'], 'complete')
            self.assertTrue((out / 'INCOMPLETE.txt').exists())

    def test_repeated_gpu_id_is_rejected_before_launch(self):
        with self.assertRaisesRegex(ValueError, 'distinct numeric GPU'):
            trial_plan(self.config(), Path('/new'), Path('/old'), ['0', '0'])
