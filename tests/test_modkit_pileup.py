import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from prediction.modkit_pileup import run_modkit_pileup, threshold_arguments


class ModkitPileupTest(unittest.TestCase):
    def fixture(self, root):
        executable = root / 'modkit executable'
        executable.write_text('mocked executable; never executed\n')
        executable.chmod(0o755)
        bam = root / 'input reads.bam'
        bam.write_bytes(b'fixture')
        Path(str(bam) + '.bai').write_bytes(b'index fixture')
        return bam, executable

    def success(self, argv, **kwargs):
        if argv[-1] == '--version':
            return SimpleNamespace(returncode=0, stdout='modkit 0.6.4\n', stderr='')
        self.assertNotIn('shell', kwargs)
        Path(argv[3]).write_text('mock bedMethyl output\n')
        Path(argv[argv.index('--log-filepath') + 1]).write_text('command line: fixture\nDone, processed 1 rows\n')
        kwargs['stdout'].write('pileup finished\n')
        return SimpleNamespace(returncode=0)

    def test_defaults_preserve_confidence_estimation_all_codes_and_original_bam(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bam, executable = self.fixture(root)
            with patch('prediction.modkit_pileup.subprocess.run', side_effect=self.success) as run:
                report = run_modkit_pileup(bam, root / 'pileup', executable=executable)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(report['modkit_version'], 'modkit 0.6.4')
            self.assertEqual(Path(report['bedmethyl']).read_text(), 'mock bedMethyl output\n')
            self.assertEqual(bam.read_bytes(), b'fixture')
            for option in ('--filter-threshold', '--mod-threshold', '--modified-bases', '--combine-mods', '--combine-strands'):
                self.assertNotIn(option, report['argv'])
            self.assertFalse((root / 'pileup/INCOMPLETE.txt').exists())
            self.assertEqual(json.loads((root / 'pileup/command.json').read_text())['argv'], report['argv'])
            self.assertEqual(json.loads((root / 'pileup/run.json').read_text())['status'], 'complete')

    def test_explicit_reference_include_bed_thresholds_and_seed_are_separate_argv_items(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bam, executable = self.fixture(root)
            reference = root / 'genome.fa'
            reference.write_text('>chr1\nA\n')
            Path(str(reference) + '.fai').write_text('chr1\t1\t6\t1\t2\n')
            include = root / 'known regions.bed'
            include.write_text('chr1\t0\t1\n')
            with patch('prediction.modkit_pileup.subprocess.run', side_effect=self.success):
                report = run_modkit_pileup(bam, root / 'pileup', executable=executable, threads=8,
                    reference_fasta=reference, include_bed=include, seed=42,
                    filter_threshold={'A': .7, 'C': .8}, mod_thresholds={'a': .9, 17596: .95})
            argv = report['argv']
            self.assertEqual(argv[0], str(executable))
            self.assertEqual(argv[2], str(bam))
            self.assertEqual(argv[argv.index('--include-bed') + 1], str(include))
            self.assertEqual(argv[argv.index('--reference') + 1], str(reference))
            self.assertEqual(argv[argv.index('--seed') + 1], '42')
            self.assertIn('17596:0.95', argv)
            self.assertEqual(argv.count('--filter-threshold'), 2)
            self.assertEqual(argv.count('--mod-threshold'), 2)

    def test_existing_output_or_missing_index_fails_before_invoking_modkit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bam, executable = self.fixture(root)
            output = root / 'pileup'
            output.mkdir()
            (output / 'keep.txt').write_text('untouched')
            with patch('prediction.modkit_pileup.subprocess.run') as run:
                with self.assertRaises(FileExistsError):
                    run_modkit_pileup(bam, output, executable=executable)
                Path(str(bam) + '.bai').unlink()
                with self.assertRaisesRegex(FileNotFoundError, 'BAM index'):
                    run_modkit_pileup(bam, root / 'other', executable=executable)
            run.assert_not_called()
            self.assertEqual((output / 'keep.txt').read_text(), 'untouched')

    def test_failure_and_changed_input_never_leave_completed_pileup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bam, executable = self.fixture(root)
            def failed(argv, **kwargs):
                if argv[-1] == '--version':
                    return self.success(argv, **kwargs)
                kwargs['stderr'].write('fixture process failed')
                return SimpleNamespace(returncode=9)
            with patch('prediction.modkit_pileup.subprocess.run', side_effect=failed):
                with self.assertRaisesRegex(RuntimeError, 'exit code 9'):
                    run_modkit_pileup(bam, root / 'failed', executable=executable)
            self.assertTrue((root / 'failed/FAILED.txt').is_file())
            self.assertTrue((root / 'failed/INCOMPLETE.txt').is_file())
            self.assertFalse((root / 'failed/run.json').exists())
            def changed(argv, **kwargs):
                result = self.success(argv, **kwargs)
                if argv[-1] != '--version':
                    bam.write_bytes(b'changed BAM fixture')
                return result
            with patch('prediction.modkit_pileup.subprocess.run', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'input changed'):
                    run_modkit_pileup(bam, root / 'changed', executable=executable)
            self.assertFalse((root / 'changed/run.json').exists())

    def test_invalid_confidence_values_and_duplicate_base_thresholds_are_rejected(self):
        for value in (-.1, 1.1, float('nan'), True, ['A:.7', 'A:.8'], {'Z': .7}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                threshold_arguments(value, None)
        with self.assertRaises(ValueError):
            threshold_arguments(None, {'m6A': .8})
        self.assertEqual(threshold_arguments(.8, None), ['--filter-threshold', '0.8'])


if __name__ == '__main__':
    unittest.main()
