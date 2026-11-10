"""Verify file isolation and error policy without converting experimental signals."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analysis.m6anet.convert_pod5_files import convert_files


class SignalConversionTest(unittest.TestCase):
    def fake_tool(self, command, stdout, stderr, **kwargs):
        if command[0] == 'blue-crab':
            source = Path(command[command.index('-o') - 1])
            output = Path(command[-1])
            output.write_bytes(b'partial-or-complete')
            if source.stem == 'bad':
                stdout.write('RuntimeError: Invalid: Compressed signal data is corrupt\n')
                return SimpleNamespace(returncode=1)
            if source.stem == 'disk_full':
                stdout.write('No space left on device\n')
                return SimpleNamespace(returncode=1)
            if source.stem == 'zstd_bad':
                stdout.write('RuntimeError - Invalid: Input data failed to decompress using zstd: (18446744073709551596 Data corruption detected)\n')
                return SimpleNamespace(returncode=1)
            if source.stem == 'zstd_other':
                stdout.write('Input data failed to decompress using zstd: Allocation error\n')
                return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    def prepare(self, root, names):
        source, out = root / 'pod5', root / 'output'
        source.mkdir()
        for name in names:
            (source / (name + '.pod5')).write_bytes(b'input')
        return source, out

    def test_corrupt_file_is_audited_and_partial_output_is_never_merged(self):
        with TemporaryDirectory() as temp:
            source, out = self.prepare(Path(temp), ['good', 'bad'])
            with patch('analysis.m6anet.convert_pod5_files.subprocess.run', self.fake_tool), \
                 patch('analysis.m6anet.convert_pod5_files.read_count', return_value=42):
                result = convert_files(source, out)
            self.assertEqual((result['converted_files'], result['skipped_files'], result['skipped_reads_known']), (1, 1, 42))
            self.assertEqual(len(list((out / 'blow5_parts_verified').glob('*.blow5'))), 1)
            self.assertEqual(len(list((out / 'signal_work').glob('bad*.blow5'))), 1)
            rows = (out / 'signal_conversion.tsv').read_text()
            self.assertIn(str(source / 'bad.pod5'), rows)
            self.assertIn('skipped_corrupt_signal', rows)

    def test_other_errors_stop_and_are_not_silently_skipped(self):
        with TemporaryDirectory() as temp:
            source, out = self.prepare(Path(temp), ['disk_full', 'z_later'])
            with patch('analysis.m6anet.convert_pod5_files.subprocess.run', self.fake_tool), \
                 patch('analysis.m6anet.convert_pod5_files.read_count', return_value=None):
                with self.assertRaises(RuntimeError):
                    convert_files(source, out, workers=1)
            result = json.loads((out / 'signal_conversion.json').read_text())
            self.assertEqual((result['processed_files'], result['failed_files'], result['skipped_files']), (1, 1, 0))

    def test_resume_reuses_verified_files_and_rejects_changed_inputs(self):
        with TemporaryDirectory() as temp:
            source, out = self.prepare(Path(temp), ['good'])
            with patch('analysis.m6anet.convert_pod5_files.subprocess.run', self.fake_tool), \
                 patch('analysis.m6anet.convert_pod5_files.read_count', return_value=42):
                convert_files(source, out)
                with patch('analysis.m6anet.convert_pod5_files.subprocess.run') as run:
                    convert_files(source, out)
                    self.assertEqual(run.call_count, 1)
                    self.assertEqual(run.call_args.args[0][1], 'quickcheck')
                (source / 'good.pod5').write_bytes(b'changed source')
                with self.assertRaisesRegex(ValueError, 'Input changed'):
                    convert_files(source, out)

    def test_all_corrupt_cannot_be_marked_complete(self):
        with TemporaryDirectory() as temp:
            source, out = self.prepare(Path(temp), ['bad'])
            with patch('analysis.m6anet.convert_pod5_files.subprocess.run', self.fake_tool), \
                 patch('analysis.m6anet.convert_pod5_files.read_count', return_value=None):
                with self.assertRaises(RuntimeError):
                    convert_files(source, out)
            result = json.loads((out / 'signal_conversion.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['skipped_files_unknown_read_count'], 1)

    def test_zstd_corrupt_input_is_skipped_but_other_zstd_errors_stop(self):
        for bad, skipped in [('zstd_bad', True), ('zstd_other', False)]:
            with self.subTest(input=bad), TemporaryDirectory() as temp:
                source, out = self.prepare(Path(temp), ['good', bad])
                with patch('analysis.m6anet.convert_pod5_files.subprocess.run', self.fake_tool), \
                     patch('analysis.m6anet.convert_pod5_files.read_count', return_value=42):
                    if skipped:
                        result = convert_files(source, out)
                        self.assertEqual(result['skipped_files'], 1)
                    else:
                        with self.assertRaises(RuntimeError):
                            convert_files(source, out)
                        result = json.loads((out / 'signal_conversion.json').read_text())
                        self.assertEqual(result['skipped_files'], 0)
                        self.assertEqual(result['failed_files'], 1)


if __name__ == '__main__':
    unittest.main()
