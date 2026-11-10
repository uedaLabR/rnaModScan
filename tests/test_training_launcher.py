"""Detect broken native imports early and reuse only an untouched merge cache."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.run_training import check_dependencies, check_outputs, run


class TrainingLauncherTest(unittest.TestCase):
    def test_preflight_imports_packages_instead_of_only_finding_them(self):
        with patch('training.run_training.importlib.import_module') as importer:
            check_dependencies(False)
        self.assertEqual([call.args[0] for call in importer.call_args_list],
                         ['ctypes', 'yaml', 'pysam', 'pyarrow'])

    def test_missing_ctypes_identifies_python_installation(self):
        error = ModuleNotFoundError("No module named '_ctypes'", name='_ctypes')
        with patch('training.run_training.importlib.import_module', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'Python installation cannot import ctypes/_ctypes'):
                check_dependencies(False)

    def test_broken_pysam_shared_library_is_reported(self):
        def import_package(name):
            if name == 'pysam':
                raise ImportError('missing native library')
        with patch('training.run_training.importlib.import_module', side_effect=import_package):
            with self.assertRaisesRegex(RuntimeError, 'Cannot import pysam: missing native library'):
                check_dependencies(False)

    def test_bad_environment_stops_before_config_or_pipeline(self):
        with patch('training.run_training.check_dependencies', side_effect=RuntimeError('broken Python')), \
                patch('utils.config.load_config') as loader:
            with self.assertRaisesRegex(RuntimeError, 'broken Python'):
                run(stages=['sites'])
            loader.assert_not_called()

    def test_complete_merge_only_can_be_reused_without_overwriting_sites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            merge = root / 'features/input_sites'
            merge.mkdir(parents=True)
            (merge / 'run.json').write_text(json.dumps(dict(status='complete')))
            (merge / 'sites.tsv').write_text('original merge')
            sites = root / 'sites'
            sites.mkdir()
            (sites / 'run.json').write_text(json.dumps(dict(status='complete')))
            check_outputs(dict(output_dir=root, references={}), ['features'])
            self.assertEqual((merge / 'sites.tsv').read_text(), 'original merge')
            self.assertTrue((sites / 'run.json').is_file())
            # An empty samples directory is safe; incomplete raw outputs are not.
            (root / 'features/samples').mkdir()
            check_outputs(dict(output_dir=root, references={}), ['features'])
            sample = root / 'features/samples/WT'
            sample.mkdir()
            (sample / 'INCOMPLETE.txt').write_text('still writing')
            with self.assertRaises(ValueError):
                check_outputs(dict(output_dir=root, references={}), ['features'])
            (sample / 'INCOMPLETE.txt').unlink()
            (sample / 'run.json').write_text(json.dumps(dict(status='complete')))
            check_outputs(dict(output_dir=root, references={}), ['features'])
            (root / 'features/index').mkdir()
            with self.assertRaises(FileExistsError):
                check_outputs(dict(output_dir=root, references={}), ['features'])

    def test_incomplete_merge_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            merge = root / 'features/input_sites'
            merge.mkdir(parents=True)
            (merge / 'INCOMPLETE.txt').write_text('incomplete')
            with self.assertRaises(ValueError):
                check_outputs(dict(output_dir=root, references={}), ['features'])


if __name__ == '__main__':
    unittest.main()
