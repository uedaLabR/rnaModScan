"""Assembly provenance for new runs without fitting any neural network."""
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.site_bag_fixture import prepare_training_bags
from utils.features.site_bags import build_site_bag_index


class TrainingAssemblyTest(unittest.TestCase):
    def feature_directories(self, root, assemblies):
        prepare_training_bags(root)
        directories = sorted((root / 'features').iterdir())
        for directory, assembly in zip(directories, assemblies):
            path = directory / 'run.json'
            report = json.loads(path.read_text())
            report.update(assembly=assembly, coordinates='BED0')
            path.write_text(json.dumps(report))
        return directories

    def test_new_index_preserves_declared_assembly_and_legacy_index_stays_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = self.feature_directories(root, ['GRCz11'] * 4)
            self.assertNotIn('assembly', json.loads((root / 'index/run.json').read_text()))
            run = build_site_bag_index(directories, root / 'species_index')
            self.assertEqual(run['assembly'], 'GRCz11')
            self.assertEqual(run['coordinates'], 'BED0')
            self.assertEqual(run['counts']['indexed_sites'], 48)

    def test_mixed_assemblies_cannot_enter_one_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = self.feature_directories(root, ['mm39', 'mm39', 'GRCz11', 'GRCz11'])
            with self.assertRaisesRegex(ValueError, 'mix feature assemblies'):
                build_site_bag_index(directories, root / 'mixed_index')
            self.assertTrue((root / 'mixed_index/FAILED.txt').exists())

    def test_export_carries_assembly_without_training_or_requiring_new_model_inputs(self):
        from training.final_model import export_final_model
        from utils.nn.keras_site import ModelConfig

        class MetadataOnlyModel:
            def save(self, path):
                Path(path).write_bytes(b'not-a-real-model; metadata test only')

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / 'index/sites.tsv'
            index.parent.mkdir()
            index.write_text('fixture index\n')
            (index.parent / 'run.json').write_text(json.dumps(dict(
                status='complete', feature_sources=[], assembly='mm39', coordinates='BED0')))
            dataset = SimpleNamespace(target='adenine', rows=[dict(row_id='fixture', sample_id='WT', label='0')])
            config = ModelConfig(target='adenine')
            # Only export metadata is exercised; no TensorFlow fit/load/inference.
            with patch.dict('sys.modules', {'tensorflow': SimpleNamespace(__version__='metadata-test')}):
                run = export_final_model(MetadataOnlyModel(), dataset, config, index, root / 'final',
                                         epochs=1, class_weights={0: 1.0}, seed=42, learning_rate=0.0003)
            self.assertEqual(run['assembly'], 'mm39')
            self.assertEqual(run['coordinates'], 'BED0')
            self.assertEqual(run['model'], asdict(config))
            self.assertEqual(run['input_schema']['class_names'], ['m6A', 'inosine', 'false_positive'])
            self.assertFalse((root / 'final/metrics.json').exists())


if __name__ == '__main__':
    unittest.main()
