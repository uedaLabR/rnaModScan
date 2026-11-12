"""Fresh training keeps declared assemblies and exports the ordinary model layout.

All inputs are tiny temporary fixtures. Neural fitting is stubbed; no research
data, model weights or completed run directories are opened or modified.
"""
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from training.pipeline import run_training_stage
from training.selection_validation import verify_source_report
from utils.features.position import GtfAnnotation, annotate_positions
from utils.features.site_bag_pipeline import extract_site_bag_sample
from utils.features import parallel_groups


def annotation(path, assembly):
    path.write_text(f'#!genome-build {assembly}\n'
                    'contig1\ttest\texon\t11\t30\t.\t+\t.\t'
                    'gene_id "g1"; transcript_id "t1";\n')
    return path


class FreshTrainingWorkflowTest(unittest.TestCase):
    def test_annotation_uses_explicit_custom_assembly_and_keeps_human_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gtf = annotation(root / 'custom.gtf', 'TAIR10')
            index = root / 'sites.tsv'
            index.write_text('chrom\tstart\tend\tstrand\ncontig1\t12\t13\t+\n')
            result = annotate_positions([index], gtf, root / 'annotated', assembly='TAIR10')
            run = json.loads((result.parent / 'run.json').read_text())
            self.assertEqual(run['assembly'], 'TAIR10')
            self.assertEqual(run['coordinates'], 'BED0')
            with result.open() as handle:
                row = next(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual((row['chrom'], row['start'], row['region_other_exon']), ('contig1', '12', '1'))
            with self.assertRaisesRegex(ValueError, 'GRCh38'):
                GtfAnnotation(gtf)
            annotation(gtf, 'GRCm39.p6')
            self.assertEqual(GtfAnnotation(gtf, assembly='mm39').assembly, 'mm39')
            with self.assertRaisesRegex(ValueError, 'TAIR10'):
                GtfAnnotation(gtf, assembly='TAIR10')

    def test_annotation_stage_passes_the_training_catalog_assembly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / 'catalog.json'
            catalog.write_text(json.dumps(dict(assembly='GRCz11', sources=[])))
            gtf = annotation(root / 'fish.gtf', 'GRCz11')
            index_dir = root / 'features/index/adenine'
            index_dir.mkdir(parents=True)
            (index_dir / 'run.json').write_text(json.dumps(dict(status='complete')))
            (index_dir / 'sites.tsv').write_text('chrom\tstart\tend\tstrand\ncontig1\t12\t13\t+\n')
            config = dict(output_dir=root, mode='train', targets=['m6A', 'inosine'],
                          references=dict(gtf=gtf, catalog=catalog), features={}, candidates={},
                          nn=dict(task='adenine_three_class'))
            report = run_training_stage(config, 'annotation')
            self.assertTrue(report['sites'].is_file())
            metadata = json.loads((root / 'annotation/run.json').read_text())
            self.assertEqual(metadata['assembly'], 'GRCz11')

    def test_feature_catalog_mode_reaches_serial_and_group_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sites = root / 'sites.tsv'
            sites.write_text('sample_id\trow_id\ncustom_WT\tsite1\n')
            bam, fasta, catalog = [root / name for name in ('sample.bam', 'genome.fa', 'catalog.json')]
            for path in [bam, fasta, catalog]:
                path.write_text('temporary fixture\n')
            sample = dict(sample_id='custom_WT', cell='custom', reference_cell='custom',
                          condition='WT', bam=bam)
            for mode, expected in [('training', True), ('inference', False)]:
                with patch('utils.features.site_bag_pipeline.ReferenceCatalog') as loader, \
                        patch('utils.features.site_bag_pipeline.extract_site_bag_rows', return_value={'status':'stubbed'}) as extract:
                    extract_site_bag_sample(sites, sample, fasta, root / mode, catalog, mode=mode)
                    loader.assert_called_once_with(catalog, for_training=expected)
                    self.assertIs(extract.call_args.args[5], loader.return_value)
                with patch('utils.features.parallel_groups.ReferenceCatalog') as loader:
                    parallel_groups.initialize_worker(catalog, mode)
                    loader.assert_called_once_with(catalog, for_training=expected)

    def test_source_validation_matches_declared_catalog_without_assuming_human(self):
        from tests.test_selection_validation import SelectionValidationTest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, source, _ = SelectionValidationTest().fixture(root)
            run = json.loads(source.read_text())
            run['assembly'] = 'TAIR10'
            source.write_text(json.dumps(run))
            catalog = root / 'catalog.json'
            catalog.write_text(json.dumps(dict(assembly='TAIR10', sources=[])))
            config['references'] = dict(catalog=catalog)
            self.assertEqual(verify_source_report(config, source)['assembly'], 'TAIR10')
            catalog.write_text(json.dumps(dict(assembly='GRCh38', sources=[])))
            with self.assertRaisesRegex(ValueError, 'assembly differs'):
                verify_source_report(config, source)

    def test_declared_feature_assembly_is_preserved_in_a_new_index(self):
        from tests.site_bag_fixture import prepare_training_bags
        from utils.features.site_bags import build_site_bag_index
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_training_bags(root)
            sources = sorted((root / 'features').iterdir())
            for directory in sources:
                path = directory / 'run.json'
                run = json.loads(path.read_text())
                run.update(assembly='TAIR10', coordinates='BED0')
                path.write_text(json.dumps(run))
            report = build_site_bag_index(sources, root / 'custom_index')
            self.assertEqual(report['assembly'], 'TAIR10')
            self.assertEqual(report['coordinates'], 'BED0')
            path = sources[-1] / 'run.json'
            run = json.loads(path.read_text());run['assembly'] = 'mm10'
            path.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, '[Aa]ssembl'):
                build_site_bag_index(sources, root / 'mixed_index')

    def test_missing_positive_class_stops_before_bam_feature_extraction(self):
        from tests.test_selection_validation import SelectionValidationTest
        from utils.references.catalog import sha256
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, selection = SelectionValidationTest().fixture(root)
            selected = selection / 'adenine/training_sites.tsv'
            with selected.open() as handle:
                reader = csv.DictReader(handle, delimiter='\t')
                fields = reader.fieldnames
                rows = [row for row in reader if row['label'] != '1']
            with selected.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fields, delimiter='\t')
                writer.writeheader(); writer.writerows(rows)
            run = json.loads((selection / 'run.json').read_text())
            run.update(selected_sites=len(rows), selected_sha256=sha256(selected))
            (selection / 'run.json').write_text(json.dumps(run))
            config.update(output_dir=root / 'output', references={})
            with patch('training.adenine_sites.select_adenine_sites', return_value=run), \
                    patch('utils.features.site_bag_pipeline.extract_site_bags_and_index') as extract:
                with self.assertRaisesRegex(ValueError, 'missing: inosine'):
                    run_training_stage(config, 'features')
                extract.assert_not_called()

    def test_five_fold_fit_and_final_export_use_matching_prediction_paths(self):
        """Exercise stage routing without importing or executing TensorFlow."""
        config = dict(output_dir=Path('/unused/new-training'), mode='train',
                      targets=['m6A','inosine'], references={}, features={}, candidates={},
                      cv=dict(seed=42), nn=dict(task='adenine_three_class', batch_sites=16,
                      max_reads=300, sampling_seed=42, folds=list(range(5)), epochs=20,
                      patience=3, learning_rate=.0003,
                      target_settings={'adenine':dict(threshold=.5,final_epochs=None)}))
        fit = ModuleType('training.fit');fit.train_fold = Mock(return_value={'status':'stubbed'})
        final = ModuleType('training.final_model');final.train_final_model = Mock(return_value={'status':'stubbed'})
        with patch.dict(sys.modules, {'training.fit':fit,'training.final_model':final}):
            run_training_stage(config, 'fit')
            run_training_stage(config, 'final')
        self.assertEqual(fit.train_fold.call_count,5)
        for fold,call in enumerate(fit.train_fold.call_args_list):
            self.assertEqual(call.args[1],config['output_dir']/f'splits/adenine/fold_{fold}')
            self.assertEqual(call.args[2],config['output_dir']/f'models/adenine/fold_{fold}')
        args = final.train_final_model.call_args
        self.assertEqual(args.args[1],config['output_dir']/'models/adenine')
        self.assertEqual(args.args[2],config['output_dir']/'final_model/adenine')
        self.assertIsNone(args.kwargs['epochs'])  # The final routine chooses CV median.


if __name__ == '__main__':
    unittest.main()
