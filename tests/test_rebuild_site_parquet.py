"""The fresh BAM entry preserves the selection and never starts NN fitting."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import yaml

from training.rebuild_site_parquet import rebuild_site_parquet, start_background


class RebuildSiteParquetTest(unittest.TestCase):
    def config(self, root):
        samples = []
        for condition in ('WT', 'IVT'):
            bam = root / f'{condition}.bam'
            bam.touch()
            Path(str(bam)+'.bai').touch()
            samples.append(dict(sample_id=condition, cell='HEK293T', reference_cell='HEK293T',
                                condition=condition, bam=str(bam)))
        fasta = root / 'reference.fa'
        fasta.touch()
        Path(str(fasta)+'.fai').touch()
        config = dict(schema_version=1, mode='train', output_dir='out', targets=['m6A', 'inosine'],
                      samples=samples, references=dict(fasta=str(fasta)),
                      nn=dict(task='adenine_three_class', max_reads=300),
                      features=dict(sample_workers=2))
        path = root / 'config.yaml'
        path.write_text(yaml.safe_dump(config))
        selected = root / 'selection/adenine/training_sites.tsv'
        selected.parent.mkdir(parents=True)
        selected.write_text('unchanged selected table\n')
        source_report = root / 'source.json'
        source_report.write_text('{}')
        return path, selected, source_report

    def test_dry_run_has_no_extraction_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, selected, _ = self.config(root)
            result = rebuild_site_parquet(config, selected.parent.parent, dry_run=True)
            self.assertFalse((root / 'out').exists())
            self.assertFalse(result['neural_fitting'])
            self.assertEqual(result['max_reads'], 300)
            self.assertFalse(result['disk_padding'])
            self.assertEqual(result['candidate_group_gap'], 10000)
            self.assertEqual(result['read_batch_size'], 256)

    def test_stages_reextract_bam_and_create_manifests_but_do_not_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, selected, source = self.config(root)
            with patch('training.rebuild_site_parquet.verify_selected_sites', return_value=(selected, {}, {'WT': 3, 'IVT': 3}, {'m6A': 2, 'inosine': 1, 'false_positive': 3})), \
                    patch('training.rebuild_site_parquet.verify_source_report'), \
                    patch('utils.features.site_bag_pipeline.extract_site_bags_and_index') as extraction, \
                    patch('training.cross_validation.build_cross_validation') as split, \
                    patch('training.cell_datasets.build_cell_datasets') as cells:
                result = rebuild_site_parquet(config, selected.parent.parent, source_report=source)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['completed_stages'], ['features', 'split', 'cell_dataset'])
            self.assertFalse(result['neural_fitting'])
            self.assertEqual(extraction.call_args.kwargs['mode'], 'training')
            self.assertEqual(extraction.call_args.kwargs['max_reads'], 300)
            self.assertEqual(extraction.call_args.kwargs['sample_workers'], 2)
            self.assertEqual(extraction.call_args.kwargs['candidate_group_gap'], 10000)
            self.assertEqual(extraction.call_args.kwargs['read_batch_size'], 256)
            self.assertEqual(extraction.call_args.args[0], selected)
            split.assert_called_once()
            cells.assert_called_once()
            self.assertEqual(selected.read_text(), 'unchanged selected table\n')
            saved = json.loads((root / 'out/preparation_run.json').read_text())
            self.assertEqual(saved['schema'], 'site_bag_preparation_v1')

    def test_background_is_detached_and_second_start_does_not_overwrite_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, selected, source = self.config(root)
            with patch('training.rebuild_site_parquet.subprocess.Popen', return_value=types.SimpleNamespace(pid=123)) as spawn:
                result = start_background(config, selected.parent.parent, source)
            self.assertEqual(result['status'], 'starting')
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            self.assertNotIn('--background', spawn.call_args.args[0])
            with patch('training.rebuild_site_parquet.subprocess.Popen') as second:
                with self.assertRaises(FileExistsError):
                    start_background(config, selected.parent.parent, source)
                second.assert_not_called()
