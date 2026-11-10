"""Species policies prevent cross-assembly, short-read and disabled DB rescue."""
import csv
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import yaml

from prediction.pipeline import run_prediction_pipeline
from prediction.reference_catalog import checked_catalog_path
from utils.config import load_config
from utils.references.catalog import ReferenceCatalog
from utils.references.rmbase import ARCHIVES, import_rmbase3

ROOT = Path(__file__).resolve().parents[1]


class PredictionSpeciesTest(unittest.TestCase):
    def test_genome_defaults_overrides_and_disabled_catalog(self):
        config_path = ROOT / 'prediction/config.yaml'
        human = load_config(config_path, 'predict')
        mouse = load_config(config_path, 'predict', genome='mm10')
        other = load_config(config_path, 'predict', genome='other')
        self.assertEqual(human['references']['catalog'].name, 'catalog.json')
        self.assertEqual(mouse['references']['catalog'].name, 'catalog.mm10.json')
        self.assertIsNone(other['references']['catalog'])
        self.assertIsNone(checked_catalog_path(other))
        with self.assertRaisesRegex(ValueError, 'genome must be'):
            load_config(config_path, 'predict', genome='mm39')

    def test_mouse_ignores_short_read_and_other_db_without_opening_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'known.bed').write_text('chr1\t10\t11\tknown\t0\t-\n')
            sources = []
            for name, kind, database, version, path in [
                    ('rmbase', 'database', 'RMBase', '3.0', 'known.bed'),
                    ('short', 'short_read', None, None, 'absent_short_read.bed'),
                    ('other', 'database', 'another_db', '1', 'absent_other_db.bed')]:
                sources.append(dict(id=name, kind=kind, database=database, version=version,
                    modification='m6A', cell='mouse' if kind == 'short_read' else '*',
                    assembly='mm10', coordinates='BED0', path=path))
            manifest = root / 'catalog.json'
            manifest.write_text(json.dumps(dict(assembly='mm10', sources=sources)))
            catalog = ReferenceCatalog(manifest)
            self.assertEqual([source['id'] for source in catalog.sources], ['rmbase'])
            self.assertEqual(catalog.matches('mouse', 'm6A', 'chr1', 10, '-'), ['rmbase'])
            self.assertEqual(catalog.matches('mouse', 'm6A', 'chr1', 10, '+'), [])
            config = dict(genome='hg38', references=dict(catalog=manifest))
            with self.assertRaisesRegex(ValueError, 'requires a GRCh38'):
                checked_catalog_path(config)
        self.assertEqual(ReferenceCatalog(None).matches('mouse', 'm6A', 'chr1', 10, '-'), [])

    def test_mouse_25_columns_only_a_to_i_and_duplicate_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for modification, (name, code) in ARCHIVES.items():
                def line(kind):
                    return '\t'.join(['chr1', '10', '11', 'site', '0', '-', kind] + ['na'] * 18) + '\n'
                data = (line(code) * 2 + (line('C-U') if modification == 'inosine' else '')).encode()
                with tarfile.open(root / f'mm10.{name}.tar.gz', 'w:gz') as archive:
                    member = tarfile.TarInfo(f'mouse.mm10.{name}.allSite.uniqID.anno.score.conserved.guide.resort.col25.bed')
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
            report = import_rmbase3(root, root / 'beds', genome='mm10')
            self.assertEqual((report['species'], report['assembly']), ('Mus musculus', 'mm10'))
            self.assertTrue(all(item['unique_sites'] == item['duplicate_rows'] == 1 for item in report['files']))
            self.assertEqual((root / 'beds/inosine.mm10.bed').read_text(), 'chr1\t10\t11\tsite\t0\t-\n')

    def test_reference_only_workflow_respects_all_three_genome_policies(self):
        for genome, expected in [('hg38', [('1', '0'), ('0', '1')]),
                                 ('mm10', [('1', '0')]), ('other', [])]:
            with self.subTest(genome=genome), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'genome.fa').write_text('>chr1\n' + 'a' * 100 + '\n')
                (root / 'genome.fa.fai').write_text('chr1\t100\t6\t100\t101\n')
                (root / 'calls.bed').write_text(''.join(
                    f'chr1 {pos} {pos+1} m 10 + {pos} {pos+1} 0,0,0 10 30 3 7 0 0 0 0 0\n'
                    for pos in (10, 20)))
                (root / 'pileup.log').write_text('command line: fixture\n')
                assembly = 'mm10' if genome == 'mm10' else 'GRCh38'
                sources = []
                for name, kind, pos in [('db', 'database', 10), ('short', 'short_read', 20)]:
                    (root / f'{name}.bed').write_text(f'chr1\t{pos}\t{pos+1}\tknown\t0\t+\n')
                    sources.append(dict(id=name, kind=kind, database='RMBase', version='3.0',
                        cell='*' if kind == 'database' else 'sample', modification='m5C',
                        assembly=assembly, coordinates='BED0', path=f'{name}.bed'))
                (root / 'catalog.json').write_text(json.dumps(dict(assembly=assembly, sources=sources)))
                values = dict(mode='predict', genome=genome, output_dir='output', targets=['m5C'],
                    references=dict(fasta='genome.fa', catalog='catalog.json'),
                    sample=dict(sample_id='sample', cell='sample', reference_cell='sample', condition='WT',
                                bedmethyl='calls.bed', pileup_log='pileup.log'))
                path = root / 'config.yaml'
                path.write_text(yaml.safe_dump(values))
                config = load_config(path, 'predict')
                if genome == 'other':
                    (root / 'catalog.json').unlink()
                    (root / 'db.bed').unlink()
                    (root / 'short.bed').unlink()
                run_prediction_pipeline(config)
                with (root / 'output/predictions/m5C/kept_sites.tsv').open() as handle:
                    rows = list(csv.DictReader(handle, delimiter='\t'))
                self.assertEqual([(row['flag_db'], row['flag_short_read']) for row in rows], expected)

    def test_softmasked_fasta_is_required_before_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = dict(mode='predict', genome='other', output_dir='output', targets=['m5C'],
                sample=dict(sample_id='sample', cell='sample', reference_cell='sample', condition='WT',
                            bedmethyl='calls.bed', pileup_log='pileup.log'))
            (root / 'calls.bed').touch()
            (root / 'pileup.log').touch()
            path = root / 'config.yaml'
            path.write_text(yaml.safe_dump(values))
            with self.assertRaisesRegex(ValueError, 'references.fasta'):
                run_prediction_pipeline(load_config(path, 'predict'))
            self.assertFalse((root / 'output').exists())
