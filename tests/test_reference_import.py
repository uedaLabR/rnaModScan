"""Positive-reference adapters keep coordinates, provenance and species explicit."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import yaml

from training.site_lists import build_site_lists
from utils.references.catalog import ReferenceCatalog, sha256
from utils.references.cli import main
from utils.references.import_catalog import build_catalog, check_catalog


class ReferenceImportTest(unittest.TestCase):
    def manifest(self, root, *, assembly='GRCz11', species='Danio rerio'):
        (root/'positive.tsv').write_text('chromosome\tposition\tstrand\nchr1\t1\t+\nchr1\t1\t+\nchr1\t1\t-\n')
        (root/'editing.csv').write_text('contig,first,last,direction\nchr2,11,11,-\n')
        config = dict(schema_version=1, assembly=assembly, species=species, sources=[
            dict(id='tissue.GLORI', kind='short_read', cell='tissue', modification='m6A',
                 path='positive.tsv', format='tsv', coordinates='one_based_position',
                 columns=dict(chrom='chromosome', position='position', strand='strand'),
                 citation='Author-selected GLORI positive sites', source_url='https://example.org/study'),
            dict(id='custom:database', kind='database', cell='*', database='curated editing', version='1',
                 modification='inosine', path='editing.csv', format='csv', coordinates='one_based_closed',
                 columns=dict(chrom='contig', start='first', end='last', strand='direction'),
                 citation='Curated positive editing sites'),
        ])
        path=root/'sources.yaml'; path.write_text(yaml.safe_dump(config))
        return path, config

    def test_explicit_conversion_dedup_source_scope_and_portable_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,config=self.manifest(root)
            original_hash=sha256(root/'positive.tsv')
            report=build_catalog(manifest,root/'references')
            self.assertEqual(report['assembly'],'GRCz11')
            self.assertEqual(report['sources'][0]['input_rows'],3)
            self.assertEqual(report['sources'][0]['unique_sites'],2)
            self.assertEqual(report['sources'][0]['duplicate_rows'],1)
            catalog=ReferenceCatalog(root/'references/catalog.json',for_training=True)
            self.assertEqual(catalog.matches('tissue','m6A','chr1',0,'+'),['tissue.GLORI'])
            self.assertEqual(catalog.matches('different_tissue','m6A','chr1',0,'+'),[])
            self.assertEqual(catalog.matches('any','inosine','chr2',10,'-'),['custom:database'])
            entry=json.loads((root/'references/catalog.json').read_text())['sources'][0]
            self.assertFalse(Path(entry['path']).is_absolute())
            provenance=json.loads((root/'references'/entry['provenance']).read_text())
            self.assertEqual(provenance['input_sha256'],original_hash)
            self.assertEqual(provenance['source']['citation'],config['sources'][0]['citation'])
            self.assertEqual(provenance['source']['columns'],config['sources'][0]['columns'])
            self.assertEqual(provenance['normalized_coordinates'],'BED0')
            self.assertIn('unknown, not negative',provenance['policy'])
            self.assertIn('negative training labels require separate controls',report['label_policy'])
            self.assertEqual(sha256(root/'positive.tsv'),original_hash)
            moved=root/'relocated';shutil.copytree(root/'references',moved)
            (root/'positive.tsv').unlink();(root/'editing.csv').unlink()
            self.assertEqual(check_catalog(moved/'catalog.json',for_training=True)['sources'][0]['unique_sites'],2)
            with self.assertRaisesRegex(ValueError,'GRCh38 or mm10'):
                ReferenceCatalog(moved/'catalog.json')

    def test_mm10_training_experiments_do_not_change_prediction_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,config=self.manifest(root,assembly='mm10',species='Mus musculus')
            config['sources']=config['sources'][:1]
            manifest.write_text(yaml.safe_dump(config))
            build_catalog(manifest,root/'references')
            path=root/'references/catalog.json'
            self.assertEqual(ReferenceCatalog(path,for_training=True).matches('tissue','m6A','chr1',0,'+'),['tissue.GLORI'])
            # Default mouse prediction still skips non-RMBase sources entirely.
            self.assertEqual(ReferenceCatalog(path).sources,[])

    def test_bed_defaults_preserve_strand_and_verify_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,config=self.manifest(root,assembly='GRCh38',species='Homo sapiens')
            (root/'sites.bed').write_text('chr1\t0\t1\tpositive\t0\t-\n')
            source=config['sources'][0]
            source.update(path='sites.bed',format='bed',coordinates='BED0');source.pop('columns')
            config['sources']=[source];manifest.write_text(yaml.safe_dump(config))
            build_catalog(manifest,root/'references')
            catalog=ReferenceCatalog(root/'references/catalog.json')
            self.assertEqual(catalog.matches('tissue','m6A','chr1',0,'-'),['tissue.GLORI'])
            path=Path(catalog.sources[0]['path']);path.write_text('chr1\t1\t2\tpositive\t0\t-\n')
            with self.assertRaisesRegex(ValueError,'checksum'):
                check_catalog(root/'references/catalog.json')

    def test_invalid_coordinates_strands_and_mapping_leave_destination_absent(self):
        cases=[
            ('chromosome\tposition\tstrand\nchr1\t0\t+\n',None),
            ('chromosome\tposition\tstrand\nchr1\t1\t.\n',None),
            ('chromosome\tposition\tstrand\nchr1\t1.0\t+\n',None),
            ('chromosome\tposition\nchr1\t1\n',None),
            ('chromosome\tposition\tstrand\nchr1\t1\t+\n',dict(chrom='chromosome',position='position')),
        ]
        for text,columns in cases:
            with self.subTest(text=text,columns=columns),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);manifest,config=self.manifest(root)
                (root/'positive.tsv').write_text(text)
                if columns is not None:
                    config['sources'][0]['columns']=columns;manifest.write_text(yaml.safe_dump(config))
                with self.assertRaises(ValueError):build_catalog(manifest,root/'references')
                self.assertFalse((root/'references').exists())
                self.assertEqual((root/'positive.tsv').read_text(),text)

    def test_rejects_mixed_assemblies_species_and_duplicate_ids(self):
        for change in ['assembly','species','duplicate_id']:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);manifest,config=self.manifest(root)
                source=config['sources'][1]
                if change=='assembly':source['assembly']='GRCh38'
                elif change=='species':source['species']='Homo sapiens'
                else:source['id']=config['sources'][0]['id']
                manifest.write_text(yaml.safe_dump(config))
                with self.assertRaises(ValueError):build_catalog(manifest,root/'references')
                self.assertFalse((root/'references').exists())

    def test_existing_empty_or_populated_output_is_never_overwritten(self):
        for populated in [False,True]:
            with self.subTest(populated=populated),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);manifest,_=self.manifest(root)
                output=root/'references';output.mkdir()
                if populated:(output/'marker').write_text('user data')
                with self.assertRaises(FileExistsError):build_catalog(manifest,output)
                self.assertEqual([p.name for p in output.iterdir()],['marker'] if populated else [])
                if populated:self.assertEqual((output/'marker').read_text(),'user data')

    def test_cli_build_and_explicit_training_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,_=self.manifest(root)
            output=root/'references';capture=io.StringIO()
            with redirect_stdout(capture):
                self.assertEqual(main(['build','--manifest',str(manifest),'--output',str(output)]),0)
            self.assertEqual(json.loads(capture.getvalue())['status'],'complete')
            capture=io.StringIO()
            with redirect_stdout(capture):
                self.assertEqual(main(['check','--catalog',str(output/'catalog.json'),'--training']),0)
            self.assertEqual(json.loads(capture.getvalue())['purpose'],'training')
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                main(['check','--catalog',str(output/'catalog.json')])
            self.assertEqual(error.exception.code,2)

    def test_custom_training_report_and_labels_use_declared_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,config=self.manifest(root)
            config['sources']=config['sources'][:1];manifest.write_text(yaml.safe_dump(config))
            build_catalog(manifest,root/'references')
            def pileup(modified):return f'chr1 0 1 a 10 + 0 1 0,0,0 10 0 {modified} {10-modified} 0 0 0 0 0\n'
            (root/'wt.bed').write_text(pileup(5));(root/'ivt.bed').write_text(pileup(0))
            (root/'pileup.log').write_text('command line: fixture\n')
            pair=dict(cell='tissue',reference_cell='tissue',wt_sample='wt',ivt_sample='ivt',
                      wt_bedmethyl=root/'wt.bed',wt_pileup_log=root/'pileup.log',
                      ivt_bedmethyl=root/'ivt.bed',ivt_pileup_log=root/'pileup.log')
            report=build_site_lists([pair],root/'references/catalog.json',root/'sites',progress_every=0)
            self.assertEqual(report['assembly'],'GRCz11')
            import csv
            with (root/'sites/all_sites.tsv').open() as handle:rows=list(csv.DictReader(handle,delimiter='\t'))
            self.assertEqual({row['condition']:row['label'] for row in rows},{'WT':'1','IVT':'0'})


if __name__=='__main__':unittest.main()
