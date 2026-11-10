import gzip
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils.io.bed import bedmethyl_rows, load_bed_sites, parse_bedmethyl_row, passes_thresholds
from rnamodscan.analysis.call_summary import bedmethyl_rows as analysis_rows
from utils.pileup.candidates import parse_row
from utils.references.catalog import load_bed_sites as catalog_sites
from utils.references.catalog import ReferenceCatalog


class SharedBedTest(unittest.TestCase):
    def test_shared_reader_identity_and_all_mod_codes(self):
        self.assertIs(analysis_rows, bedmethyl_rows)
        self.assertIs(parse_row, parse_bedmethyl_row)
        self.assertIs(catalog_sites, load_bed_sites)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'calls.bed.gz'
            codes = ['a', '17596', 'm', '17802', '69426', '19228', '19229', '19227']
            with gzip.open(path, 'wt', encoding='utf-8-sig') as handle:
                handle.write('# modkit bedMethyl\n')
                for code in codes:
                    handle.write(f'chr1 10 11 {code} 10 + 10 11 0,0,0 10 30 3 7 0 100 10 5 5\n')
            rows = list(bedmethyl_rows(path))
            self.assertEqual([r['mod_code'] for r in rows], codes)
            self.assertEqual(rows[0]['af'], 0.1)
            self.assertTrue(passes_thresholds(rows[0], 10, 3, .1))
            self.assertFalse(passes_thresholds(rows[0], 10, 3, .11))
            # The standalone paper-analysis entrypoint uses the same reader and
            # keeps the four target site sets/DRACH behavior after refactoring.
            script = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'analysis/pileup_overlap/run_pairwise_overlap.py'))
            self.assertIs(script['parse_bedmethyl_row'], parse_bedmethyl_row)
            result = script['read_drs_calls'](path, SimpleNamespace(is_drach=lambda *site: True))
            self.assertEqual(result['input_rows'], 8)
            self.assertEqual(result['supported_rows'], 4)
            self.assertTrue(all(sites == {('chr1', 10, '+')} for sites in result['all'].values()))
            self.assertEqual(result['m6A_DRACH'], {('chr1', 10, '+')})

    def test_invalid_order_duplicates_and_bedrmod(self):
        row = lambda p: f'chr1 {p} {p+1} a 10 + {p} {p+1} 0,0,0 10 30 3 7 0 0 0 0 0\n'
        for text in (row(10) + row(10), row(10) + row(9), '# bedRMod\n' + row(10)):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'calls.bed'
                path.write_text(text)
                with self.assertRaises(ValueError):
                    list(bedmethyl_rows(path))

    def test_reference_bed_headers_exact_strand_and_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sites.bgz'
            with gzip.open(path, 'wt') as handle:
                handle.write('track name=sites\nchr1\t0\t1\tx\t0\t-\nchr1\t0\t1\tx\t0\t-\n')
            self.assertEqual(load_bed_sites(path), {('chr1', 0, '-')})

    def test_catalog_hash_lookup_preserves_source_order_and_exact_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'sites.bed').write_text('chr1\t10\t11\tx\t0\t+\n')
            sources = []
            for name, kind, cell, mod, status in [
                ('database', 'database', '*', 'm6A', 'ready'),
                ('hek', 'short_read', 'HEK293T', 'm6A', 'ready'),
                ('unavailable', 'short_read', 'HeLa', 'm6A', 'unavailable'),
                ('hela', 'short_read', 'HeLa', 'm6A', 'ready'),
                ('other_mod', 'database', '*', 'm5C', 'ready'),
            ]:
                sources.append(dict(id=name, kind=kind, cell=cell, modification=mod, status=status,
                                    assembly='GRCh38', coordinates='BED0', path='sites.bed'))
            manifest = root / 'catalog.json'
            manifest.write_text(json.dumps(dict(assembly='GRCh38', sources=sources)))
            catalog = ReferenceCatalog(manifest)
            for _ in range(2):  # Repeated queries share the source scope, with the same result.
                self.assertEqual(catalog.matches('HeLa', 'm6A', 'chr1', 10, '+'), ['database', 'hela'])
                self.assertEqual(catalog.matches('HEK293T', 'm6A', 'chr1', 10, '+'), ['database', 'hek'])
                self.assertEqual(catalog.matches('HEK293', 'm6A', 'chr1', 10, '+'), ['database'])
                self.assertEqual(catalog.matches('HeLa', 'm6A', 'chr1', 10, '-'), [])
                self.assertEqual(catalog.matches('HeLa', 'm6A', 'chr1', 11, '+'), [])
                self.assertEqual(catalog.matches('HeLa', 'm5C', 'chr1', 10, '+'), ['other_mod'])
            applicable = catalog.applicable('HeLa', 'm6A')
            self.assertEqual([s['id'] for s in applicable], ['database', 'unavailable', 'hela'])
            applicable.clear()
            self.assertEqual(len(catalog.applicable('HeLa', 'm6A')), 3)


if __name__ == '__main__':
    unittest.main()
