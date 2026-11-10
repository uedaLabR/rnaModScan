import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import test_call_summary as support
from rnamodscan.analysis.call_summary import summarize_calls
from rnamodscan.analysis.overlap_plots import generate_venn_plots, read_partitions


class PlotValidationTests(unittest.TestCase):
    def write_table(self, path, counts):
        row = dict(sample_id='sample', cell='HEK293T', reference_cell='HEK293T', condition='WT',
                   threshold='test', min_depth=10, min_modified=3, min_af=.1,
                   modification='m6A', motif_group='all', reference_status='ready',
                   call_sites=4, short_read_only=1, database_only=1, both=1, neither=1)
        row.update(counts)
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row), delimiter='\t')
            writer.writeheader()
            writer.writerow(row)

    def test_rejects_inconsistent_or_negative_partitions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'table.tsv'
            for counts in ({'both': 2}, {'both': -1, 'neither': 3}, {'call_sites': -1}):
                self.write_table(path, counts)
                with self.assertRaises(ValueError):
                    read_partitions(path)

    def test_incomplete_analysis_is_not_plotted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'INCOMPLETE.txt').write_text('incomplete')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                generate_venn_plots(root)
            self.assertFalse((root / 'venn').exists())


@unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib not installed')
class PlotIntegrationTests(unittest.TestCase):
    def test_four_conditions_strata_zero_calls_missing_refs_and_exports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fa, catalog, original, thresholds = support.CallSummaryTests().fixture(root)
            config = json.loads(catalog.read_text())
            config['sources'].append(dict(config['sources'][2], id='DB2', modification='m5C'))
            catalog.write_text(json.dumps(config))
            samples = []
            for cell in ('HeLa', 'HEK293T'):
                for condition in ('WT', 'IVT'):
                    samples.append(dict(original[0], sample_id=cell + '_' + condition,
                                        cell=cell, reference_cell=cell, condition=condition))
            out = root / 'summary'
            summarize_calls(samples, thresholds, fa, catalog, out, write_called_sites=False,
                            progress_every=0)
            # Prove that replotting uses neither the pileup nor FASTA nor reference BED.
            Path(original[0]['bedmethyl']).unlink()
            fa.unlink()
            (root / 'ref.bed').unlink()
            plots = generate_venn_plots(out, formats=('svg',))
            self.assertEqual(plots['generated'], 32)  # 4 samples x 2 thresholds x 4 m6A strata
            self.assertEqual(plots['skipped'], 24)  # absent/partial references are not zero
            all_rows = [p for p in plots['plots'] if p['status'] == 'complete'
                        and p['threshold'] == 'standard' and p['motif_group'] == 'all']
            self.assertEqual(len(all_rows), 4)
            for row in all_rows:
                self.assertEqual(row['counts'], dict(short_read_only=0, database_only=0, both=1, neither=2))
                self.assertEqual(row['call_sites'], 3)
            strict = next(p for p in plots['plots'] if p['status'] == 'complete'
                          and p['threshold'] == 'strict')
            self.assertEqual(strict['call_sites'], 0)
            text = (out / 'venn' / strict['files'][0]).read_text()
            self.assertIn('No calls pass this threshold', text)
            self.assertIn('not proportional', text)
            self.assertFalse((out / 'venn/INCOMPLETE.txt').exists())
            with self.assertRaises(FileExistsError):
                generate_venn_plots(out, formats=('svg',))

    def test_png_svg_pdf_and_four_exclusive_regions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'run.json').write_text('{"status": "complete", "references": []}')
            PlotValidationTests().write_table(root / 'reference_overlap_summary.tsv', {})
            plots = generate_venn_plots(root, dpi=100)
            result = plots['plots'][0]
            self.assertEqual(result['counts'], dict(short_read_only=1, database_only=1, both=1, neither=1))
            paths = {Path(name).suffix: root / 'venn' / name for name in result['files']}
            self.assertTrue(paths['.png'].read_bytes().startswith(b'\x89PNG\r\n'))
            self.assertTrue(paths['.pdf'].read_bytes().startswith(b'%PDF'))
            svg = paths['.svg'].read_text()
            self.assertIn('Neither reference: 1 (25.0%)', svg)
            self.assertIn('Dorado/modkit called sites: 4', svg)
