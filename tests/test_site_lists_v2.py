import csv
import json
import tempfile
import unittest
from pathlib import Path

from training.site_lists import build_site_lists, ivt_override


def pileup(pos, code='a', modified=3, depth=10, strand='+'):
    return f'chr1 {pos} {pos+1} {code} {depth} {strand} {pos} {pos+1} 0,0,0 {depth} 0 {modified} {depth-modified} 0 0 0 0 0\n'


class SiteListsV2Test(unittest.TestCase):
    def setup_input(self, root, ivt=True):
        bed = root / 'truth.bed'
        bed.write_text(''.join(f'chr1\t{p}\t{p+1}\tx\t0\t+\n' for p in (10, 100, 200, 300, 400, 500)))
        sr = root / 'sr.bed'
        sr.write_text('chr1\t100\t101\tx\t0\t+\nchr1\t600\t601\tx\t0\t-\n')
        catalog = root / 'catalog.json'
        catalog.write_text(json.dumps(dict(assembly='GRCh38', sources=[
            dict(id='RMBase3.m6A', kind='database', database='RMBase', version='3.0', cell='*',
                 modification='m6A', assembly='GRCh38', coordinates='BED0', path=bed.name),
            dict(id='HeLa.GLORI', kind='short_read', cell='HeLa', modification='m6A',
                 assembly='GRCh38', coordinates='BED0', path=sr.name)])))
        wt = root / 'wt.bed'
        wt.write_text(pileup(10, modified=5) + pileup(100, modified=5) + pileup(200, modified=5)
                      + pileup(300, modified=5) + pileup(400, modified=5) + pileup(500, modified=5)
                      + pileup(600, strand='-') + pileup(700) + pileup(800, code='69426'))
        other = root / 'ivt.bed'
        other.write_text(pileup(10, modified=3) + pileup(100, modified=3, depth=20)
                         + pileup(200, modified=2, depth=4) + pileup(300, modified=3, depth=11)
                         + pileup(500, modified=0, depth=0) + pileup(900))
        log = root / 'pileup.log'
        log.write_text('command line: modkit pileup test\nsetting threshold 0.7 (qual: 179) for base A, percentile 0.1\nDone, processed 9 rows\n')
        pair = dict(cell='HeLa', reference_cell='HeLa', wt_sample='HeLa_WT', ivt_sample='HeLa_IVT',
                    wt_bedmethyl=wt, wt_pileup_log=log,
                    ivt_bedmethyl=other if ivt else None, ivt_pileup_log=log if ivt else None)
        return catalog, pair

    def read_rows(self, path):
        with path.open() as handle:
            return list(csv.DictReader(handle, delimiter='\t'))

    def test_labels_inclusive_override_provenance_and_grouping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, pair = self.setup_input(root)
            report = build_site_lists([pair], catalog, root / 'out', progress_every=0)
            rows = self.read_rows(root / 'out/all_sites.tsv')
            keyed = {(r['condition'], int(r['start'])): r for r in rows}
            self.assertEqual(keyed['WT', 10]['label'], '0')  # exactly 60%, 3 modified reads
            self.assertEqual(keyed['WT', 100]['label_reason'], 'short_read_and_database_overlap')
            self.assertEqual(keyed['WT', 200]['label'], '1')  # high AF but only 2 modified
            self.assertEqual(keyed['WT', 300]['label'], '1')  # just below 60%
            self.assertEqual(keyed['WT', 400]['ivt_comparison'], 'missing_locus')
            self.assertEqual(keyed['WT', 400]['label'], '1')
            self.assertEqual(keyed['WT', 500]['ivt_comparison'], 'zero_depth')
            self.assertEqual(keyed['WT', 500]['paired_ivt_af'], '')
            self.assertEqual(keyed['WT', 600]['label'], '1')
            self.assertEqual(keyed['WT', 700]['label'], '')
            self.assertNotIn(('WT', 800), keyed)  # 8 input mods but only 4 targets
            self.assertEqual(keyed['IVT', 100]['label'], '0')
            self.assertEqual(keyed['IVT', 10]['group_id'], keyed['WT', 10]['group_id'])
            self.assertEqual(keyed['IVT', 900]['label'], '0')
            self.assertEqual(report['inputs'][0]['pileup_log']['filter_mode'], 'sampled_per_base')
            self.assertTrue(all(r['label'] != '' for r in self.read_rows(root / 'out/m6A/training_sites.tsv')))
            summaries = self.read_rows(root / 'out/summary/sample_modification_counts.tsv')
            s = next(r for r in summaries if r['sample_id'] == 'HeLa_WT' and r['modification'] == 'm6A')
            self.assertEqual(s['total_sites'], '8')
            self.assertEqual(s['positive'], '6')
            self.assertEqual(s['negative'], '1')
            self.assertEqual(s['unlabeled'], '1')
            self.assertEqual(s['overridden_overlap_sites'], '1')
            hist = self.read_rows(root / 'out/summary/depth_af_histograms.tsv')
            self.assertEqual(sum(int(r['sites']) for r in hist if r['sample_id'] == 'HeLa_WT' and r['variable'] == 'af'), 8)
            with self.assertRaises(FileExistsError):
                build_site_lists([pair], catalog, root / 'out')

    def test_unfinished_ivt_is_pending_not_negative_or_final_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, pair = self.setup_input(root, ivt=False)
            # Wrong-cell short-read evidence must not be used, but RMBase still applies.
            pair.update(cell='HEK293T', reference_cell='HEK293T')
            build_site_lists([pair], catalog, root / 'out', progress_every=0)
            rows = self.read_rows(root / 'out/m6A/pending_sites.tsv')
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(r['label'] == '' and r['label_status'] == 'pending' for r in rows))
            by_pos = {int(r['start']): r for r in rows}
            self.assertEqual(by_pos[100]['short_read_sources'], '')
            self.assertEqual(by_pos[100]['database_sources'], 'RMBase3.m6A')
            self.assertEqual(by_pos[600]['positive_evidence'], '0')
            self.assertEqual(self.read_rows(root / 'out/m6A/training_sites.tsv'), [])
            summaries = self.read_rows(root / 'out/summary/sample_modification_counts.tsv')
            ivt = next(r for r in summaries if r['sample_id'] == 'HeLa_IVT' and r['modification'] == 'm6A')
            self.assertEqual(ivt['input_status'], 'not_available')
            self.assertEqual(ivt['negative'], '')  # not available is not zero

    def test_three_reads_means_modified_not_total_and_below_depth_can_override(self):
        wt = dict(modified_reads=8, total_reads=10)
        self.assertFalse(ivt_override(wt, dict(modified_reads=2, total_reads=3)))
        self.assertTrue(ivt_override(wt, dict(modified_reads=3, total_reads=5)))
        self.assertFalse(ivt_override(wt, None))


if __name__ == '__main__':
    unittest.main()
