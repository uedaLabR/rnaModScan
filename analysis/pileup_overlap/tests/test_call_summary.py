import csv
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from rnamodscan.analysis.call_summary import summarize_calls, bedmethyl_rows
from utils.references import ReferenceCatalog
from utils.references.rmbase import ARCHIVES, import_rmbase3


def pileup(pos, code='a', modified=3, depth=10, strand='+'):
    return f'chr1 {pos} {pos+1} {code} {depth} {strand} {pos} {pos+1} 0 {depth} 0 {modified} {depth-modified} 0 0 0 0 0\n'


def read_table(path):
    with path.open(encoding='utf-8', newline='') as h:
        return list(csv.DictReader(h, delimiter='\t'))


class CallSummaryTests(unittest.TestCase):
    def fixture(self, root):
        seq = list('C' * 60)
        seq[8:13] = 'AGACT'
        seq[28:33] = 'AGTCT'
        fa = root / 'ref.fa'
        fa.write_bytes(('>chr1\n' + ''.join(seq) + '\n').encode())
        Path(str(fa) + '.fai').write_text('chr1\t60\t6\t60\t61\n')
        bed = root / 'ref.bed'
        bed.write_text('chr1\t10\t11\tx\t0\t+\n' * 2 + 'chr1\t30\t31\ty\t0\t+\n')
        refs = [dict(id=ref, kind=kind, cell=cell, modification='m6A', assembly='GRCh38',
                     coordinates='BED0', path='ref.bed')
                for ref, kind, cell in [('A', 'short_read', 'HeLa'), ('B', 'short_read', 'HeLa'),
                                        ('DB', 'database', '*'), ('other_cell', 'short_read', 'HEK293T')]]
        manifest = root / 'catalog.json'
        manifest.write_text(json.dumps(dict(assembly='GRCh38', sources=refs)))
        source = root / 'input.bed'
        source.write_text(pileup(0) + pileup(10) + pileup(20, modified=1) + pileup(30, strand='-') + pileup(40, modified=0))
        samples = [dict(sample_id=condition, cell='HeLa', reference_cell='HeLa', condition=condition, bedmethyl=source)
                   for condition in ('WT', 'IVT')]
        thresholds = [dict(name='standard', min_depth=10, min_modified=3, min_af=.3),
                      dict(name='strict', min_depth=10, min_modified=3, min_af=.31)]
        return fa, manifest, samples, thresholds

    def test_counts_motifs_strands_unions_and_ivt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fa, manifest, samples, thresholds = self.fixture(root)
            out = root / 'out'
            summarize_calls(samples, thresholds, fa, manifest, out)
            counts = {(r['sample_id'], r['threshold'], r['modification'], r['motif_group']): r
                      for r in read_table(out / 'call_counts.tsv')}
            row = counts['IVT', 'standard', 'm6A', 'all']
            self.assertEqual([row[k] for k in ['input_sites', 'raw_call_sites', 'call_sites', 'modified_reads_at_calls', 'ivt_false_call_sites']],
                             ['5', '4', '3', '9', '3'])
            self.assertEqual(counts['WT', 'standard', 'm6A', 'all']['ivt_false_call_sites'], '')
            self.assertEqual(counts['WT', 'standard', 'm6A', 'DRACH']['call_sites'], '2')
            self.assertEqual(counts['WT', 'standard', 'm6A', 'unknown']['call_sites'], '1')
            self.assertEqual(counts['WT', 'strict', 'm6A', 'all']['call_sites'], '0')
            self.assertEqual(counts['WT', 'standard', 'm5C', 'all']['call_sites'], '0')
            matches = {r['source_id']: r for r in read_table(out / 'overlap_counts.tsv')
                       if r['sample_id'] == 'IVT' and r['threshold'] == 'standard' and r['modification'] == 'm6A' and r['motif_group'] == 'all'}
            for source in ['A', 'B', 'DB', 'short_read_any', 'database_any']:
                self.assertEqual(matches[source]['overlap_sites'], '1')
            self.assertNotIn('other_cell', matches)
            absent = [r for r in read_table(out / 'overlap_counts.tsv') if r['modification'] == 'm5C']
            self.assertTrue(all(r['overlap_sites'] == '' and r['reference_status'] == 'unavailable' for r in absent))
            self.assertFalse((out / 'training_sites.tsv').exists())
            with self.assertRaises(FileExistsError):
                summarize_calls(samples, thresholds, fa, manifest, out)

    def test_duplicate_input_is_failed_and_hg19_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fa, manifest, samples, thresholds = self.fixture(root)
            Path(samples[0]['bedmethyl']).write_text(pileup(10) * 2)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                summarize_calls(samples, thresholds, fa, manifest, root / 'out')
            self.assertTrue((root / 'out/FAILED.txt').exists())
            self.assertFalse((root / 'out/run.json').exists())
            config = json.loads(manifest.read_text())
            config['sources'][0]['assembly'] = 'hg19'
            manifest.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'lift hg19'):
                ReferenceCatalog(manifest)

    def test_gzip_and_unsorted(self):
        import gzip
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'sample.bed.gz'
            with gzip.open(path, 'wt') as h:
                h.write(pileup(20) + pileup(10))
            with self.assertRaisesRegex(ValueError, 'sorted'):
                list(bedmethyl_rows(path))


class RMBaseImportTests(unittest.TestCase):
    def test_real_format_editing_selection_dedup_and_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for mod, (name, code) in ARCHIVES.items():
                def line(kind):
                    return '\t'.join(['chr1', '10', '11', 'site1', '0', '+', kind] + ['na'] * 22) + '\n'
                data = (line(code) * 2 + (line('C-U') if mod == 'inosine' else '')).encode()
                with tarfile.open(root / f'hg38.{name}.tar.gz', 'w:gz') as archive:
                    member = tarfile.TarInfo(f'human.hg38.{name}.result.col29.bed')
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
            report = import_rmbase3(root, root / 'normalized')
            self.assertEqual([r['unique_sites'] for r in report['files']], [1] * 4)
            self.assertEqual([r['duplicate_rows'] for r in report['files']], [1] * 4)
            editing = next(r for r in report['files'] if r['modification'] == 'inosine')
            self.assertEqual(editing['observed_types'], {'A-I': 2, 'C-U': 1})
            self.assertEqual(len(read_table(root / 'normalized/inosine.provenance.tsv')), 2)
