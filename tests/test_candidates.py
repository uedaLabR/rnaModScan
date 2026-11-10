import gzip
import tempfile
import unittest
from pathlib import Path
from utils.pileup.candidates import parse_row, filter_bedmethyl


def row(pos=0, mod=3, canonical=7, fail=0, diff=0, nocall=0, deletion=0, code='a', strand='+'):
    cov=mod+canonical
    return f'chr1 {pos} {pos+1} {code} {cov} {strand} {pos} {pos+1} 255,0,0 {cov} 99.9 {mod} {canonical} 0 {deletion} {fail} {diff} {nocall}\n'


class CandidatesTest(unittest.TestCase):
    def test_denominator_and_deletion(self):
        r=parse_row(row(fail=10,diff=5,nocall=5,deletion=100),1)
        self.assertEqual(r['total_reads'],30)
        self.assertEqual(r['af'],0.1)

    def test_threshold_boundaries_gzip_and_codes(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); src=d/'in.gz'
            lines=[row(0,canonical=6), row(1,mod=2,canonical=8),
                   row(2,canonical=27),row(3,canonical=28),
                   row(4,code='17596',strand='-'),row(5,code='17802'),
                   row(6,code='m'),row(7,code='19228')]
            with gzip.open(src,'wt') as f:f.writelines(lines)
            result=filter_bedmethyl(src,d/'out','HeLa_WT')
            self.assertEqual(result['counts']['candidate_rows'],4)
            self.assertEqual(result['counts']['filtered_rows'],3)
            self.assertEqual(result['counts']['unsupported_mod_rows'],1)
            self.assertIn('chr1\t2\t3', (d/'out/candidates.bed').read_text())
            self.assertIn('HeLa_WT|inosine\t0\t-', (d/'out/candidates.bed').read_text())
            with self.assertRaises(FileExistsError):filter_bedmethyl(src,d/'out','x')

    def test_bad_input_does_not_get_success_report(self):
        for text in (row()+row(),row(strand='.'),'',row().replace(' 10 99.9',' 11 99.9')):
            with tempfile.TemporaryDirectory() as d:
                d=Path(d);src=d/'in';src.write_text(text)
                with self.assertRaises(ValueError):filter_bedmethyl(src,d/'out','x')
                self.assertTrue((d/'out/FAILED.txt').exists())
                self.assertFalse((d/'out/summary.json').exists())



if __name__=='__main__':unittest.main()
