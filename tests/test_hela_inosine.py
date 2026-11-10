"""Repeat intervals, ambiguous family overlaps and reference denominator semantics."""
from pathlib import Path
import tempfile
import unittest

from analysis.hela_inosine.compare import annotate_repeat_families, repeat_tables, venn_counts


def repeat_row(chrom,lo,hi,strand,name,family):
    return '\t'.join(map(str,[0,0,0,0,0,chrom,lo,hi,0,strand,name,'SINE',family,0,0,0,1]))+'\n'


class HeLaInosineTest(unittest.TestCase):
    def test_half_open_strand_independence_and_multiple_families(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            fai=root/'genome.fa.fai'
            fai.write_text('chr1\t100\t0\t100\t101\nchr2\t100\t0\t100\t101\n')
            rmsk=root/'rmsk.txt'
            rmsk.write_text(repeat_row('chr1',10,20,'-','AluY','Alu')+
                            repeat_row('chr1',15,18,'+','L1HS','L1')+
                            repeat_row('chr1',15,18,'+','AluSx','Alu'))
            sites={('chr1',10,'+'),('chr1',10,'-'),('chr1',15,'+'),('chr1',20,'+'),('chr2',10,'+')}
            result,audit=annotate_repeat_families(sites,rmsk,fai)
            self.assertEqual(result['chr1',10,'+']['repeat_category'],'Alu')
            self.assertEqual(result['chr1',10,'-']['repeat_category'],'Alu')
            self.assertEqual(result['chr1',15,'+']['repeat_category'],'Multiple_families')
            self.assertEqual(result['chr1',15,'+']['repeat_families'],'Alu;L1')
            self.assertEqual(result['chr1',20,'+']['repeat_category'],'Non-repeat')
            self.assertEqual(result['chr2',10,'+']['repeat_category'],'Unknown_contig')
            exclusive,inclusive,summary=repeat_tables({'test':sites},result)
            self.assertEqual(sum(row['sites'] for row in exclusive),5)
            self.assertAlmostEqual(sum(row['percent_all_sites'] for row in exclusive),100)
            self.assertEqual(summary[0]['any_Alu_sites'],3)
            self.assertEqual(summary[0]['unknown_annotation_sites'],1)
            self.assertEqual(summary[0]['multiple_family_sites'],1)
            counts={row['repeat_family']:row['sites'] for row in inclusive}
            self.assertEqual(counts,{'Alu':3,'L1':1})

    def test_venn_partitions_include_only_exact_site_keys(self):
        a={('chr1',10,'+'),('chr1',10,'-'),('chr1',20,'+')}
        b={('chr1',10,'+'),('chr1',30,'+')}
        counts=venn_counts(a,b)
        self.assertEqual(counts,dict(reference_only=2,shared=1,nn_only=1,reference_total=3,nn_total=2,union=4))

    def test_outside_genome_is_not_silently_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'ref.fai').write_text('chr1\t100\t0\t100\t101\n')
            (root/'rmsk.txt').write_text('')
            with self.assertRaisesRegex(ValueError,'outside'):
                annotate_repeat_families({('chr1',100,'+')},root/'rmsk.txt',root/'ref.fai')


if __name__=='__main__':
    unittest.main()
