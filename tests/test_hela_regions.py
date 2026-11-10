"""Stranded UTRs, boundary geometry and all-site denominator preservation."""
from pathlib import Path
import tempfile
import unittest

from analysis.hela_regions.compare import Transcript,RegionAnnotation,summarize


class RegionAnnotationTest(unittest.TestCase):
    def transcript(self,strand):
        tx=Transcript('chr1',strand,'gene','tx',exons=[(10,20),(30,50)],coding=[(15,20),(30,43)])
        tx.finish();return tx

    def test_utrs_reverse_on_minus_and_boundaries_are_half_open(self):
        plus,minus=self.transcript('+'),self.transcript('-')
        self.assertEqual(plus.region(10),'five_prime_UTR')
        self.assertEqual(plus.region(14),'five_prime_UTR')
        self.assertEqual(plus.region(15),'CDS')
        self.assertEqual(plus.region(20),'intron')
        self.assertEqual(plus.region(29),'intron')
        self.assertEqual(plus.region(30),'CDS')
        self.assertEqual(plus.region(43),'three_prime_UTR')
        self.assertEqual(minus.region(10),'three_prime_UTR')
        self.assertEqual(minus.region(43),'five_prime_UTR')

    def test_noncoding_exons_are_not_utr(self):
        tx=Transcript('chr1','+','g','nc',exons=[(1,4),(6,9)])
        tx.finish()
        self.assertEqual(tx.region(1),'noncoding_exon')
        self.assertEqual(tx.region(4),'intron')

    def test_explicit_utr_tags_are_honored(self):
        tx=Transcript('chr1','-','g','tx',exons=[(10,50)],coding=[(20,40)],utr5=[(10,20)])
        tx.finish()
        self.assertEqual(tx.region(15),'five_prime_UTR')

    def test_isoform_ambiguity_stopcodon_and_same_strand(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'a.gtf'
            path.write_text('chr1\tt\texon\t11\t30\t.\t+\t.\tgene_id "g"; transcript_id "a";\n'
                            'chr1\tt\tCDS\t16\t22\t.\t+\t.\tgene_id "g"; transcript_id "a";\n'
                            'chr1\tt\tstop_codon\t23\t25\t.\t+\t.\tgene_id "g"; transcript_id "a";\n'
                            'chr1\tt\texon\t11\t30\t.\t+\t.\tgene_id "g"; transcript_id "b";\n'
                            'chr1\tt\tCDS\t21\t29\t.\t+\t.\tgene_id "g"; transcript_id "b";\n')
            annotation=RegionAnnotation(path)
            row=annotation.annotate('chr1',17,'+')
            self.assertEqual(row['region_exclusive'],'Ambiguous_regions')
            self.assertEqual(row['region_categories'],'CDS;five_prime_UTR')
            self.assertEqual(row['transcript_region_assignments'],'a=CDS;b=five_prime_UTR')
            self.assertEqual(annotation.annotate('chr1',23,'+')['region_exclusive'],'CDS')
            self.assertEqual(annotation.annotate('chr1',17,'-')['region_exclusive'],'outside_same_strand_transcript')
            self.assertEqual(annotation.annotate('chr2',17,'+')['region_exclusive'],'unknown_annotation')

    def test_denominators_keep_ambiguous_and_outside(self):
        rows=[]
        for category,categories,alu in [('CDS','CDS',0),('Ambiguous_regions','CDS;three_prime_UTR',1),
                                       ('outside_same_strand_transcript','outside_same_strand_transcript',0)]:
            rows.append(dict(Slic_seq=1,Slic_seq_DRS_covered=0,Dorado=1,NN=1,any_Alu=alu,
                             region_exclusive=category,region_categories=categories))
        result=summarize(rows)
        exclusive=[row for row in result if row['scheme']=='exclusive' and row['group']=='NN_all']
        self.assertEqual(sum(row['sites'] for row in exclusive),3)
        self.assertAlmostEqual(sum(row['site_percent_all'] for row in exclusive),100)
        ambiguous=next(row for row in exclusive if row['region']=='Ambiguous_regions')
        self.assertEqual(ambiguous['Alu_percent_in_region'],100)
        inclusive=[row for row in result if row['scheme']=='inclusive' and row['group']=='NN_all']
        self.assertEqual(sum(row['sites'] for row in inclusive),4)


if __name__=='__main__':unittest.main()
