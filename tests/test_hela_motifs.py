"""RNA-oriented ±5nt sequence windows and descriptive motif accounting."""
import unittest

from analysis.hela_motifs.compare import distance_to_drach, jensen_shannon, member_groups, sequence_context, summarize


class TinyFasta:
    sequence='ACGTTACGTTA'

    def get_reference_length(self,chrom):
        return len(self.sequence)

    def fetch(self,chrom,lo,hi):
        return self.sequence[lo:hi]


class HeLaMotifsTest(unittest.TestCase):
    def test_strand_orientation_and_contig_edge_padding(self):
        fasta=TinyFasta()
        plus,padding=sequence_context(fasta,'chr1',5,'+')
        minus,_=sequence_context(fasta,'chr1',5,'-')
        self.assertEqual(plus,'ACGTTACGTTA')
        self.assertEqual(minus,'TAACGTAACGT')
        self.assertEqual(padding,0)
        edge,padding=sequence_context(fasta,'chr1',0,'+')
        self.assertEqual(edge,'NNNNNACGTTA')
        self.assertEqual(padding,5)
        reverse,_=sequence_context(fasta,'chr1',0,'-')
        self.assertEqual(reverse,'TAACGTNNNNN')

    def test_minimum_distance_uses_allowed_iupac_bases(self):
        self.assertEqual(distance_to_drach('GGACT'),0)
        self.assertEqual(distance_to_drach('GGATT'),1)
        self.assertEqual(distance_to_drach('CCACC'),2)
        self.assertIsNone(distance_to_drach('NGACT'))

    def test_group_membership_and_frequency_denominators(self):
        self.assertEqual(member_groups(1,1),['GLORI_all','NN_all','shared'])
        records=[]
        for glori,nn,seq in [(1,0,'AAGGATTTTAA'),(1,1,'AAGGATTTTAA'),(0,1,'NAGGATTTTAA')]:
            records.append(dict(GLORI=glori,NN=nn,dna_11nt=seq,central_5mer=seq[3:8],
                DRACH_distance=distance_to_drach(seq[3:8]),reference_A=1,padding_bases=0,
                weblogo_complete_11nt=int('N' not in seq)))
        _,summary,positions,_,distances,_=summarize(records,5)
        counts={row['group']:row for row in summary}
        self.assertEqual(counts['GLORI_all']['sites'],2)
        self.assertEqual(counts['NN_all']['sites'],2)
        self.assertEqual(counts['NN_all']['complete_11nt_sites'],1)
        for group in counts:
            for position in range(-5,6):
                rows=[row for row in positions if row['group']==group and row['position']==position and row['scope']=='complete_11nt']
                self.assertEqual(sum(row['count'] for row in rows),counts[group]['complete_11nt_sites'])
        for group in counts:
            self.assertEqual(sum(row['count'] for row in distances if row['group']==group),counts[group]['sites'])

    def test_js_is_symmetric_zero_for_equal_and_one_for_disjoint(self):
        self.assertEqual(jensen_shannon([1,0,0,0],[1,0,0,0]),0)
        self.assertEqual(jensen_shannon([1,0,0,0],[0,1,0,0]),1)
        self.assertAlmostEqual(jensen_shannon([.4,.3,.2,.1],[.1,.2,.3,.4]),
                               jensen_shannon([.1,.2,.3,.4],[.4,.3,.2,.1]))


if __name__=='__main__':
    unittest.main()
