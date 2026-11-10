"""Scientific set denominators, isoform collapse and separation of rescue."""
import unittest

from analysis.hela_m6a.compare import collapse_isoforms, comparison_views, motif_group, venn_regions


class HeLaM6AComparisonTest(unittest.TestCase):
    def test_isoforms_are_one_locus_without_summing_coverage(self):
        rows=[dict(chrom='chr1',start=10,strand='+',transcript_id='a',n_reads=25,probability_modified=.2),
              dict(chrom='chr1',start=10,strand='+',transcript_id='b',n_reads=20,probability_modified=.9),
              dict(chrom='chr1',start=10,strand='-',transcript_id='c',n_reads=30,probability_modified=.89)]
        loci,scored,positive=collapse_isoforms(rows,.9)
        self.assertEqual(len(scored),2)
        self.assertEqual(positive,{('chr1',10,'+')})
        self.assertEqual(loci['chr1',10,'+']['n_reads_max'],25)
        self.assertEqual(len(loci['chr1',10,'+']['transcripts']),2)
        _,_,lower=collapse_isoforms(rows,.8)
        self.assertEqual(lower,{('chr1',10,'+'),('chr1',10,'-')})
        self.assertTrue(positive <= lower)

    def test_disjoint_regions_sum_to_union_and_reconstruct_each_method(self):
        methods=dict(GLORI={1,3,7},m6Anet={2,3,7},rnaModScan={4,7})
        rows=venn_regions(methods)
        self.assertEqual(sum(row['count'] for row in rows),5)
        self.assertEqual({row['pattern']:row['count'] for row in rows}['111'],1)
        for index,sites in enumerate(methods.values()):
            self.assertEqual(sum(row['count'] for row in rows if row['pattern'][index]=='1'),len(sites))

    def test_rescue_and_nn_are_separate_and_scored_scope_does_not_change_full_catalog(self):
        regions=dict(DRACH={1,2,3},non_DRACH={4,5},unknown={6})
        sets=dict(GLORI={1,3,4,5},m6Anet={1,2},Dorado={1,2,4},NN={2},Final={1,2,4})
        views={view['name']:view for view in comparison_views(regions,sets,{1,2},{1,4})}
        self.assertEqual(views['DRACH__NN__full_catalog']['methods']['GLORI'],{1,3})
        self.assertEqual(views['DRACH__NN__m6anet_scored']['methods']['GLORI'],{1})
        self.assertEqual(views['non_DRACH__NN__full_catalog']['methods']['rnaModScan'],set())
        self.assertTrue(all(view['stage']!='Final' for view in views.values()))
        self.assertNotIn('m6Anet',views['non_DRACH__NN__full_catalog']['methods'])
        self.assertEqual(views['non_DRACH__NN__DRS_depth_eligible']['methods']['GLORI'],{4})

    def test_unknown_motif_is_not_non_drach(self):
        self.assertEqual(motif_group('GGACT'),'DRACH')
        self.assertEqual(motif_group('CAACA'),'non_DRACH')
        self.assertEqual(motif_group('NNANN'),'unknown')
        self.assertEqual(motif_group('GGCCT'),'non_DRACH')


if __name__=='__main__':
    unittest.main()
