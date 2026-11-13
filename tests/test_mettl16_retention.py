"""Exact METTL16 rescue is a post-classification policy for original m6A calls."""
from array import array
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from prediction.adenine_filtering import filter_joint_inputs, retained_adenine_classes
from prediction.bam_recalibration import read_site_decision, write_filtered_bam
from prediction.bed_output import export_filtered_bed
from prediction.candidates import FIELDS, prepare_prediction_sites
from prediction.predict import DECISION_FIELDS, write_predictions
from prediction.routing import ROUTE_FIELDS, route_prediction_sites
from utils.config import load_config
from utils.features.index import INDEX_FIELDS
from utils.references.catalog import ReferenceCatalog, sha256
from utils.references.genome import IndexedFasta, mettl16, mettl16_context

ROOT = Path(__file__).resolve().parents[1]


def table(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def context(nonamer='TACAGAGAA'):
    # The fourth base of the nonamer is the center of the 19-nt window.
    return 'CCCCCC' + nonamer + 'CCCC'


class Mettl16RetentionTest(unittest.TestCase):
    def fixture(self, root):
        sequence = list('C' * 220)
        for pos, motif in [(20, 'TACAGAGAA'), (100, 'TACAGAAAA'),
                           (160, 'TACAGAGAA'), (180, 'TACAGAGAA')]:
            sequence[pos-3:pos+6] = motif
        sequence[55:64] = 'TTCTCTGTA'  # Minus-strand A at BED0 60.
        sequence[138:143] = 'AGACA'
        fasta = root / 'reference.fa'
        fasta.write_text('>chr1\n' + ''.join(sequence) + '\n')
        Path(str(fasta)+'.fai').write_text('chr1\t220\t6\t220\t221\n')
        candidates = root / 'candidates'
        candidates.mkdir()
        (candidates/'run.json').write_text(json.dumps({'status':'complete'}))
        for target, sites in [('m6A', [(20,'+'),(60,'-'),(100,'+'),(140,'+')]),
                              ('inosine', [(20,'+'),(160,'+')]), ('m5C', [(180,'+')])]:
            folder = candidates/target
            folder.mkdir()
            with (folder/'sites.tsv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,FIELDS,delimiter='\t');writer.writeheader()
                for pos,strand in sites:
                    writer.writerow(dict(row_id=f'{target}:{pos}',sample_id='sample',chrom='chr1',
                        start=pos,end=pos+1,strand=strand,modification=target,total_reads=10,
                        modified_reads=3,af=.3))
        return fasta,candidates,''.join(sequence)

    def route_and_filter(self, root, *, enabled=True, condition='WT'):
        fasta,candidates,sequence=self.fixture(root)
        routing=root/'routing'
        policy=dict(reference_only_targets=['m5C','pseU'],m6a_drach_keep=True,
                    m6a_mettl16_keep=enabled)
        route_prediction_sites(candidates,routing,['m6A','inosine','m5C'],policy,fasta)
        decisions={}
        for target in ['m6A','inosine','m5C']:
            source=routing/target/('policy_sites.tsv' if target=='m5C' else 'nn_sites.tsv')
            rows=table(source)
            fields=list(rows[0]) if rows else list(dict.fromkeys(INDEX_FIELDS+ROUTE_FIELDS))
            for row in rows:
                row['retained_depth']=10
            updated=root/(target+'.tsv')
            with updated.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fields,delimiter='\t');writer.writeheader();writer.writerows(rows)
            scores={} if target=='m5C' else {row['row_id']:.1 for row in rows}
            out=root/('decisions_'+target)
            write_predictions(updated,scores,out,ReferenceCatalog(None),'sample',condition,.5,
                              site_fields=fields)
            decisions[target]=out
        return fasta,sequence,decisions

    def test_exact_canonical_only_and_candidate_aligned_window(self):
        for seq in ['TACAGAGAA','UACAGAGAA','tacagagaa']:
            with self.subTest(seq=seq):self.assertTrue(mettl16(seq))
        for seq in ['', 'TACAGAAAA', 'UACAGARAA', 'NACAGAGAA', 'TACAGAGAN',
                    'CTACAGAGAA', 'TACAGAGA', 'ACAGAGAAC']:
            with self.subTest(seq=seq):self.assertFalse(mettl16(seq))
        self.assertTrue(mettl16_context(context()))
        self.assertTrue(mettl16_context(context().replace('T','U')))
        self.assertFalse(mettl16_context(context('TACAGAAAA')))
        self.assertFalse(mettl16_context('CCCCC'+'TACAGAGAA'+'CCCCC'))
        self.assertFalse(mettl16_context('TACAGAGAA'))

    def test_fasta_context_orients_asymmetric_flanks_on_both_strands(self):
        with tempfile.TemporaryDirectory() as tmp:
            fasta,_,_=self.fixture(Path(tmp));genome=IndexedFasta(fasta)
            try:
                genome.chunk_size=8  # Force cache-boundary transitions for asymmetric windows.
                self.assertEqual(genome.context('chr1',20,'+',left=3,right=5),'TACAGAGAA')
                self.assertEqual(genome.context('chr1',60,'-',left=3,right=5),'TACAGAGAA')
                self.assertFalse(mettl16(genome.context('chr1',21,'+',left=3,right=5)))
                self.assertFalse(mettl16(genome.context('chr1',0,'+',left=3,right=5)))
                self.assertEqual(len(genome.context('chr1',0,'-',left=3,right=5)),9)
                self.assertFalse(mettl16(genome.context('chr1',219,'-',left=3,right=5)))
                self.assertEqual(genome.motif('chr1',20,'+'),'ACAGA')
                self.assertEqual(genome.motif('chr1',60,'-'),'ACAGA')
            finally:genome.close()

    def test_original_m6a_only_and_nn_class_still_independent(self):
        scores=dict(predicted_class='false_positive')
        self.assertEqual(retained_adenine_classes(scores,[],False,['m6A','inosine'],True),['m6A'])
        self.assertEqual(retained_adenine_classes(scores,[],False,['inosine'],True),[])
        self.assertEqual(retained_adenine_classes(scores,[],False,['m6A']),[])
        self.assertEqual(retained_adenine_classes(scores,['inosine'],False,['m6A','inosine'],True),['m6A','inosine'])

    def test_config_is_explicit_strict_bool_with_legacy_compatibility(self):
        self.assertTrue(load_config(ROOT/'prediction/config.yaml','predict')['filtering']['m6a_mettl16_keep'])
        self.assertTrue(yaml.safe_load((ROOT/'docker/predict.yaml').read_text())['filtering']['m6a_mettl16_keep'])
        for filename in ['config_hela_wt.yaml','config_hela_wt_v2.yaml']:
            self.assertFalse(load_config(ROOT/'prediction'/filename,'predict')['filtering']['m6a_mettl16_keep'])
        with tempfile.TemporaryDirectory() as tmp:
            values=yaml.safe_load((ROOT/'prediction/config.yaml').read_text())
            path=Path(tmp)/'config.yaml'
            for value in [None,0,1,'true',[],{}]:
                values['filtering']['m6a_mettl16_keep']=value;path.write_text(yaml.safe_dump(values))
                with self.subTest(value=value),self.assertRaisesRegex(ValueError,'m6a_mettl16_keep'):
                    load_config(path,'predict')

    def test_routing_and_decisions_preserve_low_scores_and_independent_flags(self):
        for condition in ['WT','IVT']:
            with self.subTest(condition=condition),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);_,_,decisions=self.route_and_filter(root,condition=condition)
                rows=table(decisions['m6A']/'all_predictions.tsv')
                self.assertEqual([r['filter_mode'] for r in rows],['nn_mettl16_keep','nn_mettl16_keep','nn','nn_drach_keep'])
                self.assertEqual([r['final_keep'] for r in rows],['1','1','0','1'])
                self.assertEqual([r['flag_mettl16'] for r in rows],['1','1','0','0'])
                self.assertEqual([r['flag_drach'] for r in rows],['0','0','0','1'])
                self.assertTrue(all(r['nn_keep']=='0' and r['keep_score']=='0.1' for r in rows))
                self.assertTrue(all(r['prediction_status']=='predicted' for r in rows))
                self.assertTrue(all(r['flag_mettl16']=='0' and r['final_keep']=='0'
                                    for r in table(decisions['inosine']/'all_predictions.tsv')))
                refonly=table(decisions['m5C']/'all_predictions.tsv')[0]
                self.assertEqual((refonly['flag_mettl16'],refonly['final_keep'],refonly['keep_score']),('0','0',''))
                motifs=table(decisions['m6A']/'motif_summary.tsv')
                self.assertEqual(sum(int(r['mettl16_keep_sites']) for r in motifs),2)

    def test_disabled_rule_and_upstream_thresholds_do_not_create_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,decisions=self.route_and_filter(root,enabled=False)
            self.assertEqual([r['final_keep'] for r in table(decisions['m6A']/'all_predictions.tsv')],['0','0','0','1'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fasta,_,_=self.fixture(root)
            bed=root/'calls.bed'
            bed.write_text('chr1 20 21 a 10 + 20 21 0,0,0 10 20 2 8 0 0 0 0 0\n'
                           'chr1 100 101 a 10 + 100 101 0,0,0 10 30 3 7 0 0 0 0 0\n')
            log=root/'pileup.log';log.write_text('command line: synthetic\n')
            prepared=root/'thresholded'
            prepare_prediction_sites(bed,log,'sample',prepared,{'m6A':dict(min_depth=10,min_modified=3,min_af=.1)})
            route_prediction_sites(prepared,root/'routed',['m6A'],dict(reference_only_targets=[],
                m6a_drach_keep=False,m6a_mettl16_keep=True),fasta)
            rows=table(root/'routed/m6A/nn_sites.tsv')
            self.assertEqual([r['start'] for r in rows],['100'])
            self.assertEqual(rows[0]['filter_mode'],'nn')

    def test_zero_read_canonical_kept_and_forged_motif_mode_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fields=list(dict.fromkeys(INDEX_FIELDS+['mettl16_motif']))
            row=dict(row_id='original',sample_id='sample',chrom='chr1',start=20,end=21,strand='+',
                     modification='m6A',retained_depth=0,filter_mode='nn_mettl16_keep',
                     motif_group='non_DRACH',mettl16_motif='TACAGAGAA')
            def run(name,record):
                source=root/(name+'.tsv')
                with source.open('w',newline='') as h:
                    w=csv.DictWriter(h,fields,delimiter='\t');w.writeheader();w.writerow(record)
                write_predictions(source,{},root/name,ReferenceCatalog(None),'sample','WT',.5,site_fields=fields)
                return table(root/name/'all_predictions.tsv')[0]
            result=run('valid',row)
            self.assertEqual((result['final_keep'],result['flag_mettl16'],result['keep_score']),('1','1',''))
            self.assertEqual(result['prediction_status'],'no_reads')
            for name,bad in [('wrong','TACAGAAAA'),('missing','')]:
                with self.subTest(name=name),self.assertRaisesRegex(ValueError,'METTL16'):
                    run(name,dict(row,mettl16_motif=bad))
            legacy=run('legacy_context',dict(row,mettl16_motif='',reference_sequence=context()))
            self.assertEqual(legacy['flag_mettl16'],'1')
            with self.assertRaisesRegex(ValueError,'disagrees'):
                run('disagree',dict(row,reference_sequence=context('TACAGAAAA')))

    def test_bam_decision_supports_fifth_flag_and_old_four_flag_tables(self):
        row=dict(chrom='chr1',start='20',end='21',strand='+',modification='m6A',
                 prediction_status='predicted',keep_score='.1',nn_keep='0',rescued='0',final_keep='1',
                 nn_class='false_positive',flag_db='0',flag_short_read='0',flag_nn='0',
                 flag_drach='0',flag_mettl16='1')
        self.assertEqual(read_site_decision(row)[1],'retained')
        with self.assertRaisesRegex(ValueError,'flags'):
            read_site_decision(dict(row,flag_mettl16='0'))
        legacy=dict(row,flag_drach='1');legacy.pop('flag_mettl16')
        self.assertEqual(read_site_decision(legacy)[1],'retained')
        with self.assertRaises(ValueError):
            read_site_decision(dict(row,modification='inosine'))

    @unittest.skipUnless(importlib.util.find_spec('pysam'),'pysam required')
    def test_original_bed_and_optional_bam_keep_motif_rejected_by_nn(self):
        import pysam
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,sequence,decisions=self.route_and_filter(root)
            source=root/'source.bed'
            source.write_text(''.join(f'chr1\t{pos}\t{pos+1}\ta\t10\t{strand}\t{pos}\t{pos+1}'
                '\t0,0,0\t10\t30.000\t3\t7\t0\t0\t0\t0\t0\textra\n'
                for pos,strand in [(20,'+'),(60,'-'),(100,'+'),(140,'+')]))
            report=export_filtered_bed(source,[decisions['m6A']],'sample',root/'bed')
            returned=[r.split('\t') for r in (root/'bed/filtered.bed').read_text().splitlines()
                      if not r.startswith('#')]
            self.assertEqual([int(r[1]) for r in returned],[20,60,140])
            self.assertEqual(returned[0][-5:],['0','0','0','2','1'])
            self.assertEqual(returned[1][-5:],['0','0','0','2','1'])
            self.assertEqual(returned[2][-5:],['0','0','0','1','1'])
            self.assertEqual(report['appended_columns'],['flag_db','flag_short_read','flag_nn','motif_keep','filter_pass'])
            self.assertEqual(set(report['motif_keep_encoding']),{'0','1','2'})
            self.assertTrue((root/'bed/filtered.bed').read_text().startswith('#'))
            original={r.split('\t')[1]:r for r in source.read_text().splitlines()}
            self.assertTrue(all('\t'.join(r[:-5])==original[r[1]] for r in returned))
            bam=root/'source.bam'
            with pysam.AlignmentFile(str(bam),'wb',header=dict(HD=dict(SO='coordinate'),SQ=[dict(SN='chr1',LN=220)])) as h:
                for pos in [20,100]:
                    read=pysam.AlignedSegment(h.header);read.query_name=str(pos)
                    read.query_sequence=sequence[pos-9:pos+10];read.query_qualities=array('B',[30]*19)
                    read.reference_id=0;read.reference_start=pos-9;read.cigarstring='19M';read.flag=0
                    read.set_tag('MM',f'A+a?,{read.query_sequence[:9].count("A")};')
                    read.set_tag('ML',array('B',[220]));h.write(read)
            original_sha=sha256(bam)
            write_filtered_bam(bam,[decisions['m6A']],'sample',root/'bam',threads=1)
            with pysam.AlignmentFile(str(root/'bam/filtered.bam'),'rb') as h:
                rows=list(h.fetch(until_eof=True))
            self.assertEqual([list(r.get_tag('ML')) for r in rows],[[220],[0]])
            self.assertTrue((root/'bam/filtered.bam.bai').is_file())
            self.assertEqual(sha256(bam),original_sha)

    @unittest.skipUnless(importlib.util.find_spec('pyarrow'),'pyarrow required')
    def test_joint_cached_legacy_metadata_does_not_manufacture_m6a_or_skip_scores(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        for enabled in [False,True]:
            with self.subTest(enabled=enabled),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);inputs=root/'inputs';scores=root/'scores';inputs.mkdir();scores.mkdir()
                rows=[]
                for pos,motif,m6a,ino,depth in [(20,'TACAGAGAA',1,1,10),(40,'TACAGAGAA',0,1,10),
                                                (60,'TACAGAAAA',1,1,10),(80,'TACAGAGAA',1,0,0)]:
                    row={name:None for name in INDEX_FIELDS}
                    row.update(row_id=str(pos),sample_id='sample',chrom='chr1',start=pos,end=pos+1,strand='+',
                               modification='adenine',retained_depth=depth,filter_mode='nn',
                               reference_sequence=context(motif),motif_group='non_DRACH',
                               candidate_m6A=m6a,candidate_inosine=ino)
                    rows.append(row)
                (inputs/'part').mkdir();(scores/'part').mkdir()
                site=inputs/'part/sites.parquet';pq.write_table(pa.Table.from_pylist(rows),site)
                score_rows=[dict(row_id=str(p),predicted_class='false_positive',score_m6A=.1,
                                 score_inosine=.1,score_false_positive=.8) for p in [20,40,60]]
                score=scores/'part/scores.parquet';pq.write_table(pa.Table.from_pylist(score_rows),score)
                model={'fixture':'fixed-no-training'}
                manifest=dict(status='complete',schema='prediction_inputs_v9',target='adenine',model=model,threshold=.5,
                    partitions=[dict(chrom='chr1',strand='+',sites_path='part/sites.parquet',sites_size_bytes=site.stat().st_size,input_count=3)])
                (inputs/'run.json').write_text(json.dumps(manifest))
                (scores/'run.json').write_text(json.dumps(dict(status='complete',schema='prediction_scores_v1',target='adenine',
                    model=model,threshold=.5,input_run_sha256=sha256(inputs/'run.json'),
                    partitions=[dict(chrom='chr1',strand='+',path='part/scores.parquet',size_bytes=score.stat().st_size,rows=3)])))
                filter_joint_inputs(inputs,scores,root/'output',ReferenceCatalog(None),'sample',
                                    m6a_drach_keep=False,m6a_mettl16_keep=enabled)
                m6a=table(root/'output/m6A/all_predictions.tsv');ino=table(root/'output/inosine/all_predictions.tsv')
                self.assertEqual([r['start'] for r in m6a],['20','60','80'])
                self.assertEqual([r['final_keep'] for r in m6a],['1','0','1'] if enabled else ['0','0','0'])
                self.assertTrue(all(r['nn_keep']=='0' and r['keep_score']=='0.1' for r in m6a[:2]))
                self.assertTrue(all(r['final_keep']=='0' and r['flag_mettl16']=='0' for r in ino))
                classes=table(root/'output/adenine/all_classifications.tsv')
                self.assertEqual(len(classes),4)
                self.assertEqual([r['mettl16_keep'] for r in classes],['1','0','0','1'] if enabled else ['0']*4)
                self.assertTrue(all(r['drach_keep']=='0' for r in classes))


if __name__=='__main__':
    unittest.main()
