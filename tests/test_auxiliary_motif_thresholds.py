"""Known-site retention stays independent of stricter m5C/psi motif cutoffs."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from prediction.auxiliary import auxiliary_motif_eligible
from prediction.bam_recalibration import read_site_decision
from prediction.bed_output import export_filtered_bed, write_bed_header
from prediction.evidence import FLAG_FIELDS
from prediction.filtering import filter_scored_inputs
from prediction.input_parquet import prepare_policy_inputs, SITE_FIELDS
from prediction.predict import write_predictions
from utils.config import load_config
from utils.references.catalog import ReferenceCatalog


class AuxiliaryMotifThresholdTest(unittest.TestCase):
    def fixture(self, root):
        sources=[]
        for target in ['m5C','pseU']:
            bed=root/f'{target}.bed'
            bed.write_text(''.join(f'chr1\t{p}\t{p+1}\tknown\t0\t+\n'for p in [10,20,30]))
            for kind in ['database','short_read']:
                sources.append(dict(id=f'{target}_{kind}',kind=kind,cell='*'if kind=='database'else'HEK293T',
                    modification=target,assembly='GRCh38',coordinates='BED0',path=bed.name,database='fixture',version='1'))
        catalog=root/'catalog.json';catalog.write_text(json.dumps(dict(assembly='GRCh38',sources=sources)))
        return ReferenceCatalog(catalog)

    def rows(self, target):
        # Positions10/20/30 are known;40/50/60 are unknown, with matching motifs.
        result=[]
        for p,depth,modified in [(10,19,8),(20,100,39),(30,20,8),(40,19,8),(50,100,39),(60,20,8)]:
            result.append(dict(row_id=f'{target}:{p}',sample_id='sample',chrom='chr1',start=p,end=p+1,
                strand='+',modification=target,filter_mode='reference_only',retained_depth=0,eligible_depth=0,
                read_offset=0,features_dir='',reference_sequence='',total_reads=depth,modified_reads=modified,
                af=modified/depth,policy_motif_code=3 if target=='m5C' else 5,
                policy_motif_sequence='CTCCA'if target=='m5C'else'GTTCAAAAC',
                auxiliary_motif_eligible=1))  # Stale cached routing claims all eligible.
        return result

    def write_rows(self, path, rows):
        fields=list(dict.fromkeys(SITE_FIELDS+list(rows[0])))
        with path.open('w',newline='')as handle:
            writer=csv.DictWriter(handle,fields,delimiter='\t');writer.writeheader();writer.writerows(rows)
        return fields

    def test_final_writer_rechecks_stale_routing_but_retains_low_known_sites(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);catalog=self.fixture(root)
            for target in ['m5C','pseU']:
                path=root/f'{target}.tsv';fields=self.write_rows(path,self.rows(target))
                out=root/f'pred_{target}';write_predictions(path,{},out,catalog,'HEK293T','WT',.5,site_fields=fields)
                with (out/'all_predictions.tsv').open()as h:rows=list(csv.DictReader(h,delimiter='\t'))
                self.assertEqual([r['final_keep']for r in rows],['1','1','1','0','0','1'])
                self.assertEqual([r['auxiliary_motif_eligible']for r in rows],['0','0','1','0','0','1'])
                self.assertEqual([r['motif_keep']for r in rows],['0','0','3'if target=='m5C'else'5','0','0','3'if target=='m5C'else'5'])
                self.assertTrue(all(r['reference_overlap']=='1'and r['flag_db']=='1'and r['flag_short_read']=='1'for r in rows[:3]))
                self.assertTrue(all(r['keep_score']==''and r['nn_keep']==''and r['flag_nn']=='0'for r in rows))
                self.assertEqual([read_site_decision(r)[1]for r in rows],['retained','retained','retained','rejected','rejected','retained'])
                self.assertEqual(rows[3]['final_reason'],'auxiliary_motif_below_threshold')
                bad=dict(rows[0]);bad['flag_nsun6'if target=='m5C'else'flag_trub1']='1'
                with self.assertRaisesRegex(ValueError,'recorded auxiliary motif thresholds'):
                    read_site_decision(bad)

    def test_stale_cached_parquet_filter_stage_rechecks_and_honors_override(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);catalog=self.fixture(root);source=root/'sites.tsv'
            self.write_rows(source,self.rows('m5C'))
            prepare_policy_inputs(source,'m5C',root/'inputs')
            for name,kwargs,expected,expected_codes in [
                    ('strict',{},['1','1','1','0','0','1'],['0','0','3','0','0','3']),
                    ('previous_AF',dict(auxiliary_motif_min_af=.30),['1','1','1','0','1','1'],['0','3','3','0','3','3']),
                    ('original',dict(auxiliary_motif_min_depth=10,auxiliary_motif_min_af=.1),['1']*6,['3']*6)]:
                out=root/name
                filter_scored_inputs(root/'inputs',None,out,catalog,'HEK293T',**kwargs)
                with (out/'all_predictions.tsv').open()as h:rows=list(csv.DictReader(h,delimiter='\t'))
                self.assertEqual([r['final_keep']for r in rows],expected)
                self.assertEqual([r['motif_keep']for r in rows],expected_codes)
            out=root/'ivt';filter_scored_inputs(root/'inputs',None,out,catalog,'HEK293T',condition='IVT')
            with (out/'all_predictions.tsv').open()as h:rows=list(csv.DictReader(h,delimiter='\t'))
            self.assertEqual([r['final_keep']for r in rows],['0','0','1','0','0','1'])
            self.assertTrue(all(r['flag_db']=='0'and r['flag_short_read']=='0'for r in rows))

    def test_bed_stale_motif_rejected_but_low_known_allowed_and_header_actual(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);source=root/'source.bed'
            source.write_text('chr1\t10\t11\tm\t10\t+\t10\t11\t0,0,0\t10\t30\t3\t7\t0\t0\t0\t0\t0\n')
            decisions=root/'decisions';decisions.mkdir();(decisions/'run.json').write_text(json.dumps(dict(status='complete')))
            fields=['sample_id','chrom','start','strand','modification','final_keep','filter_pass',*FLAG_FIELDS]
            path=decisions/'all_predictions.tsv'
            def save(flag):
                r=dict(sample_id='sample',chrom='chr1',start=10,strand='+',modification='m5C',final_keep=1,filter_pass=1,
                       **dict.fromkeys(FLAG_FIELDS,0));r[flag]=1
                with path.open('w',newline='')as h:w=csv.DictWriter(h,fields,delimiter='\t');w.writeheader();w.writerow(r)
            save('flag_nsun6')
            with self.assertRaisesRegex(ValueError,'auxiliary motif thresholds'):
                export_filtered_bed(source,[decisions],'sample',root/'bad')
            save('flag_db');export_filtered_bed(source,[decisions],'sample',root/'good')
            emitted=[line for line in(root/'good/filtered.bed').read_text().splitlines()if not line.startswith('#')]
            self.assertEqual(emitted[0].split('\t')[-5:],['1','0','0','0','1'])
            header=io.StringIO();write_bed_header(header,auxiliary_motif_min_depth=15,auxiliary_motif_min_af=.25)
            self.assertIn('depth>=15',header.getvalue());self.assertIn('AF>=0.25',header.getvalue())
            self.assertIn('Known-site m5C/pseU flags use the original candidate thresholds',header.getvalue())

    def test_boundaries_defaults_and_non_auxiliary_targets(self):
        for target in ['m5C','pseU']:
            row=dict(modification=target,total_reads=20,modified_reads=8)
            self.assertTrue(auxiliary_motif_eligible(row))
            self.assertFalse(auxiliary_motif_eligible(dict(row,total_reads=19)))
            self.assertFalse(auxiliary_motif_eligible(dict(row,modified_reads=7)))
            self.assertFalse(auxiliary_motif_eligible(dict(row,total_reads=100,modified_reads=39)))
            self.assertFalse(auxiliary_motif_eligible(dict(row,total_reads=10,modified_reads=2),1,.1))
        for target in ['m6A','inosine']:self.assertTrue(auxiliary_motif_eligible(dict(modification=target)))
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);self.fixture(root)
            config=dict(schema_version=1,mode='predict',targets=['m5C'],output_dir='out',
                references=dict(catalog='catalog.json',fasta='genome.fa'),sample=dict(sample_id='sample',cell='HEK293T',
                reference_cell='HEK293T',condition='WT',bedmethyl='calls.bed',pileup_log='pileup.log'),nn={})
            path=root/'config.json';path.write_text(json.dumps(config));loaded=load_config(path,'predict')
            self.assertEqual(loaded['filtering']['auxiliary_motif_min_depth'],20)
            self.assertEqual(loaded['filtering']['auxiliary_motif_min_af'],.40)


if __name__=='__main__':unittest.main()
