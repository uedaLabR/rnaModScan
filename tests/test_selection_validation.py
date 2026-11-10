import csv
import json
from pathlib import Path
import tempfile
import unittest

from training.selection_validation import verify_selected_sites, verify_source_report
from utils.adenine import CLASSES
from utils.io.artifacts import file_info
from utils.references.catalog import sha256


class SelectionValidationTest(unittest.TestCase):
    def fixture(self, root):
        samples, inputs = [], []
        for condition in ('WT', 'IVT'):
            for suffix in ('bed', 'log', 'bam', 'bam.bai'):
                (root / f'{condition}.{suffix}').write_text('synthetic\n')
            sample_id = 'HEK293T_' + condition
            samples.append(dict(sample_id=sample_id, cell='HEK293T', reference_cell='HEK293T', condition=condition,
                enabled=True, bam=root / f'{condition}.bam', bedmethyl=root / f'{condition}.bed', pileup_log=root / f'{condition}.log'))
            inputs.append(dict(sample_id=sample_id, status='available', input=file_info(root / f'{condition}.bed'),
                               pileup_log=dict(sha256=sha256(root / f'{condition}.log'))))
        source = dict(schema='site_lists_v2', status='complete', assembly='GRCh38', coordinates='BED0',
                      thresholds=dict(min_depth=10,min_modified=3,min_af=.1),
                      ivt_min_modified=3, ivt_af_ratio=.6, inputs=inputs,
                      pairs=[dict(cell='HEK293T',reference_cell='HEK293T',wt_sample='HEK293T_WT',ivt_sample='HEK293T_IVT')])
        source_path = root / 'source_run.json'
        source_path.write_text(json.dumps(source))
        selection = root / 'output/selection'
        (selection / 'adenine').mkdir(parents=True)
        path = selection / 'adenine/training_sites.tsv'
        fields = ['row_id','sample_id','cell','reference_cell','condition','chrom','start','end','strand',
                  'modification','label','label_status']
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle,fields,delimiter='\t')
            writer.writeheader()
            for label, condition in [('0','WT'),('1','WT'),('2','IVT')]:
                start = 100 + int(label) * 100
                writer.writerow(dict(row_id=f'{condition}:{start}', sample_id='HEK293T_'+condition,
                    cell='HEK293T',reference_cell='HEK293T',condition=condition,chrom='chr1',start=start,end=start+1,
                    strand='+',modification='adenine',label=label,label_status='final'))
        run = dict(status='complete',schema='adenine_training_sites_v1',class_names=list(CLASSES),
                   seed=42,window_flank=9,selected_sites=3,selected_sha256=sha256(path))
        (selection / 'run.json').write_text(json.dumps(run))
        config = dict(targets=['m6A', 'inosine'], samples=samples,
                      candidates=dict(min_depth=10, min_modified=3, min_af=.1,
                                      ivt_min_modified=3, ivt_af_ratio=.6),
                      features=dict(flank=9), selection=dict(seed=42),
                      nn=dict(task='adenine_three_class'))
        return config, source_path, selection

    def test_selected_labels_and_source_provenance_are_checked_without_parent_lists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, source, selection = self.fixture(root)
            report = verify_source_report(config, source)
            path, run, sample_counts, class_counts = verify_selected_sites(config, selection)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(path, selection / 'adenine/training_sites.tsv')
            self.assertEqual(run['selected_sites'], 3)
            self.assertEqual(sample_counts, {'HEK293T_WT': 2, 'HEK293T_IVT': 1})
            self.assertEqual(class_counts, dict(m6A=1, inosine=1, false_positive=1))

    def test_changed_selected_sites_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, _, selection = self.fixture(Path(temporary))
            path = selection / 'adenine/training_sites.tsv'
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_selected_sites(config, selection)

    def test_changed_original_bedmethyl_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, source, _ = self.fixture(root)
            (root / 'WT.bed').write_text('changed source input\n')
            with self.assertRaisesRegex(ValueError, 'bedMethyl changed'):
                verify_source_report(config, source)


if __name__ == '__main__':
    unittest.main()
