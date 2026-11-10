import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from prediction.predict import write_predictions
from utils.features.index import INDEX_FIELDS
from utils.references.catalog import ReferenceCatalog

ROOT = Path(__file__).resolve().parents[1]


class PredictionPolicyTest(unittest.TestCase):
    def test_reference_only_complete_workflow_requires_fasta_but_no_bam_or_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'calls.bed').write_text(
                'chr1 10 11 m 10 + 10 11 0,0,0 10 30 3 7 0 0 0 0 0\n'
                'chr1 20 21 m 10 + 20 21 0,0,0 10 30 3 7 0 0 0 0 0\n'
                'chr2 30 31 17802 10 - 30 31 0,0,0 10 30 3 7 0 0 0 0 0\n')
            (root / 'pileup.log').write_text('command line: fixture\n')
            (root / 'genome.fa').write_text('>chr1\n' + 'A' * 100 + '\n')
            (root / 'genome.fa.fai').write_text('chr1\t100\t6\t100\t101\n')
            (root / 'known.bed').write_text('chr1\t10\t11\tknown\t0\t+\n')
            (root / 'psi.bed').write_text('chr2\t30\t31\tknown_psi\t0\t-\n')
            sources = [dict(id=target, kind='database', database='RMBase', version='3.0', cell='*',
                            modification=target, assembly='GRCh38', coordinates='BED0', path=path)
                       for target, path in [('m5C', 'known.bed'), ('pseU', 'psi.bed')]]
            (root / 'catalog.json').write_text(json.dumps(dict(assembly='GRCh38', sources=sources)))
            config = dict(schema_version=1, mode='predict', output_dir='out', targets=['m5C', 'pseU'],
                          references=dict(catalog='catalog.json', fasta='genome.fa'),
                          filtering=dict(reference_only_targets=['m5C', 'pseU'], m6a_drach_keep=True),
                          sample=dict(sample_id='sample', cell='HEK293T', reference_cell='HEK293T',
                                      condition='WT', bedmethyl='calls.bed', pileup_log='pileup.log'),
                          nn={})
            (root / 'config.yaml').write_text(yaml.safe_dump(config))
            code = ('import sys; from rnamodscan.cli import main; '
                    'main(["predict", "--config", sys.argv[1]]); '
                    'assert "tensorflow" not in sys.modules; assert "pysam" not in sys.modules')
            result = subprocess.run([sys.executable, '-B', '-c', code, str(root / 'config.yaml')],
                                    cwd=ROOT, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for target, expected in [('m5C', ['1', '0']), ('pseU', ['1'])]:
                with (root / 'out/predictions' / target / 'all_predictions.tsv').open() as handle:
                    rows = list(csv.DictReader(handle, delimiter='\t'))
                self.assertEqual([row['final_keep'] for row in rows], expected)
                for row in rows:
                    self.assertEqual(row['keep_score'], '')
                    self.assertEqual(row['prediction_status'], 'reference_only')
                    self.assertEqual(row['af'], '0.3')
                    self.assertEqual(row['modified_reads'], '3')
                    self.assertEqual(row['call_count_m6A'], '')  # Missing differs from zero.
                self.assertFalse((root / 'out/scores' / target).exists())
            self.assertFalse((root / 'out/features').exists())

    def test_drach_keeps_low_nn_score_but_non_drach_uses_score_and_reference_rescue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / 'catalog.json'
            (root / 'known.bed').write_text('chr1\t30\t31\tknown\t0\t+\n')
            catalog_path.write_text(json.dumps(dict(assembly='GRCh38', sources=[dict(
                id='known', kind='short_read', cell='HEK293T', modification='m6A',
                assembly='GRCh38', coordinates='BED0', path='known.bed')])))
            index = root / 'sites.tsv'
            with index.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, INDEX_FIELDS, delimiter='\t')
                writer.writeheader()
                for start, mode, group in [(10, 'nn_drach_keep', 'DRACH'), (20, 'nn', 'non_DRACH'), (30, 'nn', 'non_DRACH')]:
                    writer.writerow(dict(row_id=str(start), sample_id='sample', chrom='chr1',
                        start=start, end=start+1, strand='+', modification='m6A', retained_depth=10,
                        filter_mode=mode, motif_group=group))
            write_predictions(index, {'10': .1, '20': .1, '30': .1}, root / 'out',
                              ReferenceCatalog(catalog_path), 'HEK293T', 'WT', .5)
            with (root / 'out/all_predictions.tsv').open() as handle:
                rows = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual([row['final_keep'] for row in rows], ['1', '0', '1'])
            self.assertTrue(all(row['nn_keep'] == '0' and row['keep_score'] == '0.1' for row in rows))
            self.assertEqual(rows[0]['final_reason'], 'drach_keep')
            self.assertEqual(rows[0]['drach_keep'], '1')
            self.assertEqual(rows[2]['final_reason'], 'reference_rescue')

    def test_reference_only_rejection_is_used_for_bam_recalibration(self):
        from prediction.bam_recalibration import read_site_decision
        row = dict(chrom='chr1', start='10', end='11', strand='+', modification='m5C',
                   prediction_status='reference_only', keep_score='', nn_keep='', rescued='0', final_keep='0')
        self.assertEqual(read_site_decision(row)[1], 'rejected')
        row['final_keep'] = '1'
        self.assertEqual(read_site_decision(row)[1], 'retained')
        row['modification'] = 'm6A'
        with self.assertRaises(ValueError):
            read_site_decision(row)

    def test_routing_sends_drach_and_non_drach_to_nn_and_reference_only_away_from_nn(self):
        from prediction.routing import route_prediction_sites
        from prediction.candidates import FIELDS
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = root / 'candidates'
            candidates.mkdir()
            (candidates / 'run.json').write_text(json.dumps(dict(status='complete')))
            sequence = list('A' * 100)
            sequence[8:13] = 'AGACA'
            sequence[58:63] = 'TGTCT'  # Reverse complement is AGACA.
            fasta = root / 'reference.fa'
            fasta.write_text('>chr1\n' + ''.join(sequence) + '\n')
            Path(str(fasta) + '.fai').write_text('chr1\t100\t6\t100\t101\n')
            for target, sites in [('m6A', [(10, '+'), (30, '+'), (60, '-')]), ('m5C', [(11, '+')])]:
                folder = candidates / target
                folder.mkdir()
                with (folder / 'sites.tsv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, FIELDS, delimiter='\t')
                    writer.writeheader()
                    for pos, strand in sites:
                        writer.writerow(dict(row_id=f'{target}:{pos}', sample_id='sample', chrom='chr1',
                            start=pos, end=pos+1, strand=strand, modification=target))
            route_prediction_sites(candidates, root / 'routing', ['m6A', 'm5C'],
                dict(reference_only_targets=['m5C', 'pseU'], m6a_drach_keep=True), fasta)
            with (root / 'routing/m6A/nn_sites.tsv').open() as handle:
                rows = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual([row['filter_mode'] for row in rows], ['nn_drach_keep', 'nn', 'nn_drach_keep'])
            self.assertEqual([row['motif'] for row in rows], ['AGACA', 'AAAAA', 'AGACA'])
            with (root / 'routing/m5C/nn_sites.tsv').open() as handle:
                self.assertEqual(list(csv.DictReader(handle, delimiter='\t')), [])


if __name__ == '__main__':
    unittest.main()
