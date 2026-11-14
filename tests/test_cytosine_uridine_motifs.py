"""Sequence-rule outputs preserve original candidates without neural inference."""
import csv
from array import array
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from prediction.bam_recalibration import read_site_decision
from prediction.evidence import evidence_values
from prediction.motifs import (POLICY_MOTIFS, motif_matches, policy_motif_flags,
                               motif_keep_code)
from utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = [rule[2] for rule in POLICY_MOTIFS]


class CytosineUridineMotifTest(unittest.TestCase):
    def fixture(self, root, *, enabled=True, condition='WT'):
        fasta, fai, bed = [], [], []
        cases = []
        offset = 0
        examples = [('m5C', 'm', 'CTCCA', 0, 3),
                    ('m5C', 'm', 'CAGGG', 0, 4),
                    ('pseU', '17802', 'GTTCAAAAC', 2, 5),
                    ('pseU', '17802', 'TGTAG', 2, 6)]
        for mod, code, sequence, index, expected in examples:
            for strand in ('+', '-'):
                chrom = f'chr{expected}{"plus" if strand == "+" else "minus"}'
                motif = sequence if strand == '+' else sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
                genome = 'A'*20 + motif + 'A'*30
                start = 20+index if strand == '+' else 20+len(sequence)-index-1
                header = '>'+chrom+'\n'
                fasta.append(header+genome+'\n')
                fai.append(f'{chrom}\t{len(genome)}\t{offset+len(header)}\t{len(genome)}\t{len(genome)+1}\n')
                offset += len(header)+len(genome)+1
                bed.append(self.bed_row(chrom, start, code, strand))
                cases.append((chrom, start, strand, mod, expected))
        # Unknown bases are not accepted as IUPAC wildcard matches; no partial
        # matching at contig edges, and no structure-only or broader PUS7 rule.
        for chrom, genome, start, code in [('chrN', 'A'*20+'CNGGG'+'A'*30, 20, 'm'),
                ('chrPusBroad', 'A'*20+'TATAG'+'A'*30, 22, '17802'),
                ('chrEdgePsi', 'TCAAAA', 0, '17802'), ('chrEdgeC', 'AAAAAC', 5, 'm'),
                ('chrKnown', 'A'*20+'C'+'A'*30, 20, 'm')]:
            header = '>'+chrom+'\n'
            fasta.append(header+genome+'\n')
            fai.append(f'{chrom}\t{len(genome)}\t{offset+len(header)}\t{len(genome)}\t{len(genome)+1}\n')
            offset += len(header)+len(genome)+1
            bed.append(self.bed_row(chrom, start, code, '+', depth=10, nmod=3) if chrom=='chrKnown'
                       else self.bed_row(chrom, start, code, '+'))
        # Threshold-failing duplicate of an otherwise qualifying motif is on a
        # separate contig, so it cannot be introduced by motif retention.
        header, genome = '>chrLow\n', 'A'*20+'CTCCA'+'A'*30
        fasta.append(header+genome+'\n')
        fai.append(f'chrLow\t{len(genome)}\t{offset+len(header)}\t{len(genome)}\t{len(genome)+1}\n')
        bed.append(self.bed_row('chrLow', 20, 'm', '+', nmod=2))
        (root/'genome.fa').write_text(''.join(fasta))
        (root/'genome.fa.fai').write_text(''.join(fai))
        (root/'calls.bed').write_text(''.join(bed))
        (root/'pileup.log').write_text('command line: fixture\n')
        known = 'chr3plus\t20\t21\tknown_motif\t0\t+\nchrKnown\t20\t21\tknown_only\t0\t+\n'
        (root/'known.bed').write_text(known)
        sources = [dict(id='known_'+kind, kind=kind, cell='*' if kind == 'database' else 'HEK293T', modification='m5C',
                        assembly='GRCh38', coordinates='BED0', path='known.bed',
                        database='fixture', version='1') for kind in ('database', 'short_read')]
        (root/'catalog.json').write_text(json.dumps(dict(assembly='GRCh38', sources=sources)))
        filtering = dict(reference_only_targets=['m5C', 'pseU'])
        if enabled is not None:
            filtering.update({key: enabled for key in OPTIONS})
        config = dict(schema_version=1, mode='predict', output_dir='out', targets=['m5C', 'pseU'],
            references=dict(catalog='catalog.json', fasta='genome.fa'), filtering=filtering,
            sample=dict(sample_id='sample', cell='HEK293T', reference_cell='HEK293T', condition=condition,
                        bedmethyl='calls.bed', pileup_log='pileup.log'), nn={})
        (root/'config.yaml').write_text(yaml.safe_dump(config))
        return cases, config

    @staticmethod
    def bed_row(chrom, start, code, strand, *, nmod=8, depth=20):
        return (f'{chrom}\t{start}\t{start+1}\t{code}\t10\t{strand}\t{start}\t{start+1}'
                f'\t0,0,0\t{depth}\t{100*nmod/depth:.3f}\t{nmod}\t{depth-nmod}\t0\t0\t0\t0\t0\n')

    def run_cli(self, root):
        code = ('import sys; from rnamodscan.cli import main; '
                'main(["predict", "--config", sys.argv[1]]); '
                'assert "tensorflow" not in sys.modules; assert "pysam" not in sys.modules')
        result = subprocess.run([sys.executable, '-B', '-c', code, str(root/'config.yaml')],
                                cwd=ROOT, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        records = {}
        for target in ['m5C', 'pseU']:
            with (root/'out/predictions'/target/'all_predictions.tsv').open() as handle:
                for row in csv.DictReader(handle, delimiter='\t'):
                    records[row['chrom']] = row
        return records

    def test_full_policy_pipeline_both_strands_metadata_headers_and_bam_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, _ = self.fixture(root)
            records = self.run_cli(root)
            for chrom, start, strand, mod, code in cases:
                row = records[chrom]
                self.assertEqual((row['modification'], int(row['start']), row['strand']), (mod, start, strand))
                self.assertEqual((row['motif_keep'], row['final_keep'], row['flag_nn']), (str(code), '1', '0'))
                self.assertEqual((row['nn_keep'], row['keep_score']), ('', ''))
                self.assertEqual(read_site_decision(row)[1], 'retained')
                values = evidence_values(row)
                self.assertEqual(sum(map(int, values[3:])), 1)
            self.assertEqual([records['chr3plus'][f] for f in ['flag_db', 'flag_short_read', 'flag_nsun6']], ['1']*3)
            self.assertEqual((records['chrKnown']['motif_keep'], records['chrKnown']['final_keep']), ('0', '1'))
            for chrom in ['chrN', 'chrPusBroad', 'chrEdgePsi', 'chrEdgeC']:
                self.assertEqual((records[chrom]['motif_keep'], records[chrom]['final_keep']), ('0', '0'))
                self.assertEqual(read_site_decision(records[chrom])[1], 'rejected')
            self.assertNotIn('chrLow', records)
            source = {line.split('\t')[0]: line.rstrip('\n') for line in (root/'calls.bed').read_text().splitlines()}
            for target in ['m5C', 'pseU']:
                text = (root/'out/predictions/bedmethyl'/target/'filtered.bed').read_text()
                self.assertIn('3=m5C NSUN6-associated', text)
                self.assertIn('6=pseU PUS7-associated', text)
                self.assertIn('not proof of modification', text)
                for line in text.splitlines():
                    if not line.startswith('#'):
                        raw, *flags = line.rsplit('\t', 5)
                        self.assertEqual(raw, source[line.split('\t')[0]])
                        self.assertEqual(flags[-1], '1')
            self.assertFalse((root/'out/features').exists())
            self.assertFalse((root/'out/scores').exists())

    def test_motif_rules_apply_to_ivt_but_known_site_flags_do_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cases, _ = self.fixture(root, condition='IVT')
            records = self.run_cli(root)
            for chrom, *_ in cases:
                self.assertEqual(records[chrom]['final_keep'], '1')
                self.assertEqual((records[chrom]['flag_db'], records[chrom]['flag_short_read']), ('0', '0'))
            self.assertEqual(records['chrKnown']['final_keep'], '0')

    def test_omitted_switches_preserve_legacy_known_only_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cases, _ = self.fixture(root, enabled=None)
            config = load_config(root/'config.yaml', 'predict')
            self.assertTrue(all(config['filtering'][key] is False for key in OPTIONS))
            records = self.run_cli(root)
            for chrom, *_ in cases:
                self.assertEqual(records[chrom]['motif_keep'], '0')
                self.assertEqual(records[chrom]['final_keep'], '1' if chrom == 'chr3plus' else '0')

    def test_iupac_and_candidate_offset_validation(self):
        self.assertTrue(motif_matches('TGTAA', 'TGTAR'))
        self.assertTrue(motif_matches('UGUAG', 'TGTAR'))
        self.assertFalse(motif_matches('TGTAC', 'TGTAR'))
        self.assertFalse(motif_matches('TGTAN', 'TGTAR'))
        self.assertFalse(motif_matches('CNTGG', 'CNGGG'))
        for sequence, pattern in [('CTCCA', 'CTCCA'), ('CAGGG', 'CNGGG'),
                                  ('GTTCAAAAC', 'GTTCNANNC'), ('TGTAG', 'TGTAR')]:
            for index in range(len(sequence)):
                with self.subTest(sequence=sequence, index=index):
                    self.assertFalse(motif_matches(sequence[:index]+'N'+sequence[index+1:], pattern))
        for target, sequence, code in [('m5C', 'CTCCA', '3'), ('pseU', 'GTTCAAAAC', '5')]:
            flags = policy_motif_flags(dict(modification=target, policy_motif_code=code,
                                           policy_motif_sequence=sequence))
            self.assertEqual(motif_keep_code(0, 0, *flags.values()), int(code))
            with self.assertRaises(ValueError):
                policy_motif_flags(dict(modification='inosine', policy_motif_code=code,
                                       policy_motif_sequence=sequence))
        with self.assertRaises(ValueError):
            motif_keep_code(0, 0, 1, 1)

    def test_new_configuration_types_and_shipped_defaults(self):
        for path in [ROOT/'prediction/config.yaml', ROOT/'docker/predict.yaml']:
            config = yaml.safe_load(path.read_text())
            self.assertTrue(all(config['filtering'][key] is True for key in OPTIONS))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, config = self.fixture(root)
            for key in OPTIONS:
                bad = json.loads(json.dumps(config)); bad['filtering'][key] = 'true'
                (root/'bad.yaml').write_text(yaml.safe_dump(bad))
                with self.subTest(option=key), self.assertRaisesRegex(ValueError, key):
                    load_config(root/'bad.yaml', 'predict')

    @unittest.skipUnless(importlib.util.find_spec('pysam'), 'pysam not installed')
    def test_filtered_bam_keeps_motif_calls_and_zeros_only_rejected_targets(self):
        import pysam
        from prediction.bam_recalibration import write_filtered_bam
        from utils.references.catalog import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root); self.run_cli(root)
            cases = [('chr3plus', 20, 'C', 'm', 201), ('chr5plus', 22, 'T', '17802', 202),
                     ('chrN', 20, 'C', 'm', 203), ('chrPusBroad', 22, 'T', '17802', 204),
                     ('chrKnown', 20, 'C', 'm', 205)]  # Known at depth10 remains retained.
            bam = root/'source.bam'
            header = dict(HD=dict(VN='1.6', SO='coordinate'), SQ=[dict(SN=c[0], LN=100) for c in cases])
            present = {c[0] for c in cases}
            for line in (root/'genome.fa.fai').read_text().splitlines():
                chrom, length, *_ = line.split('\t')
                if chrom not in present:
                    header['SQ'].append(dict(SN=chrom, LN=int(length)))
            with pysam.AlignmentFile(str(bam), 'wb', header=header) as handle:
                for ref_id, (chrom, start, base, code, score) in enumerate(cases):
                    read = pysam.AlignedSegment(); read.query_name = chrom
                    read.query_sequence = base; read.flag = 0; read.reference_id = ref_id
                    read.reference_start = start; read.mapping_quality = 60; read.cigarstring = '1M'
                    read.query_qualities = array('B', [30])
                    read.set_tag('MM', f'{base}+{code}?,0;'); read.set_tag('ML', array('B', [score])); read.set_tag('MN', 1)
                    handle.write(read)
            before = sha256(bam)
            directories = [root/'out/predictions'/target for target in ['m5C', 'pseU']]
            write_filtered_bam(bam, directories, 'sample', root/'bam_output', threads=1)
            self.assertEqual(sha256(bam), before)
            with pysam.AlignmentFile(str(root/'bam_output/filtered.bam'), 'rb', require_index=True) as handle:
                output = list(handle.fetch(until_eof=True))
            self.assertEqual([list(r.get_tag('ML')) for r in output], [[201], [202], [0], [0], [205]])
            self.assertEqual([r.get_tag('MM') for r in output], [f'{c[2]}+{c[3]}?,0;' for c in cases])
            self.assertTrue(all(list(r.query_qualities) == [30] and r.get_tag('MN') == 1 for r in output))


if __name__ == '__main__':
    unittest.main()
