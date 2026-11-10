"""Exercise optional pileup with a fake executable, never a real modkit BAM run."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]


class ModkitPredictionWorkflowTest(unittest.TestCase):
    def test_optional_pileup_and_reference_only_outputs_keep_total_depth_af(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / 'fake_modkit'
            executable.write_text('#!' + sys.executable + '\n' + '''import sys
from pathlib import Path
if sys.argv[1] == '--version':
    print('modkit synthetic-test')
else:
    assert sys.argv[1] == 'pileup'
    output = Path(sys.argv[3])
    output.write_text(''.join(f'chr1 {pos} {pos+1} m 10 + {pos} {pos+1} 0,0,0 10 30 3 7 0 0 10 0 0\\n' for pos in (10,20)))
    log = Path(sys.argv[sys.argv.index('--log-filepath')+1])
    log.write_text('setting threshold 0.7 (qual: 179) for base C, percentile 0.1\\n')
''')
            executable.chmod(0o755)
            (root / 'input.bam').write_bytes(b'synthetic placeholder; not read by pysam')
            (root / 'input.bam.bai').write_bytes(b'index placeholder')
            (root / 'genome.fa').write_text('>chr1\n' + 'A' * 100 + '\n')
            (root / 'genome.fa.fai').write_text('chr1\t100\t6\t100\t101\n')
            (root / 'known.bed').write_text('chr1\t10\t11\tknown\t0\t+\n')
            (root / 'catalog.json').write_text(json.dumps(dict(assembly='GRCh38', sources=[dict(
                id='known_m5C', kind='database', database='RMBase', version='3.0', cell='*', modification='m5C',
                assembly='GRCh38', coordinates='BED0', path='known.bed')])) )
            config = dict(mode='predict', output_dir='output', targets=['m5C'],
                modkit=dict(run_pileup=True, executable=str(executable)),
                references=dict(catalog='catalog.json', fasta='genome.fa'),
                sample=dict(sample_id='sample', cell='HEK293T', reference_cell='HEK293T', condition='WT', bam='input.bam'),
                filtering=dict(reference_only_targets=['m5C']), nn={})
            path = root / 'config.yaml'
            path.write_text(yaml.safe_dump(config))
            code = ('import sys; from rnamodscan.cli import main; main(["predict","--config",sys.argv[1]]); '
                    'assert "tensorflow" not in sys.modules; assert "pysam" not in sys.modules')
            result = subprocess.run([sys.executable, '-B', '-c', code, str(path)], cwd=ROOT,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(next(iter(report['stages'])), 'pileup')
            with (root / 'output/predictions/m5C/kept_sites.tsv').open() as handle:
                kept = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(len(kept), 1)
            self.assertEqual(kept[0]['start'], '10')
            self.assertEqual(kept[0]['total_reads'], '20')
            self.assertEqual(kept[0]['failed_calls'], '10')
            self.assertEqual(float(kept[0]['af']), .15)
            self.assertEqual(float(kept[0]['af_pass']), .3)
            self.assertTrue((root / 'output/pileup/command.json').is_file())
            self.assertFalse((root / 'output/features').exists())


if __name__ == '__main__':
    unittest.main()
