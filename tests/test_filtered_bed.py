"""Original bedMethyl values/order survive independent evidence filtering."""
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from prediction.bed_output import FLAG_FIELDS, export_filtered_bed


class FilteredBedTest(unittest.TestCase):
    def fixture(self, root, condition='WT'):
        source = root / 'calls.bed.gz'
        lines = []
        for chrom, start, code, strand in [('chr1', 10, 'a', '+'), ('chr1', 10, '17596', '-'),
                                         ('chr2', 20, 'm', '+'), ('chr2', 30, '17802', '-')]:
            lines.append(f'{chrom}\t{start}\t{start+1}\t{code}\t10\t{strand}\t{start}\t{start+1}'
                         '\t0,0,0\t10\t30.000\t3\t7\t0\t0\t0\t0\t0\textra_source\n')
        with gzip.open(source, 'wt') as handle:
            handle.writelines(lines)
        directory = root / 'decisions'
        directory.mkdir()
        (directory / 'run.json').write_text(json.dumps(dict(status='complete')))
        fields = ['sample_id', 'chrom', 'start', 'strand', 'modification', 'final_keep', 'filter_pass', *FLAG_FIELDS]
        with (directory / 'all_predictions.tsv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fields, delimiter='\t')
            writer.writeheader()
            for chrom, start, strand, mod, flags in [('chr1', 10, '+', 'm6A', [0, 0, 0, 1]),
                                                  ('chr1', 10, '-', 'inosine', [1, 1, 1, 0]),
                                                  ('chr2', 20, '+', 'm5C', [0, 1, 0, 0]),
                                                  ('chr2', 30, '-', 'pseU', [0, 0, 0, 0])]:
                writer.writerow(dict(sample_id='sample', chrom=chrom, start=start, strand=strand,
                                     modification=mod, final_keep=int(any(flags)), filter_pass=int(any(flags)),
                                     **dict(zip(FLAG_FIELDS, flags))))
        return source, directory, lines

    def test_preserves_values_extra_columns_order_and_separate_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, decisions, lines = self.fixture(root)
            report = export_filtered_bed(source, [decisions], 'sample', root / 'out')
            returned = (root / 'out/filtered.bed').read_text().splitlines()
            self.assertEqual([line.rsplit('\t', 5)[0] for line in returned], [line.rstrip() for line in lines[:3]])
            self.assertEqual(returned[1].split('\t')[-5:], ['1', '1', '1', '0', '1'])
            self.assertEqual(sum(report['counts'].values()), 3)
            self.assertEqual((root / 'out/m6A/filtered.bed').read_text().strip(), returned[0])
            self.assertFalse((root / 'out/flags.sqlite').exists())

    def test_refuses_pass_without_original_bed_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, decisions, _ = self.fixture(root)
            text = (decisions / 'all_predictions.tsv').read_text()
            (decisions / 'all_predictions.tsv').write_text(text.replace('chr1\t10\t+\tm6A', 'chr1\t11\t+\tm6A'))
            with self.assertRaisesRegex(ValueError, 'absent'):
                export_filtered_bed(source, [decisions], 'sample', root / 'out')
            self.assertTrue((root / 'out/FAILED.txt').exists())


if __name__ == '__main__':
    unittest.main()
