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
            for chrom, start, strand, mod, flags in [('chr1', 10, '+', 'm6A', [0, 0, 0, 1, 0]),
                                                  ('chr1', 10, '-', 'inosine', [1, 1, 1, 0, 0]),
                                                  ('chr2', 20, '+', 'm5C', [0, 1, 0, 0, 0]),
                                                  ('chr2', 30, '-', 'pseU', [0, 0, 0, 0, 0])]:
                writer.writerow(dict(sample_id='sample', chrom=chrom, start=start, strand=strand,
                                     modification=mod, final_keep=int(any(flags)), filter_pass=int(any(flags)),
                                     **dict(zip(FLAG_FIELDS, flags))))
        return source, directory, lines

    def test_preserves_values_extra_columns_order_and_separate_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, decisions, lines = self.fixture(root)
            report = export_filtered_bed(source, [decisions], 'sample', root / 'out')
            all_lines = (root / 'out/filtered.bed').read_text().splitlines()
            returned = [line for line in all_lines if not line.startswith('#')]
            self.assertEqual([line.rsplit('\t', 5)[0] for line in returned],
                             [line.rstrip() for line in lines[:3]])
            self.assertEqual(returned[1].split('\t')[-5:], ['1', '1', '1', '0', '1'])
            self.assertEqual(sum(report['counts'].values()), 3)
            self.assertEqual([line for line in (root / 'out/m6A/filtered.bed').read_text().splitlines()
                              if not line.startswith('#')], [returned[0]])
            self.assertTrue(all_lines[0].startswith('#'))
            self.assertEqual(report['schema'],'filtered_bedmethyl_v3')
            self.assertEqual(report['appended_columns'],['flag_db','flag_short_read','flag_nn','motif_keep','filter_pass'])
            self.assertEqual(set(report['motif_keep_encoding']),{'0','1','2'})
            for target in ['m6A','inosine','m5C','pseU']:
                content=(root/'out'/target/'filtered.bed').read_text()
                self.assertTrue(content.startswith('#'))
                self.assertIn('motif_keep',content)
                self.assertIn('DRACH',content)
                self.assertIn('METTL16',content)
            self.assertEqual([line for line in (root/'out/pseU/filtered.bed').read_text().splitlines()
                              if not line.startswith('#')],[])
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

    def test_legacy_four_flag_decisions_remain_readable_with_new_flag_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source,decisions,lines=self.fixture(root)
            path=decisions/'all_predictions.tsv'
            with path.open(newline='') as handle:
                reader=csv.DictReader(handle,delimiter='\t')
                fields=[name for name in reader.fieldnames if name!='flag_mettl16']
                rows=[{key:value for key,value in row.items() if key!='flag_mettl16'} for row in reader]
            with path.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fields,delimiter='\t');writer.writeheader();writer.writerows(rows)
            export_filtered_bed(source,[decisions],'sample',root/'out')
            returned=[line.split('\t') for line in (root/'out/filtered.bed').read_text().splitlines()
                      if not line.startswith('#')]
            self.assertEqual(len(returned),3)
            self.assertEqual([row[-2] for row in returned],['1','0','0'])
            self.assertEqual(['\t'.join(row[:-5]) for row in returned],[line.rstrip() for line in lines[:3]])

    def test_conflicting_drach_and_mettl16_flags_are_not_silently_encoded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source,decisions,_=self.fixture(root)
            path=decisions/'all_predictions.tsv'
            with path.open(newline='') as handle:
                reader=csv.DictReader(handle,delimiter='\t');fields=reader.fieldnames;rows=list(reader)
            rows[0]['flag_mettl16']='1'  # This m6A row already carries DRACH=1.
            with path.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fields,delimiter='\t');writer.writeheader();writer.writerows(rows)
            with self.assertRaises(ValueError):
                export_filtered_bed(source,[decisions],'sample',root/'out')
            self.assertTrue((root/'out/FAILED.txt').is_file())


if __name__ == '__main__':
    unittest.main()
