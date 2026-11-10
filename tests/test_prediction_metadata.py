import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from prediction.candidates import prepare_prediction_sites
from utils.features.position import GtfAnnotation, POSITION_CHANNEL_NAMES, position_inputs
from utils.io.bed import CALL_COUNT_FIELDS


def bed_row(position, code, strand, count):
    return (f'chr1 {position} {position + 1} {code} 10 {strand} {position} {position + 1} '
            f'0,0,0 10 {count * 10} {count} {10 - count} 0 0 0 0 0\n')


class PredictionMetadataTest(unittest.TestCase):
    def prepare(self, root):
        bed = root / 'calls.bed'
        bed.write_text(
            bed_row(10, 'a', '+', 4)
            + bed_row(10, '17596', '-', 6)
            + bed_row(10, '17596', '+', 3)
            + bed_row(10, '69426', '+', 0)
            + bed_row(10, 'a', '-', 5)
            + bed_row(20, 'a', '+', 3)
        )
        log = root / 'pileup.log'
        log.write_text('setting threshold 0.7 (qual: 179) for base A, percentile 0.1\n'
                       'Done, processed 6 rows\n')
        thresholds = {target: dict(min_depth=10, min_modified=3, min_af=.1)
                      for target in ('m6A', 'inosine')}
        return prepare_prediction_sites(bed, log, 'sample', root / 'candidates', thresholds)

    def test_eight_counts_are_joined_by_coordinate_and_strand(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = self.prepare(root)
            self.assertEqual(report['counts']['input_rows'], 6)
            by_target = {}
            for target in ('m6A', 'inosine'):
                with (root / 'candidates' / target / 'sites.tsv').open() as handle:
                    rows = list(csv.DictReader(handle, delimiter='\t'))
                by_target[target] = {(row['start'], row['strand']): row for row in rows}
            plus = by_target['m6A']['10', '+']
            minus = by_target['m6A']['10', '-']
            self.assertEqual((plus['call_count_m6A'], plus['call_count_inosine']), ('4', '3'))
            self.assertEqual((minus['call_count_m6A'], minus['call_count_inosine']), ('5', '6'))
            self.assertEqual(plus['call_count_2OmeA'], '0')  # Observed zero differs from absent.
            self.assertEqual(plus['call_count_m5C'], '')
            self.assertEqual(by_target['m6A']['20', '+']['call_count_inosine'], '')
            for field in CALL_COUNT_FIELDS:
                self.assertEqual(plus[field], by_target['inosine']['10', '+'][field])
            self.assertEqual((plus['total_reads'], plus['modified_reads'], plus['af']), ('10', '4', '0.4'))


    def test_gene_names_preserve_same_strand_gene_overlaps_as_metadata_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'reference.gtf'
            lines = []
            for gene, transcript, name, strand, feature in (
                ('g2', 't2', 'GENE_B', '+', 'CDS'),
                ('g1', 't1', 'GENE_A', '+', 'UTR'),
                ('g3', 't3', 'ANTISENSE', '-', 'CDS'),
            ):
                attrs = f'gene_id "{gene}"; transcript_id "{transcript}"; gene_name "{name}";'
                for kind in ('exon', feature):
                    lines.append(f'chr1\ttest\t{kind}\t11\t20\t.\t{strand}\t.\t{attrs}\n')
            path.write_text(''.join(lines))
            row = GtfAnnotation(path).annotate('chr1', 12, '+')
            self.assertEqual(row['gene_ids'], 'g1;g2')
            self.assertEqual(row['gene_names'], 'GENE_A;GENE_B')
            self.assertEqual((row['region_cds'], row['region_utr'], row['region_ambiguous']), (1, 1, 1))
            vector = position_inputs(row, {('chr1', 12, '+'): row})
            self.assertEqual(len(vector), len(POSITION_CHANNEL_NAMES))
            self.assertEqual(vector, [1., 1., 1., 0., 0., 0., 1.])


if __name__ == '__main__':
    unittest.main()
