import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from utils.features.position import (
    GtfAnnotation, POSITION_CHANNEL_NAMES, POSITION_FIELDS,
    annotate_positions, load_position_sites, position_inputs,
)
from utils.references.catalog import sha256


def gtf_line(chrom, feature, start, end, strand, transcript='t1', gene='g1'):
    return (f'{chrom}\ttest\t{feature}\t{start + 1}\t{end}\t.\t{strand}\t.\t'
            f'gene_id "{gene}"; transcript_id "{transcript}";\n')


class PositionFeaturesTest(unittest.TestCase):
    def annotation(self, root):
        path = root / 'test.gtf'
        text = '#!genome-build GRCh38.p14\n'
        for feature, start, end in [('exon', 100, 110), ('exon', 200, 210),
                                    ('CDS', 105, 110), ('CDS', 200, 203),
                                    ('stop_codon', 203, 206), ('UTR', 100, 105), ('UTR', 203, 210)]:
            text += gtf_line('chr1', feature, start, end, '+')
        # The same position can be CDS in t1 and intronic in t2.
        for start, end in [(100, 103), (207, 210)]:
            text += gtf_line('chr1', 'exon', start, end, '+', 't2')
        # A noncoding/unknown exon is neither CDS nor intron.
        text += gtf_line('chr1', 'exon', 100, 120, '-', 'antisense', 'g2')
        text += gtf_line('chr2', 'exon', 300, 310, '-', 'minus', 'g3')
        text += gtf_line('chr2', 'CDS', 304, 308, '-', 'minus', 'g3')
        text += gtf_line('chr2', 'UTR', 300, 304, '-', 'minus', 'g3')
        text += gtf_line('chr2', 'UTR', 308, 310, '-', 'minus', 'g3')
        path.write_text(text, encoding='utf-8')
        return path

    def test_gencode_gff3_matches_gtf_regions_and_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gtf = self.annotation(root)
            gff = root / 'test.gff3'
            lines = ['##gff-version 3\n', '# GRCh38\n']
            for line in gtf.read_text().splitlines():
                if line.startswith('#'):
                    continue
                fields = line.split('\t')
                if fields[2] == 'UTR':
                    fields[2] = 'five_prime_UTR' if int(fields[3]) < 200 else 'three_prime_UTR'
                import re
                attrs = dict(re.findall(r'(\S+)\s+"([^"]*)"', fields[8]))
                fields[8] = ';'.join(f'{key}={value}' for key, value in attrs.items()) + ';gene_name=Gene%20A'
                lines.append('\t'.join(fields) + '\n')
            gff.write_text(''.join(lines))
            expected, actual = GtfAnnotation(gtf), GtfAnnotation(gff)
            for chrom, strand in [('chr1', '+'), ('chr1', '-'), ('chr2', '-')]:
                for start in [100, 104, 105, 150, 204, 206, 305, 310]:
                    before = expected.annotate(chrom, start, strand)
                    after = actual.annotate(chrom, start, strand)
                    for key in POSITION_CHANNEL_NAMES + ('gene_ids', 'transcript_ids'):
                        self.assertEqual(before[key], after[key])
            self.assertEqual(actual.annotate('chr1', 105, '+')['gene_names'], 'Gene A')

    def test_categories_and_stop_codon_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            annotation = GtfAnnotation(self.annotation(Path(tmp)))
            utr = annotation.annotate('chr1', 104, '+')
            self.assertEqual(utr['region_utr'], 1)
            self.assertEqual(utr['region_intron'], 1)  # Alternative isoform.
            stop = annotation.annotate('chr1', 204, '+')
            self.assertEqual(stop['region_cds'], 1)
            self.assertEqual(stop['region_utr'], 0)
            self.assertEqual(annotation.annotate('chr1', 206, '+')['region_utr'], 1)
            self.assertEqual(annotation.annotate('chr1', 150, '+')['region_intron'], 1)

    def test_isoforms_keep_ambiguous_flags_and_audit_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            annotation = GtfAnnotation(self.annotation(Path(tmp)))
            row = annotation.annotate('chr1', 105, '+')
            self.assertEqual((row['region_cds'], row['region_intron'], row['region_ambiguous']), (1, 1, 1))
            self.assertEqual(row['transcript_count'], 2)
            self.assertEqual(row['gene_count'], 1)
            self.assertEqual(row['transcript_ids'], 't1;t2')
            self.assertEqual(row['gene_ids'], 'g1')
            vector = position_inputs(row, {('chr1', 105, '+'): row})
            self.assertEqual(len(vector), 7)
            self.assertEqual(vector, [1., 0., 1., 1., 0., 0., 1.])

    def test_strand_unknown_and_outside_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            annotation = GtfAnnotation(self.annotation(Path(tmp)))
            other = annotation.annotate('chr1', 105, '-')
            self.assertEqual(other['region_other_exon'], 1)
            self.assertEqual((other['region_cds'], other['region_intron']), (0, 0))
            self.assertEqual(annotation.annotate('chr2', 305, '-')['region_cds'], 1)
            outside = annotation.annotate('chr2', 305, '+')
            self.assertEqual(outside['annotation_known'], 1)
            self.assertEqual(outside['region_outside_transcript'], 1)
            unknown = annotation.annotate('chr_missing', 305, '+')
            self.assertTrue(all(unknown[name] == 0 for name in POSITION_CHANNEL_NAMES))
            self.assertEqual(position_inputs(other, {}), [0.] * 7)
            self.assertEqual(load_position_sites(None), {})

    def test_bed_boundaries_and_compressed_gtf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.annotation(root)
            gz = root / 'test.gtf.gz'
            with gzip.open(gz, 'wt', encoding='utf-8') as handle:
                handle.write(source.read_text())
            annotation = GtfAnnotation(gz)
            self.assertEqual(annotation.annotate('chr2', 299, '-')['region_outside_transcript'], 1)
            self.assertEqual(annotation.annotate('chr2', 300, '-')['region_utr'], 1)
            self.assertEqual(annotation.annotate('chr2', 309, '-')['region_utr'], 1)
            self.assertEqual(annotation.annotate('chr2', 310, '-')['region_outside_transcript'], 1)

    def test_shared_annotation_output_deduplicates_and_tracks_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gtf = self.annotation(root)
            indexes = []
            for name in ('training', 'prediction'):
                index = root / f'{name}.tsv'
                index.write_text('chrom\tstart\tend\tstrand\nchr1\t105\t106\t+\n'
                                 'chr_missing\t100\t101\t+\n', encoding='utf-8')
                indexes.append(index)
            result = annotate_positions(indexes, gtf, root / 'out')
            rows = load_position_sites(result)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows['chr1', 105, '+']['region_cds'], 1)
            run = json.loads((root / 'out/run.json').read_text())
            self.assertEqual(run['status'], 'complete')
            self.assertEqual(run['schema'], 'site_regions_v1')
            self.assertEqual(run['gtf']['sha256'], sha256(gtf))
            self.assertEqual(run['sites_sha256'], sha256(result))
            self.assertEqual(run['position_channel_names'], list(POSITION_CHANNEL_NAMES))
            self.assertFalse((root / 'out/INCOMPLETE.txt').exists())
            self.assertEqual(run['counts']['region_ambiguous'], 1)
            with self.assertRaises(FileExistsError):
                annotate_positions(indexes, gtf, root / 'out')

    def test_invalid_inputs_are_not_silently_classified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid = root / 'invalid.gtf'
            invalid.write_text(gtf_line('chr1', 'exon', 10, 20, '+')
                               + gtf_line('chr1', 'CDS', 15, 25, '+'))
            with self.assertRaisesRegex(ValueError, 'outside exons'):
                GtfAnnotation(invalid)
            invalid.write_text('#!genome-build GRCh37\n' + gtf_line('chr1', 'exon', 10, 20, '+'))
            with self.assertRaisesRegex(ValueError, 'GRCh38'):
                GtfAnnotation(invalid)
            invalid.write_text(gtf_line('chr1', 'transcript', 10, 20, '+'))
            with self.assertRaisesRegex(ValueError, 'no transcripts with exon records'):
                GtfAnnotation(invalid)
            path = root / 'sites.tsv'
            row = GtfAnnotation(self.annotation(root)).annotate('chr1', 105, '+')
            row['annotation_known'] = 0  # An unknown annotation cannot assert CDS.
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=POSITION_FIELDS, delimiter='\t')
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, 'Inconsistent'):
                load_position_sites(path)
            row['annotation_known'] = 1
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=POSITION_FIELDS, delimiter='\t')
                writer.writeheader()
                writer.writerows([row, row])
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_position_sites(path)


if __name__ == '__main__':
    unittest.main()
