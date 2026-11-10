"""Actual BAM -> common per-site Parquet -> checked index and NN batch inputs."""
from array import array
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pysam
import pyarrow.parquet as pq

from utils.features.bam import extract_bam_features
from utils.features.site_bags import SiteBagWriter, build_site_bag_index
from utils.features.site_dataset import SiteBagBatches
from utils.features.site_metadata import TRAINING_FIELDS
from utils.nn.inputs import CHANNEL_NAMES
from utils.features.compact_inputs import restore_read_inputs, tensor_column


def decoded_rows(table, window):
    rows = table.to_pylist()
    for index, row in enumerate(rows):
        row['read_features'] = restore_read_inputs(table[tensor_column(table.schema)][index], window).tolist()
    return rows
from utils.references.catalog import ReferenceCatalog


class SiteBagExtractionTest(unittest.TestCase):
    def test_match_and_mismatch_one_hot_follow_rna_orientation_on_both_strands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fasta = root / 'reference.fa'
            fasta.write_text('>chr1\n' + ''.join('a' if i % 5 == 0 else 'A' for i in range(100)) + '\n')
            pysam.faidx(str(fasta))
            header = pysam.AlignmentHeader.from_references(['chr1'], [100])
            bam = root / 'reads.bam'
            with pysam.AlignmentFile(str(bam), 'wb', header=header) as handle:
                for reverse in (False, True):
                    for mismatch in (False, True):
                        read = pysam.AlignedSegment(header)
                        read.query_name = ('minus' if reverse else 'plus') + ('_mismatch' if mismatch else '_match')
                        sequence = list('A' * 10)
                        sequence[2] = 'C' if mismatch else 'A'
                        read.query_sequence = ''.join(sequence)
                        read.reference_id, read.reference_start = 0, 10
                        read.flag, read.mapping_quality, read.cigarstring = 16 if reverse else 0, 60, '10M'
                        read.query_qualities = array('B', [30] * 10)
                        handle.write(read)
            pysam.index(str(bam))
            rows = self.rows()[:2]
            for row in rows:
                row['sample_id'] = 'sample'
            sites = root / 'sites.tsv'
            with sites.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, list(rows[0]), delimiter='\t')
                writer.writeheader()
                writer.writerows(rows)
            sample = dict(sample_id='sample', cell='HEK293T', reference_cell='HEK293T', condition='WT')
            out = root / 'out'
            sink = SiteBagWriter(out, rows, sample, None, mode='inference', flank=2)
            report = extract_bam_features(sites, 'sample', bam, fasta, out,
                                          flank=2, max_reads=300, sampling_seed=42, site_sink=sink)
            expected = {'plus_match': 'match', 'plus_mismatch': 'C',
                        'minus_match': 'match', 'minus_mismatch': 'G'}
            observed = {}
            categories = ('match', 'A', 'C', 'G', 'T', 'N')
            for item in report['parquet_files']['sites']:
                for site in decoded_rows(pq.ParquetFile(out / 'sites' / item['path']).read(), 5):
                    for raw, features in zip(site['raw_reads'], site['read_features']):
                        category = expected[raw['read_id']]
                        values = [features[2][CHANNEL_NAMES.index('base_' + base)] for base in categories]
                        self.assertEqual(values, [float(base == category) for base in categories])
                        self.assertEqual(raw['mismatch'][2], int(raw['read_id'].endswith('_mismatch')))
                        observed[raw['read_id']] = category
            self.assertEqual(observed, expected)

    def fixture(self, root):
        fasta = root / 'reference.fa'
        fasta.write_text('>chr1\n' + ''.join('a' if i % 5 == 0 else 'A' for i in range(100)) + '\n')
        pysam.faidx(str(fasta))
        header = pysam.AlignmentHeader.from_references(['chr1'], [100])
        bam = root / 'reads.bam'
        with pysam.AlignmentFile(str(bam), 'wb', header=header) as handle:
            for index in range(320):
                reverse = index >= 310
                read = pysam.AlignedSegment(header)
                read.query_name = ('minus_' if reverse else 'plus_') + str(index)
                read.query_sequence = ('T' if reverse else 'A') * 10
                read.reference_id, read.reference_start = 0, 10
                read.flag, read.mapping_quality, read.cigarstring = 16 if reverse else 0, 60, '10M'
                read.query_qualities = array('B', [30] * 10)
                if reverse:
                    read.set_tag('MM', 'A+17596?,7;')
                    read.set_tag('ML', array('B', [128]))
                elif index % 3 != 2:
                    read.set_tag('MM', 'A+a?,2;')
                    read.set_tag('ML', array('B', [0 if index % 3 == 0 else 255]))
                handle.write(read)
        pysam.index(str(bam))
        sources = []
        examples = [('short_m6A', 'short_read', 'HEK293T', 'm6A', 12, '+'),
                    ('short_I_elsewhere', 'short_read', 'HEK293T', 'inosine', 60, '+'),
                    ('other_cell_I', 'short_read', 'HeLa', 'inosine', 12, '+'),
                    ('db_m6A', 'database', '*', 'm6A', 12, '+'),
                    ('db_I', 'database', '*', 'inosine', 12, '-')]
        for name, kind, cell, modification, start, strand in examples:
            path = root / (name + '.bed')
            path.write_text(f'chr1\t{start}\t{start + 1}\tx\t0\t{strand}\n')
            source = dict(id=name, kind=kind, cell=cell, modification=modification,
                          path=path.name, assembly='GRCh38', coordinates='BED0')
            if kind == 'database':
                source.update(database='RMBase', version='3.0')
            sources.append(source)
        catalog_path = root / 'catalog.json'
        catalog_path.write_text(json.dumps(dict(assembly='GRCh38', sources=sources)))
        return bam, fasta, ReferenceCatalog(catalog_path)

    def sample(self, condition='WT'):
        return dict(sample_id='HEK_' + condition, cell='HEK293T', reference_cell='HEK293T', condition=condition)

    def rows(self, condition='WT'):
        rows = []
        examples = [(12, '+', '0', '1', ''), (12, '-', '1', '', '1'), (50, '+', '2', '0', '0')]
        for start, strand, label, original_m6A, original_I in examples:
            if condition == 'IVT':
                label, original_m6A, original_I = '2', '0', ''
            rows.append(dict(row_id=f'HEK_{condition}|chr1:{start}:{strand}:adenine',
                sample_id='HEK_' + condition, chrom='chr1', start=str(start), end=str(start + 1),
                strand=strand, modification='adenine', label=label, label_status='final',
                label_m6A=original_m6A, label_inosine=original_I,
                af_m6A='0.3', af_inosine='', total_reads_m6A='310',
                call_count_m6A='93', call_count_inosine='', gene_names='GENE1', region='CDS'))
        return rows

    def extract(self, root, condition='WT', mode='training'):
        bam, fasta, catalog = self.fixture(root)
        rows = self.rows(condition)
        path = root / 'sites.tsv'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, list(rows[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
        out = root / 'out'
        sink = SiteBagWriter(out, rows, self.sample(condition), catalog, mode=mode,
                             flank=2, max_reads=300, batch_sites=1, file_sites=2)
        report = extract_bam_features(path, self.sample(condition)['sample_id'], bam, fasta, out,
                                      flank=2, max_reads=300, sampling_seed=42, site_sink=sink)
        records = []
        for item in report['parquet_files']['sites']:
            records.extend(decoded_rows(pq.ParquetFile(out / 'sites' / item['path']).read(), 5))
        return out, report, {row['row_id']: row for row in records}

    def test_real_parquet_reservoir_metadata_nn_values_and_no_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out, report, records = self.extract(root)
            source_rows = self.rows()
            plus = records[source_rows[0]['row_id']]
            minus = records[source_rows[1]['row_id']]
            empty = records[source_rows[2]['row_id']]
            self.assertEqual(report['schema'], 'site_bags_parquet_v4')
            self.assertFalse((out / 'reads').exists())
            self.assertEqual(plus['eligible_depth'], 310)
            self.assertEqual(plus['retained_depth'], 300)
            self.assertEqual(len(plus['raw_reads']), 300)
            self.assertEqual(len(plus['read_features']), 300)
            self.assertEqual(len({read['read_id'] for read in plus['raw_reads']}), 300)
            self.assertTrue(plus['support_short_read_m6A'])
            self.assertTrue(plus['support_rmbase_m6A'])
            self.assertFalse(plus['support_short_read_inosine'])  # HeLa support is excluded.
            self.assertTrue(minus['support_rmbase_inosine'])
            self.assertEqual(plus['sequence_repeat_mask'], [True, False, False, False, False])
            self.assertEqual(minus['sequence_repeat_mask'], [False, False, False, False, True])
            self.assertEqual(plus['training_class'], 'm6A')
            self.assertEqual(minus['training_class'], 'inosine')
            self.assertEqual(empty['training_class'], 'false_positive')
            self.assertEqual([plus[name] for name in TRAINING_FIELDS], [True, False, False])
            self.assertEqual((plus['chrom'], plus['position'], plus['strand']), ('chr1', 12, '+'))
            self.assertEqual(plus['gene_names'], 'GENE1')
            self.assertEqual(plus['call_count_m6A'], 93)
            self.assertIsNone(plus['call_count_inosine'])
            self.assertEqual(plus['af_m6A'], .3)
            self.assertIsNone(plus['af_inosine'])
            self.assertEqual(len(minus['read_features']), 10)
            self.assertEqual(len(minus['raw_reads']), 10)
            self.assertEqual(minus['reference_sequence'], 'T' * 5)
            self.assertEqual(minus['sequence_tokens'], [63, 63, 63])
            for raw, features in zip(minus['raw_reads'], minus['read_features']):
                self.assertEqual(raw['ml_byte'][2][1], 128)
                self.assertAlmostEqual(features[2][CHANNEL_NAMES.index('ml_inosine')], 128 / 255, places=6)
            self.assertEqual(empty['raw_reads'], [])
            self.assertEqual(empty['read_features'], [])
            observed = set()
            for raw, features in zip(plus['raw_reads'], plus['read_features']):
                self.assertEqual(len(raw['ml_byte'][2]), 8)
                value = raw['ml_byte'][2][0]
                observed.add(value)
                self.assertEqual(features[2][CHANNEL_NAMES.index('ml_m6A')], 0 if value is None else value / 255)
            self.assertEqual(observed, {None, 0, 255})
            self.assertTrue(all('/strand=' in entry['path'] for entry in report['parquet_files']['sites']))
            index = build_site_bag_index([out], root / 'index')
            self.assertEqual(index['counts']['input_sites'], 3)
            self.assertEqual(index['counts']['indexed_sites'], 2)
            self.assertEqual(index['counts']['zero_read_sites'], 1)
            data = SiteBagBatches(root / 'index/sites.tsv', None, batch_sites=2)
            inputs, labels = next(data.batches())
            self.assertEqual(inputs['read_features'].shape, (2, 300, 5, 18))
            for index, row in enumerate(data.rows):
                original = records[row['row_id']]
                depth = original['retained_depth']
                np.testing.assert_array_equal(inputs['read_features'][index, :depth], original['read_features'])
                np.testing.assert_array_equal(inputs['sequence_repeat_mask'][index], original['sequence_repeat_mask'])
                self.assertEqual(inputs['read_mask'][index].sum(), depth)
                self.assertFalse(inputs['read_mask'][index, depth:].any())
                self.assertEqual(labels[index], ('m6A', 'inosine', 'false_positive').index(original['training_class']))

    def test_common_wrapper_reuses_same_inputs_and_rejects_changed_reference_bed(self):
        from utils.features.site_bag_pipeline import extract_site_bag_sample
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bam, fasta, catalog = self.fixture(root)
            rows = self.rows()
            sites = root / 'sites.tsv'
            with sites.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, list(rows[0]), delimiter='\t')
                writer.writeheader()
                writer.writerows(rows)
            sample = dict(self.sample(), bam=str(bam))
            out = root / 'out'
            first = extract_site_bag_sample(sites, sample, fasta, out, catalog.path,
                                            mode='training', flank=2, candidate_group_gap=None)
            self.assertEqual(first['mode'], 'training')
            self.assertEqual(first['short_read_scope'].split(';')[0], 'explicit_reference_cell')
            self.assertEqual(first['feature_set'], 'sequence_reads_repeat18_v1')
            self.assertEqual(extract_site_bag_sample(sites, sample, fasta, out, catalog.path,
                                                    mode='training', flank=2), json.loads(json.dumps(first)))
            (root / 'short_m6A.bed').write_text('chr1\t13\t14\tx\t0\t+\n')
            with self.assertRaisesRegex(ValueError, 'changed|different'):
                extract_site_bag_sample(sites, sample, fasta, out, catalog.path,
                                        mode='training', flank=2)

    def test_ivt_keeps_saved_false_positive_label_even_with_known_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, records = self.extract(Path(temporary), condition='IVT')
            plus = records[self.rows('IVT')[0]['row_id']]
            self.assertTrue(plus['support_short_read_m6A'])
            self.assertEqual(plus['training_class'], 'false_positive')
            self.assertEqual([plus[name] for name in TRAINING_FIELDS], [False, False, True])

    def test_inference_same_format_has_null_labels_and_keeps_zero_read_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out, _, records = self.extract(root, mode='inference')
            for row in records.values():
                self.assertIsNone(row['label'])
                self.assertIsNone(row['training_class'])
                self.assertTrue(all(row[name] is None for name in TRAINING_FIELDS))
            report = build_site_bag_index([out], root / 'index', require_labels=False)
            self.assertEqual(report['counts']['indexed_sites'], 3)
            data = SiteBagBatches(root / 'index/sites.tsv', None, require_labels=False)
            inputs, labels = next(data.batches())
            self.assertEqual(len(data.rows), 2)
            np.testing.assert_array_equal(sorted(inputs['read_mask'].sum(axis=1)), [10, 300])
            np.testing.assert_array_equal(labels, [-1, -1])
