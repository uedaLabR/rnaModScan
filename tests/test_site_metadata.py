"""Common site evidence is explicit-cell, source-aware and separate from joint labels."""
import json
from pathlib import Path
import tempfile
import unittest

from utils.features.site_metadata import (TRAINING_FIELDS, annotate_site_metadata,
                                         metadata_fields, metadata_provenance)
from utils.references.catalog import ReferenceCatalog


class SiteMetadataTest(unittest.TestCase):
    def sample(self, **changes):
        return dict(dict(sample_id='HEK_WT', cell='HEK293T', reference_cell='HEK293T', condition='WT'), **changes)

    def row(self, **changes):
        values = dict(sample_id='HEK_WT', chrom='chr1', start='100', end='101', strand='-',
                      modification='adenine', label='0', label_status='final',
                      label_m6A='1', label_inosine='')
        values.update(changes)
        return values

    def catalog(self, root):
        sources = []
        examples = [
            ('matched_m6A', 'short_read', 'HEK293T', 'm6A', 100, '-'),
            ('other_cell_I', 'short_read', 'HeLa', 'inosine', 100, '-'),
            ('HEK293_I', 'short_read', 'HEK293', 'inosine', 100, '-'),
            ('matched_I_elsewhere', 'short_read', 'HEK293T', 'inosine', 200, '-'),
            ('database_I', 'database', '*', 'inosine', 100, '-'),
            ('database_m6A_wrong_strand', 'database', '*', 'm6A', 100, '+'),
        ]
        for name, kind, cell, target, start, strand in examples:
            path = root / (name + '.bed')
            path.write_text(f'chr1\t{start}\t{start + 1}\tx\t0\t{strand}\n')
            source = dict(id=name, kind=kind, cell=cell, modification=target,
                          assembly='GRCh38', coordinates='BED0', path=path.name)
            if kind == 'database':
                source.update(database='RMBase', version='3.0')
            sources.append(source)
        cross = dict(sources[0], id='mapped_from_HepG2', source_cell='HepG2',
                     modification='inosine', path='matched_m6A.bed')
        sources.append(cross)
        manifest = root / 'catalog.json'
        manifest.write_text(json.dumps(dict(assembly='GRCh38', sources=sources)))
        return ReferenceCatalog(manifest)

    def test_exact_cell_strand_and_source_provenance_do_not_assign_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = self.catalog(Path(directory))
            row = self.row()
            result = annotate_site_metadata(row, self.sample(), catalog, 'training')
            self.assertTrue(result['support_short_read_m6A'])
            self.assertEqual(result['support_short_read_sources_m6A'], ['matched_m6A'])
            self.assertFalse(result['support_short_read_inosine'])
            self.assertTrue(result['support_rmbase_inosine'])
            self.assertFalse(result['support_rmbase_m6A'])  # Exact strand, not an unstranded match.
            self.assertEqual(result['training_class'], 'm6A')
            self.assertEqual([result[name] for name in TRAINING_FIELDS], [True, False, False])
            self.assertEqual(row['label_inosine'], '')  # Not overwritten with a joint one-hot value.
            provenance = metadata_provenance(catalog)
            self.assertTrue(provenance['catalog_sha256'])
            self.assertIn('sha256', provenance['sources'][0])

    def test_inference_labels_are_null_even_when_stale_labels_exist(self):
        result = annotate_site_metadata(self.row(label='2', label_m6A='1', label_inosine='1'),
                                        self.sample(), None, 'inference')
        self.assertIsNone(result['training_class'])
        self.assertTrue(all(result[name] is None for name in TRAINING_FIELDS))
        self.assertIsNone(result['support_short_read_m6A'])
        self.assertEqual(result['support_short_read_sources_m6A'], [])

    def test_missing_evidence_does_not_change_ivt_override_label(self):
        sample = self.sample(sample_id='HEK_IVT', condition='IVT')
        row = self.row(sample_id='HEK_IVT', label='2', label_m6A='0', label_inosine='')
        with tempfile.TemporaryDirectory() as directory:
            result = annotate_site_metadata(row, sample, self.catalog(Path(directory)), 'training')
        self.assertTrue(result['support_short_read_m6A'])
        self.assertEqual(result['training_class'], 'false_positive')
        self.assertEqual([result[name] for name in TRAINING_FIELDS], [False, False, True])
        with self.assertRaisesRegex(ValueError, 'unknown partner'):
            annotate_site_metadata(self.row(label='2', label_m6A='0'), self.sample(), None, 'training')
        with self.assertRaisesRegex(ValueError, 'Dual-positive'):
            annotate_site_metadata(self.row(label_m6A='1', label_inosine='1'), self.sample(), None, 'training')

    def test_explicit_reference_mapping_is_required_and_missing_training_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = self.catalog(Path(directory))
            mapped = self.sample(reference_cell='HEK293')
            result = annotate_site_metadata(self.row(), mapped, catalog, 'training')
            self.assertTrue(result['support_short_read_inosine'])
            self.assertEqual(result['support_short_read_sources_inosine'], ['HEK293_I'])
        with self.assertRaisesRegex(ValueError, 'saved final'):
            annotate_site_metadata(self.row(label=''), self.sample(), None, 'training')
        with self.assertRaisesRegex(ValueError, 'disagree'):
            annotate_site_metadata(self.row(cell='HeLa'), self.sample(), None, 'training')

    def test_nullable_arrow_fields_round_trip(self):
        import pyarrow as pa
        fields = metadata_fields()
        values = annotate_site_metadata(self.row(), self.sample(), None, 'inference')
        table = pa.Table.from_pylist([values], schema=pa.schema(fields))
        self.assertEqual(table.to_pylist(), [values])
        self.assertTrue(all(field.nullable for field in fields))


if __name__ == '__main__':
    unittest.main()
