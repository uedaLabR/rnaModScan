import csv
import importlib.util
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from prediction.predict import site_decision, validate_input_schema, write_predictions
from utils.nn.inputs import CHANNEL_NAMES, MODIFICATIONS
from utils.adenine import CLASSES
from utils.references.catalog import ReferenceCatalog, sha256
from utils.features.index import INDEX_FIELDS


def input_schema(window=19):
    return dict(feature_set='sequence_reads_repeat18_v1', missing_ml='zero', window=window, modification_channels=list(MODIFICATIONS),
                read_channel_names=list(CHANNEL_NAMES),
                splice_distance='same_read_CIGAR_N_query_bases_adjacent_exon_zero',
                splice_distance_cap=1000, splice_missing_distance=1000,
                splice_distance_transform='log1p(min(d,1000))/log1p(1000)', sequence_repeat_mask='softmasked_FASTA_lowercase_acgt', repeat_mask_window=window, ml_transform='ML/255',
                quality_transform='1/(Q+1)', window_mode='genomic', orientation='RNA_5to3',
                class_names=list(CLASSES), max_reads=300, sampling_seed=42)


class InferenceDecisionTest(unittest.TestCase):
    def test_rescue_is_separate_from_neural_decision_and_disabled_for_ivt(self):
        self.assertEqual(site_decision(.1, .5, True),
                         dict(nn_keep=0, rescued=1, final_keep=1, final_reason='reference_rescue'))
        self.assertEqual(site_decision(None, .5, True),
                         dict(nn_keep='', rescued=1, final_keep=1, final_reason='reference_rescue'))
        self.assertEqual(site_decision(.5, .5, True)['final_reason'], 'nn_and_reference')
        self.assertEqual(site_decision(.1, .5, True, 'IVT')['final_keep'], 0)
        self.assertEqual(site_decision(.9, .5, True, 'IVT')['final_reason'], 'nn_pass')
        self.assertEqual(site_decision(None, .5, False)['final_reason'], 'no_prediction')
        for score in (math.nan, math.inf, -0.1, 1.1):
            with self.assertRaises(ValueError):
                site_decision(score, .5, False)

    def fixture(self, root):
        specs = [('RMBase3.m6A', 'database', '*', 'm6A', [(20, '+'), (40, '+'), (50, '+'), (60, '-')]),
                 ('HeLa.GLORI', 'short_read', 'HeLa', 'm6A', [(30, '+'), (40, '+')]),
                 ('HEK293T.GLORI', 'short_read', 'HEK293T', 'm6A', [(70, '+')]),
                 ('RMBase3.pseU', 'database', '*', 'pseU', [(60, '+')])]
        sources = []
        for i, (name, kind, cell, modification, sites) in enumerate(specs):
            bed = root / f'reference_{i}.bed'
            bed.write_text(''.join(f'chr1\t{p}\t{p+1}\tx\t0\t{strand}\n' for p, strand in sites))
            source = dict(id=name, kind=kind, cell=cell, modification=modification, assembly='GRCh38',
                          coordinates='BED0', path=bed.name)
            if kind == 'database':
                source.update(database='RMBase', version='3.0')
            sources.append(source)
        sources.append(dict(id='HeLa.unavailable', kind='short_read', cell='HeLa', modification='m6A',
                            assembly='GRCh38', coordinates='BED0', status='unavailable'))
        catalog_path = root / 'catalog.json'
        catalog_path.write_text(json.dumps(dict(assembly='GRCh38', sources=sources)))
        index = root / 'sites.tsv'
        rows = []
        for pos in range(10, 80, 10):
            row = dict(row_id=f'sample|chr1:{pos}:+:m6A', sample_id='sample', chrom='chr1', start=pos,
                       end=pos+1, strand='+', modification='m6A', label='', label_reason='', af=.3,
                       features_dir='features', read_offset=pos, retained_depth=0 if pos == 50 else 2,
                       eligible_depth=0 if pos == 50 else 2, reference_sequence='A' * 19)
            rows.append(row)
        with index.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, INDEX_FIELDS, delimiter='\t')
            writer.writeheader(); writer.writerows(rows)
        return index, ReferenceCatalog(catalog_path), rows

    def test_all_candidates_output_exact_cell_strand_modification_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index, catalog, rows = self.fixture(root)
            scores = {row['row_id']: .9 if row['start'] in (10, 40) else .1 for row in rows if row['start'] != 50}
            write_predictions(index, scores, root / 'out', catalog, 'HeLa', 'WT', .5)
            with (root / 'out/all_predictions.tsv').open() as handle:
                predictions = {int(r['start']): r for r in csv.DictReader(handle, delimiter='\t')}
            self.assertEqual(len(predictions), 7)
            self.assertEqual(predictions[20]['rescued'], '1')
            self.assertEqual(predictions[30]['short_read_sources'], 'HeLa.GLORI')
            self.assertEqual(predictions[40]['final_reason'], 'nn_and_reference')
            self.assertEqual(predictions[40]['rescued'], '0')
            self.assertEqual(predictions[50]['prediction_status'], 'no_reads')
            self.assertEqual(predictions[50]['keep_score'], '')
            self.assertEqual(predictions[50]['nn_keep'], '')
            self.assertEqual(predictions[50]['final_keep'], '1')
            self.assertEqual(predictions[60]['final_keep'], '0')  # Wrong strand / different mod cannot rescue.
            self.assertEqual(predictions[70]['final_keep'], '0')  # Wrong cell cannot rescue.
            with (root / 'out/summary.tsv').open() as handle:
                summary = next(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(summary['input_sites'], '7')
            self.assertEqual(summary['predicted_sites'], '6')
            self.assertEqual(summary['reference_overlap_sites'], '4')
            self.assertEqual(summary['rescued_sites'], '3')
            self.assertEqual(summary['final_kept_sites'], '5')
            self.assertEqual(len((root / 'out/kept_sites.bed').read_text().splitlines()), 5)
            with (root / 'out/source_overlap_counts.tsv').open() as handle:
                sources = {r['source_id']: r for r in csv.DictReader(handle, delimiter='\t')}
            self.assertEqual(sources['HeLa.unavailable']['overlap_sites'], '')
            self.assertNotIn('HEK293T.GLORI', sources)
            self.assertFalse((root / 'out/INCOMPLETE.txt').exists())
            write_predictions(index, scores, root / 'ivt', catalog, 'HeLa', 'IVT', .5)
            with (root / 'ivt/summary.tsv').open() as handle:
                ivt = next(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(ivt['rescued_sites'], '0')
            self.assertEqual(ivt['final_kept_sites'], '2')

    def test_schema_rejects_cv_model_channel_order_and_transform_changes(self):
        run = dict(role='final_all_labeled_sites', target='adenine',
                   model=dict(target='adenine', read_channels=len(CHANNEL_NAMES), window=19, max_reads=300, sampling_seed=42), input_schema=input_schema())
        validate_input_schema(run)
        for key, value in [('quality_transform', '1/Q'), ('window_mode', 'transcript'),
                           ('modification_channels', list(reversed(MODIFICATIONS))),
                           ('feature_set', 'sequence_reads_and_site_features_v0'),
                           ('feature_set', 'sequence_reads_fixed_v1'),
                           ('splice_distance_cap', 100000),
                           ('splice_missing_distance', 0),
                           ('read_channel_names', list(reversed(CHANNEL_NAMES)))]:
            changed = dict(run, input_schema=dict(run['input_schema'], **{key: value}))
            with self.assertRaisesRegex(ValueError, 'schema mismatch'):
                validate_input_schema(changed)
        with self.assertRaisesRegex(ValueError, 'final all-labeled-sites'):
            validate_input_schema(dict(run, role='cross_validation'))
        legacy = dict(run, input_schema={key: value for key, value in run['input_schema'].items()
                                        if key != 'feature_set'})
        with self.assertRaisesRegex(ValueError, 'schema mismatch'):
            validate_input_schema(legacy)
        for encoding in ('class_family_flags', 'family_embedding_and_repeat_mask'):
            old = dict(run, input_schema=dict(run['input_schema'], repeat_encoding=encoding))
            with self.assertRaisesRegex(ValueError, 'schema mismatch'):
                validate_input_schema(old)

    def test_unknown_score_ids_leave_failed_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index, catalog, _ = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'unknown candidate'):
                write_predictions(index, {'unknown': .8}, root / 'out', catalog, 'HeLa', 'WT', .5)
            self.assertTrue((root / 'out/FAILED.txt').exists())
            self.assertFalse((root / 'out/run.json').exists())


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow not installed')
class FinalBundleTest(unittest.TestCase):
    def save_bundle(self, root, position_annotation=None):
        from utils.nn.keras_site import ModelConfig, build_site_model
        config = ModelConfig(target='adenine', max_reads=300)
        root.mkdir(parents=True, exist_ok=True)
        model_path = root / 'model.keras'
        build_site_model(config).save(model_path)
        run = dict(status='complete', role='final_all_labeled_sites', target='adenine', model=asdict(config),
                   threshold=.5, model_file=model_path.name, model_sha256=sha256(model_path),
                   input_schema=input_schema())
        run['position_annotation'] = position_annotation
        (root / 'run.json').write_text(json.dumps(run))
        return run

    def test_load_final_bundle_checks_saved_shapes_and_checksums(self):
        import tensorflow as tf
        from prediction.predict import _load_model
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save_bundle(root)
            model, _, loaded = _load_model(root)
            self.assertEqual(tuple(model.output_shape), (None, 3))
            self.assertEqual(loaded['threshold'], .5)
            self.assertEqual({tensor.name for tensor in model.inputs},
                             {'sequence_tokens', 'read_features', 'read_mask', 'sequence_repeat_mask'})
            self.assertFalse((root / 'repeat_vocabulary.json').exists())
            with (root / 'model.keras').open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                _load_model(root)
        tf.keras.backend.clear_session()



if __name__ == '__main__':
    unittest.main()
