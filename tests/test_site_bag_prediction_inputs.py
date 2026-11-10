"""Prepared inference snapshots common Parquet without rewriting neural features."""
import csv
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from prediction.site_bag_inputs import prepare_site_bag_prediction_inputs
from utils.adenine import CLASSES
from utils.features.index import INDEX_FIELDS
from utils.features.site_dataset import LABEL_COLUMNS
from utils.nn.inputs import CHANNEL_NAMES, MODIFICATIONS
from utils.references.catalog import sha256


class SiteBagPredictionInputsTest(unittest.TestCase):
    def fixture(self, root, *, bad_training_flag=False):
        source = root / 'source'
        index = root / 'index/sites.tsv'
        index.parent.mkdir(parents=True)
        tensor_type = pa.list_(pa.list_(pa.list_(pa.float32(), 18), 19))
        schema = pa.schema([
            ('row_id', pa.string()), ('chrom', pa.string()), ('start', pa.int64()),
            ('strand', pa.string()), ('retained_depth', pa.int64()),
            ('sequence_tokens', pa.list_(pa.int32(), 17)),
            ('sequence_repeat_mask', pa.list_(pa.bool_(), 19)), ('read_features', tensor_type),
            ('training_class', pa.string()), ('training_label_m6A', pa.bool_()),
            ('training_label_inosine', pa.bool_()), ('training_label_false_positive', pa.bool_()),
        ], metadata={b'schema_version': b'site_bags_parquet_v3'})
        specifications = [('chr1', '+', (1, 0)), ('chr1', '+', (2,)), ('chr2', '-', (0,))]
        rows, files = [], []
        number = 0
        for file_number, (chrom, strand, depths) in enumerate(specifications):
            relative = Path(f'chrom={chrom}/strand={"plus" if strand == "+" else "minus"}/part-{file_number:06d}.parquet')
            path = source / 'sites' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            records = []
            for offset, depth in enumerate(depths):
                row_id = f'sample|{chrom}:{100 + number * 100}:{strand}:adenine'
                record = dict(row_id=row_id, chrom=chrom, start=100 + number * 100, strand=strand,
                              retained_depth=depth, sequence_tokens=[number] * 17,
                              sequence_repeat_mask=[position % 3 == 0 for position in range(19)],
                              read_features=np.full((depth, 19, 18), number / 10, np.float32).tolist(),
                              **dict.fromkeys(LABEL_COLUMNS))
                if bad_training_flag and number == 0:
                    record.update(training_class='m6A', training_label_m6A=True,
                                  training_label_inosine=False, training_label_false_positive=False)
                records.append(record)
                row = dict.fromkeys(INDEX_FIELDS, '')
                row.update(row_id=row_id, sample_id='sample', chrom=chrom, start=record['start'],
                           end=record['start'] + 1, strand=strand, modification='adenine',
                           reference_sequence='A' * 19, features_dir=os.path.relpath(source, index.parent),
                           retained_depth=depth, eligible_depth=depth * 5,
                           site_parquet=str(Path('sites') / relative), site_row_group=0, site_row_offset=offset,
                           af_m6A=.25, call_count_m6A=3, call_count_inosine=0,
                           **dict.fromkeys(LABEL_COLUMNS))
                rows.append(row)
                number += 1
            pq.write_table(pa.Table.from_pylist(records, schema=schema), path)
            files.append(dict(path=str(relative), rows=len(records), size_bytes=path.stat().st_size))
        source_run = dict(status='complete', schema='site_bags_parquet_v3', sample_id='sample',
                          mode='inference', max_reads=300, sampling_seed=42, window_flank=9,
                          parquet_files=dict(sites=files))
        (source / 'run.json').write_text(json.dumps(source_run))
        fields = list(dict.fromkeys([*INDEX_FIELDS, 'site_parquet', 'site_row_group', 'site_row_offset', *LABEL_COLUMNS]))
        self.write_index(index, fields, rows)
        index_run = dict(status='complete', schema='site_bag_index_v3', target='adenine', require_labels=False,
                         feature_sources=[dict(path=str(source), run_sha256=sha256(source / 'run.json'))],
                         max_reads=300, sampling_seed=42, index_sha256=sha256(index))
        (index.parent / 'run.json').write_text(json.dumps(index_run))
        model = root / 'model'
        model.mkdir()
        (model / 'model.keras').write_bytes(b'weights must not be opened during CPU preparation')
        schema = dict(feature_set='sequence_reads_repeat18_v1', missing_ml='zero', window=19,
                      modification_channels=list(MODIFICATIONS), read_channel_names=list(CHANNEL_NAMES),
                      splice_distance='same_read_CIGAR_N_query_bases_adjacent_exon_zero',
                      splice_distance_cap=1000, splice_missing_distance=1000,
                      splice_distance_transform='log1p(min(d,1000))/log1p(1000)', sequence_repeat_mask='softmasked_FASTA_lowercase_acgt', repeat_mask_window=19, ml_transform='ML/255',
                      quality_transform='1/(Q+1)', window_mode='genomic', orientation='RNA_5to3',
                      class_names=list(CLASSES), max_reads=300, sampling_seed=42)
        model_run = dict(status='complete', role='final_all_labeled_sites', target='adenine', threshold=.5,
                         model_file='model.keras', model_sha256=sha256(model / 'model.keras'),
                         model=dict(target='adenine', window=19, read_channels=18, max_reads=300, sampling_seed=42),
                         input_schema=schema)
        (model / 'run.json').write_text(json.dumps(model_run))
        return index, model, source, fields, rows

    def write_index(self, path, fields, rows):
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fields, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)

    def test_snapshot_multiple_shards_zero_read_and_policy_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, model, source, fields, rows = self.fixture(root)
            policy = root / 'policy.tsv'
            policy_row = dict(rows[-1], row_id='policy', chrom='chr3', strand='+', filter_mode='reference_only')
            self.write_index(policy, fields, [policy_row])
            out = root / 'prepared'
            with patch('utils.features.site_dataset.SiteBagBatches._batch', side_effect=AssertionError('Tensor rewrite')):
                run = prepare_site_bag_prediction_inputs(index, model, out, policy_sites=policy)
            self.assertEqual(run['layout'], 'site_bags_parquet_v3')
            self.assertEqual(sum(part['site_count'] for part in run['partitions']), 5)
            self.assertEqual(sum(part['input_count'] for part in run['partitions']), 2)
            parts = {(part['chrom'], part['strand']): part for part in run['partitions']}
            self.assertEqual(len(parts['chr1', '+']['input_shards']), 2)
            self.assertEqual(parts['chr2', '-']['input_shards'], [])
            self.assertEqual(parts['chr3', '+']['input_shards'], [])
            for shard in parts['chr1', '+']['input_shards']:
                path = out / shard['path']
                original = Path(shard['source_path'])
                self.assertEqual(path.read_bytes(), original.read_bytes())
                self.assertEqual(path.stat().st_ino, original.stat().st_ino)
                self.assertEqual(shard['snapshot_method'], 'hardlink')
            metadata = pq.ParquetFile(out / parts['chr1', '+']['sites_path']).read().to_pylist()
            self.assertEqual(metadata[0]['af_m6A'], .25)
            self.assertEqual(metadata[0]['call_count_inosine'], 0)
            shutil.rmtree(source)
            shutil.rmtree(index.parent)
            # The GPU worker needs only prepared files, despite removed sources.
            linked = out / parts['chr1', '+']['input_shards'][0]['path']
            self.assertEqual(pq.ParquetFile(linked).metadata.num_rows, 2)
            self.assertFalse((out / 'INCOMPLETE.txt').exists())

    def test_cross_filesystem_fallback_copies_the_exact_existing_parquet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, model, _, _, _ = self.fixture(root)
            with patch('prediction.site_bag_inputs.os.link', side_effect=OSError(errno.EXDEV, 'Cross-device link')):
                run = prepare_site_bag_prediction_inputs(index, model, root / 'prepared')
            for part in run['partitions']:
                for shard in part['input_shards']:
                    original, saved = Path(shard['source_path']), root / 'prepared' / shard['path']
                    self.assertEqual(shard['snapshot_method'], 'copy')
                    self.assertNotEqual(original.stat().st_ino, saved.stat().st_ino)
                    self.assertEqual(original.read_bytes(), saved.read_bytes())

    def test_nonnull_training_flags_and_unindexed_source_sites_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, model, _, _, _ = self.fixture(root / 'bad', bad_training_flag=True)
            with self.assertRaisesRegex(ValueError, 'null training labels'):
                prepare_site_bag_prediction_inputs(index, model, root / 'bad_output')
            self.assertTrue((root / 'bad_output/FAILED.txt').is_file())
            index, model, _, fields, rows = self.fixture(root / 'subset')
            self.write_index(index, fields, [rows[0], *rows[2:]])  # Remove zero row from a copied shard.
            run_path = index.parent / 'run.json'
            run = json.loads(run_path.read_text())
            run['index_sha256'] = sha256(index)
            run_path.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, 'absent from the prediction index'):
                prepare_site_bag_prediction_inputs(index, model, root / 'subset_output')

    def test_cpu_snapshot_does_not_import_tensorflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, model, _, _, _ = self.fixture(root)
            code = """
import sys
from prediction.site_bag_inputs import prepare_site_bag_prediction_inputs
prepare_site_bag_prediction_inputs(sys.argv[1], sys.argv[2], sys.argv[3])
assert 'tensorflow' not in sys.modules
"""
            result = subprocess.run([sys.executable, '-B', '-c', code, str(index), str(model), str(root / 'out')],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
