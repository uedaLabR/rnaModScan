"""Small current-format site bags for training and prediction integration tests."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from utils.adenine import CLASSES
from utils.features.site_bags import build_site_bag_index


def prepare_training_bags(root, high_depth=False):
    root = Path(root)
    directories = []
    schema = pa.schema([
        ('row_id', pa.string()), ('sample_id', pa.string()), ('chrom', pa.string()),
        ('start', pa.int64()), ('end', pa.int64()), ('strand', pa.string()),
        ('modification', pa.string()), ('label_status', pa.string()),
        ('reference_sequence', pa.string()), ('eligible_depth', pa.int64()),
        ('retained_depth', pa.int64()), ('read_offset', pa.int64()),
        ('af_m6A', pa.float64()), ('af_inosine', pa.float64()),
        ('training_class', pa.string()), ('training_label_m6A', pa.bool_()),
        ('training_label_inosine', pa.bool_()), ('training_label_false_positive', pa.bool_()),
        ('sequence_tokens', pa.list_(pa.int32(), 17)),
        ('sequence_repeat_mask', pa.list_(pa.bool_(), 19)),
        ('read_features', pa.list_(pa.list_(pa.list_(pa.float32(), 18), 19))),
    ], metadata={b'schema_version': b'site_bags_parquet_v3',
                 b'feature_set': b'sequence_reads_repeat18_v1', b'orientation': b'RNA_5to3'})
    for sample in ('HEK293T_WT', 'HEK293T_IVT', 'HeLa_WT', 'HeLa_IVT'):
        directory = root / 'features' / sample
        path = directory / 'sites/chrom=chr1/strand=plus/part-000000.parquet'
        path.parent.mkdir(parents=True)
        records = []
        for index, start in enumerate(range(100, 1300, 100)):
            label = 2 if sample.endswith('IVT') else 0 if sample.startswith('HeLa') else index % 2
            retained = 300 if high_depth and sample == 'HEK293T_WT' and index == 0 else 2
            eligible = 601 if retained == 300 else retained
            features = np.zeros((retained, 19, 18), np.float32)
            features[:, :, 8] = 1 / 31
            features[:, :, 10] = 1
            features[:, :, 16:] = 1
            features[:, 9, 0] = (40 if label == 2 else 240) / 255
            record = dict(row_id=f'{sample}|chr1:{start}:+:adenine', sample_id=sample,
                chrom='chr1', start=start, end=start + 1, strand='+', modification='adenine',
                label_status='final', reference_sequence='A' * 19, eligible_depth=eligible,
                retained_depth=retained, read_offset=0, af_m6A=.3, af_inosine=.2,
                training_class=CLASSES[label], sequence_tokens=[0] * 17,
                sequence_repeat_mask=[position % 3 == 0 for position in range(19)],
                read_features=features.tolist())
            record.update({'training_label_' + name: name == CLASSES[label] for name in CLASSES})
            records.append(record)
        pq.write_table(pa.Table.from_pylist(records, schema=schema), path, row_group_size=3)
        run = dict(status='complete', schema='site_bags_parquet_v3',
            feature_set='sequence_reads_repeat18_v1', sample_id=sample, max_reads=300,
            sampling_seed=42, window_flank=9, parquet_files=dict(sites=[dict(
                path=str(path.relative_to(directory / 'sites')), rows=len(records), size_bytes=path.stat().st_size)]))
        (directory / 'run.json').write_text(json.dumps(run))
        directories.append(directory)
    build_site_bag_index(directories, root / 'index')
    return root / 'index/sites.tsv'
