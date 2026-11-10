"""Scalar fields shared by per-site Parquet indices and prediction reports."""
from utils.adenine import FIELDS as ADENINE_FIELDS
from utils.io.artifacts import completed_run
from utils.io.bed import CALL_COUNT_FIELDS

INDEX_FIELDS = ['row_id', 'sample_id', 'chrom', 'start', 'end', 'strand', 'modification',
                'label', 'label_reason', 'af', 'features_dir', 'read_offset', 'retained_depth',
                'eligible_depth', 'reference_sequence']
INDEX_FIELDS += ['total_reads', 'modified_reads', *CALL_COUNT_FIELDS]
INDEX_FIELDS += ['filter_mode', 'motif', 'motif_group']
INDEX_FIELDS += ADENINE_FIELDS
