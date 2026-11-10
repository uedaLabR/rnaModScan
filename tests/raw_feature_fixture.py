"""Read tiny raw-output fixtures for CIGAR/MM audits; never used by pipelines."""
import json
from pathlib import Path

import pyarrow.parquet as pq


class RawReadAudit:
    """Materialize synthetic fixture rows in their recorded logical order."""
    def __init__(self, directory):
        directory = Path(directory)
        run = json.loads((directory / 'run.json').read_text())
        files = {item['path']: pq.ParquetFile(directory / 'reads' / item['path']).read().to_pylist()
                 for item in run['parquet_files']['reads']}
        self.data = []
        if run.get('read_ranges') is not None:
            for entry in run['read_ranges']:
                start = entry['file_offset']
                self.data.extend(files[entry['path']][start:start + entry['rows']])
        else:
            for item in run['parquet_files']['reads']:
                self.data.extend(files[item['path']])

    def rows(self, offset, count, row_id):
        selected = self.data[offset:offset + count]
        if len(selected) != count or any(row['row_id'] != row_id for row in selected):
            raise AssertionError('Synthetic raw audit offsets disagree with the site ID')
        return selected
