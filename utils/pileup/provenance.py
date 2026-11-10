"""Read the actual upstream filter settings from a modkit pileup log."""
import math
import re
from pathlib import Path
from utils.references.catalog import sha256


def read_pileup_log(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig')
    command = re.search(r'command line: (.+)', text)
    rows = re.search(r'Done, processed ([0-9,]+) rows', text)
    filters = {}
    for probability, quality, base, percentile in re.findall(
            r'setting threshold ([0-9.eE+-]+) \(qual: (\d+)\) for base ([ACGT]), percentile ([0-9.]+)', text):
        p = float(probability)
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f'Invalid probability in {path}')
        filters[base] = dict(probability=p, logged_quality=int(quality), percentile=float(percentile))
    return dict(path=str(path.resolve()), sha256=sha256(path),
                command=command[1] if command else None,
                filter_mode='sampled_per_base' if filters else 'not_resolved_from_log',
                base_thresholds=filters,
                completed_rows=int(rows[1].replace(',', '')) if rows else None,
                per_read_ml_verified=False)
