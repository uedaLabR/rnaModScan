"""Combine per-target decision tables without reading any BAM or Parquet."""
import csv
import json
from pathlib import Path

from utils.references.catalog import sha256
from utils.features.index import completed_run


def combine_predictions(directories, out, *, include_sites=True):
    directories, out = [Path(p) for p in directories], Path(out)
    runs = [completed_run(path) for path in directories]
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE.txt').write_text('Summary not complete.', encoding='utf-8')
    try:
        filenames = ['summary.tsv', 'decision_counts.tsv', 'source_overlap_counts.tsv', 'motif_summary.tsv']
        if include_sites:
            filenames.insert(0, 'all_predictions.tsv')
            filenames.insert(1, 'kept_sites.tsv')
        for filename in filenames:
            with (out / filename).open('w', encoding='utf-8', newline='') as dest:
                writer, columns = None, None
                for directory in directories:
                    with (directory / filename).open(encoding='utf-8', newline='') as source:
                        reader = csv.DictReader(source, delimiter='\t')
                        if writer is None:
                            columns = reader.fieldnames
                            writer = csv.DictWriter(dest, columns, delimiter='\t')
                            writer.writeheader()
                        if reader.fieldnames != columns:
                            raise ValueError('Prediction output columns differ')
                        writer.writerows(reader)
        if include_sites:
            with (out / 'kept_sites.bed').open('w', encoding='utf-8') as dest:
                for directory in directories:
                    with (directory / 'kept_sites.bed').open(encoding='utf-8') as source:
                        for line in source:
                            dest.write(line)
        report = dict(status='complete', sources=[dict(path=str(p.resolve()), run_sha256=sha256(p / 'run.json'))
                                                  for p in directories],
                      include_sites=include_sites,
                      interpretation='NN filtering and reference rescue counts; not experimental accuracy')
        (out / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as exc:
        (out / 'FAILED.txt').write_text(str(exc), encoding='utf-8')
        raise
