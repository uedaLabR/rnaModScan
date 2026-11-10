"""Shared metadata for one A-site and its two candidate modification types."""
import csv
import json
import sqlite3
from heapq import merge
from itertools import groupby
from pathlib import Path

TARGET = 'adenine'
MODIFICATIONS = ('m6A', 'inosine')
CLASSES = ('m6A', 'inosine', 'false_positive')
FIELDS = ['feature_row_id']
for modification in MODIFICATIONS:
    FIELDS += [f'{name}_{modification}' for name in
               ('source_row_id', 'candidate', 'af', 'total_reads', 'modified_reads', 'label',
                'short_read_sources', 'database_sources')]


def site_key(row):
    return row['chrom'], int(row['start']), row['sample_id'], row['strand']


def sorted_rows(path):
    previous = None
    with Path(path).open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            key = site_key(row)
            if previous is not None and key <= previous:
                raise ValueError('Adenine inputs must have unique sites sorted by chromosome/position/sample/strand')
            previous = key
            yield row


def disk_sorted_rows(paths, database_path):
    """Prediction BED contigs need not be in lexical chromosome order."""
    database_path = Path(database_path)
    database = sqlite3.connect(str(database_path))
    try:
        database.execute('CREATE TABLE rows (chrom TEXT, start INTEGER, sample TEXT, strand TEXT, target TEXT, payload TEXT, '
                         'UNIQUE(chrom,start,sample,strand,target))')
        for path in paths:
            with Path(path).open(encoding='utf-8', newline='') as handle:
                batch = []
                for row in csv.DictReader(handle, delimiter='\t'):
                    batch.append((*site_key(row), row['modification'], json.dumps(row)))
                    if len(batch) >= 10000:
                        database.executemany('INSERT INTO rows VALUES (?,?,?,?,?,?)', batch)
                        batch.clear()
                database.executemany('INSERT INTO rows VALUES (?,?,?,?,?,?)', batch)
        database.commit()
        for result in database.execute('SELECT payload FROM rows ORDER BY chrom,start,sample,strand,target'):
            yield json.loads(result[0])
    finally:
        database.close()
        database_path.unlink()


def merged_sites(paths, *, sort_database=None):
    """Merge sorted target tables, buffering only one site at a time."""
    if sort_database is None:
        streams = [sorted_rows(path) for path in paths]
        ordered = merge(*streams, key=site_key)
    else:
        ordered = disk_sorted_rows(paths, sort_database)
    for _, rows in groupby(ordered, key=site_key):
        rows = list(rows)
        by_target = {row['modification']: row for row in rows}
        if len(by_target) != len(rows) or any(target not in MODIFICATIONS for target in by_target):
            raise ValueError('Expected at most one m6A and one inosine record per sample/site')
        yield by_target


def combine_site(rows):
    """Preserve each original AF, call count, label and source ID separately."""
    first = next(iter(rows.values()))
    result = dict(first)
    result.update(row_id=f"{first['sample_id']}|{first['chrom']}:{first['start']}:{first['strand']}:{TARGET}",
                  modification=TARGET, feature_row_id='')
    for target in MODIFICATIONS:
        row = rows.get(target, {})
        result[f'source_row_id_{target}'] = row.get('row_id', '')
        result[f'candidate_{target}'] = int(bool(row))
        for name in ('af', 'total_reads', 'modified_reads', 'label', 'short_read_sources', 'database_sources'):
            result[f'{name}_{target}'] = row.get(name, '')
    result['af'] = max(float(row['af']) for row in rows.values())
    return result
