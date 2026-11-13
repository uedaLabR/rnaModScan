"""Return original bedMethyl rows that pass any independent evidence flag."""
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import sqlite3

from prediction.bam_recalibration import prediction_tables
from prediction.evidence import FLAG_FIELDS, evidence_values
from utils.io.artifacts import file_info
from utils.io.bed import open_text, parse_bedmethyl_row
from utils.references.catalog import sha256

APPENDED_FIELDS = ('flag_db', 'flag_short_read', 'flag_nn', 'motif_keep', 'filter_pass')
MOTIF_KEEP_ENCODING = {'0': 'none', '1': 'DRACH', '2': 'canonical METTL16 UAC[m6A]GAGAA'}


def motif_keep_code(drach, mettl16):
    """Encode the mutually exclusive motif-retention rules in one BED column."""
    if drach not in (0, 1) or mettl16 not in (0, 1):
        raise ValueError('Motif retention flags must be binary')
    if drach and mettl16:
        raise ValueError('DRACH and canonical METTL16 motif flags are mutually exclusive')
    return drach + 2 * mettl16


def write_bed_header(handle):
    """BED comment lines describe appended columns after any source columns."""
    handle.write('# rnaModScan filtered bedMethyl: original input columns are retained in their original order.\n')
    handle.write('# appended_columns\t' + '\t'.join(APPENDED_FIELDS) + '\n')
    handle.write('# flag_db, flag_short_read, flag_nn: 0=absent, 1=present; filter_pass=1 for every exported row.\n')
    handle.write('# motif_keep: 0=none; 1=DRACH; 2=canonical METTL16 UAC[m6A]GAGAA (candidate A is nonamer position 4).\n')


def index_pass_flags(database, directories, sample_id):
    """Keep full-genome decisions on disk; only one chromosome enters RAM."""
    database.execute('CREATE TABLE flags (chrom TEXT, start INTEGER, strand TEXT, target TEXT, '
                     'db INTEGER, short INTEGER, nn INTEGER, drach INTEGER, mettl16 INTEGER, '
                     'PRIMARY KEY(chrom,start,strand,target))')
    sources, targets = [], set()
    for directory in directories:
        for path, partition in prediction_tables(directory):
            sources.append(dict(**file_info(path), sha256=sha256(path)))
            with path.open(newline='') as handle:
                for row in csv.DictReader(handle, delimiter='\t'):
                    if row['sample_id'] != sample_id:
                        continue
                    targets.add(row['modification'])
                    if partition and (row['chrom'], row['strand']) != partition:
                        raise ValueError('Decision row disagrees with its partition')
                    values = evidence_values(row)
                    motif_keep_code(int(values[3]), int(values[4]))
                    passed = int('1' in values)
                    if row['final_keep'] != str(passed) or row['filter_pass'] != str(passed):
                        raise ValueError('PASS must equal OR of all evidence flags')
                    if passed:
                        database.execute('INSERT INTO flags VALUES (?,?,?,?,?,?,?,?,?)',
                            (row['chrom'], int(row['start']), row['strand'], row['modification'], *map(int, values)))
    database.commit()
    return sources, sorted(targets)


def export_filtered_bed(bedmethyl, directories, sample_id, out):
    """Append three evidence flags, a motif code and PASS to original BED rows."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE.txt').write_text('Filtered bedMethyl export not complete.\n')
    database_path = out / 'flags.sqlite'
    database = sqlite3.connect(str(database_path))
    source = file_info(bedmethyl)
    try:
        sources, targets = index_pass_flags(database, directories, sample_id)
        counts = {target: 0 for target in targets}
        with ExitStack() as stack:
            combined = stack.enter_context((out / 'filtered.bed').open('w'))
            write_bed_header(combined)
            handles = {}
            for target in targets:
                (out / target).mkdir()
                handles[target] = stack.enter_context((out / target / 'filtered.bed').open('w'))
                write_bed_header(handles[target])
            chrom, flags = None, {}
            finished = set()
            with open_text(bedmethyl) as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip() or line.startswith(('#', 'track', 'browser')):
                        continue
                    columns = line.split()
                    # Additional source columns are retained verbatim as values.
                    row = parse_bedmethyl_row('\t'.join(columns[:18]), number)
                    if row['chrom'] != chrom:
                        if flags:
                            raise ValueError('Retained predictions are absent from the source BED')
                        if row['chrom'] in finished:
                            raise ValueError('Source BED chromosome blocks must be contiguous')
                        if chrom is not None:
                            finished.add(chrom)
                        chrom = row['chrom']
                        flags = {(start, strand, target): values for start, strand, target, *values in
                                 database.execute('SELECT start,strand,target,db,short,nn,drach,mettl16 '
                                                  'FROM flags WHERE chrom=?', (chrom,))}
                    key = row['start'], row['strand'], row['modification']
                    values = flags.pop(key, None)
                    if values is None:
                        continue
                    original = line.rstrip('\r\n') if '\t' in line else '\t'.join(columns)
                    db, short, nn, drach, mettl16 = values
                    bed_values = [db, short, nn, motif_keep_code(drach, mettl16), 1]
                    output = original + '\t' + '\t'.join(map(str, bed_values)) + '\n'
                    combined.write(output)
                    handles[row['modification']].write(output)
                    counts[row['modification']] += 1
            expected = database.execute('SELECT COUNT(*) FROM flags').fetchone()[0]
            if flags or sum(counts.values()) != expected:
                raise ValueError('Retained predictions are absent from the source BED')
        if file_info(bedmethyl) != source:
            raise ValueError('Source BED changed during export')
        report = dict(status='complete', schema='filtered_bedmethyl_v3', sample_id=sample_id,
                      source_bed=source, prediction_sources=sources, counts=counts,
                      appended_columns=list(APPENDED_FIELDS), motif_keep_encoding=MOTIF_KEEP_ENCODING,
                      internal_evidence_flags=list(FLAG_FIELDS),
                      policy='Original candidate rows only; PASS = DB OR short-read OR NN OR DRACH OR canonical METTL16 motif; WT reference flags only')
        (out / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as error:
        (out / 'FAILED.txt').write_text(str(error) + '\n')
        raise
    finally:
        database.close()
        database_path.unlink(missing_ok=True)
