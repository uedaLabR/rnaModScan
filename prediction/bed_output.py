"""Return original bedMethyl rows that pass any independent evidence flag."""
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import sqlite3

from prediction.bam_recalibration import prediction_tables
from prediction.evidence import FLAG_FIELDS, evidence_values
from prediction.motifs import MOTIF_ENCODING, motif_keep_code
from prediction.auxiliary import (AUXILIARY_TARGETS, auxiliary_motif_eligible,
                                  validate_auxiliary_motif_thresholds)
from utils.io.artifacts import file_info
from utils.io.bed import open_text, parse_bedmethyl_row
from utils.references.catalog import sha256

APPENDED_FIELDS = ('flag_db', 'flag_short_read', 'flag_nn', 'motif_keep', 'filter_pass')
MOTIF_KEEP_ENCODING = MOTIF_ENCODING


def write_bed_header(handle, *, auxiliary_motif_min_depth=20, auxiliary_motif_min_af=.40):
    """BED comment lines describe appended columns after any source columns."""
    validate_auxiliary_motif_thresholds(auxiliary_motif_min_depth, auxiliary_motif_min_af)
    handle.write('# rnaModScan filtered bedMethyl: original input columns are retained in their original order.\n')
    handle.write('# appended_columns\t' + '\t'.join(APPENDED_FIELDS) + '\n')
    handle.write('# flag_db, flag_short_read, flag_nn: 0=absent, 1=present; filter_pass=1 for every exported row.\n')
    handle.write('# motif_keep: ' + '; '.join(code+'='+name for code, name in MOTIF_KEEP_ENCODING.items()) + '.\n')
    handle.write('# motif targets: DRACH central A; METTL16 nonamer A at position 4; NSUN6/NSUN2 first C; TRUB1/PUS7 candidate U at position 3. N=A/C/G/U; R=A/G.\n')
    handle.write('# Motif flags identify enabled sequence rules, not proof of modification, RNA structure or the responsible enzyme; original candidate thresholds still apply.\n')
    handle.write(f'# auxiliary motif thresholds (m5C/pseU motif_keep=3..6 only): depth>={auxiliary_motif_min_depth}; N_mod>=3; AF>={auxiliary_motif_min_af!r}; AF=N_mod/(valid+fail+diff+nocall).\n')
    handle.write('# Known-site m5C/pseU flags use the original candidate thresholds (defaults: depth>=10, N_mod>=3, AF>=0.10), independently of the stronger motif rule. m6A/inosine thresholds are unchanged.\n')


def index_pass_flags(database, directories, sample_id):
    """Keep full-genome decisions on disk; only one chromosome enters RAM."""
    database.execute('CREATE TABLE flags (chrom TEXT, start INTEGER, strand TEXT, target TEXT, '
                     + ', '.join(name+' INTEGER' for name in FLAG_FIELDS) + ', '
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
                    code = motif_keep_code(*map(int, values[3:]))
                    if row.get('motif_keep') not in (None, '', str(code)):
                        raise ValueError('Public motif code disagrees with independent evidence flags')
                    passed = int('1' in values)
                    if row['final_keep'] != str(passed) or row['filter_pass'] != str(passed):
                        raise ValueError('PASS must equal OR of all evidence flags')
                    if passed:
                        database.execute('INSERT INTO flags VALUES ('+','.join('?' for _ in range(4+len(FLAG_FIELDS)))+')',
                            (row['chrom'], int(row['start']), row['strand'], row['modification'], *map(int, values)))
    database.commit()
    return sources, sorted(targets)


def export_filtered_bed(bedmethyl, directories, sample_id, out, *, auxiliary_motif_min_depth=20, auxiliary_motif_min_af=.40):
    """Append three evidence flags, a motif code and PASS to original BED rows."""
    validate_auxiliary_motif_thresholds(auxiliary_motif_min_depth, auxiliary_motif_min_af)
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
            write_bed_header(combined,auxiliary_motif_min_depth=auxiliary_motif_min_depth,auxiliary_motif_min_af=auxiliary_motif_min_af)
            handles = {}
            for target in targets:
                (out / target).mkdir()
                handles[target] = stack.enter_context((out / target / 'filtered.bed').open('w'))
                write_bed_header(handles[target],auxiliary_motif_min_depth=auxiliary_motif_min_depth,auxiliary_motif_min_af=auxiliary_motif_min_af)
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
                                 database.execute('SELECT start,strand,target,'+','.join(FLAG_FIELDS)+' '
                                                  'FROM flags WHERE chrom=?', (chrom,))}
                    key = row['start'], row['strand'], row['modification']
                    values = flags.pop(key, None)
                    if values is None:
                        continue
                    if (row['modification'] in AUXILIARY_TARGETS and motif_keep_code(*values[3:]) >= 3
                            and not auxiliary_motif_eligible(row,auxiliary_motif_min_depth,auxiliary_motif_min_af)):
                        raise ValueError('Retained m5C/psi motif decision violates auxiliary motif thresholds; rerun filtering with the current configuration')
                    original = line.rstrip('\r\n') if '\t' in line else '\t'.join(columns)
                    db, short, nn = values[:3]
                    bed_values = [db, short, nn, motif_keep_code(*values[3:]), 1]
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
                      auxiliary_motif_thresholds=dict(min_depth=auxiliary_motif_min_depth,min_modified=3,min_af=auxiliary_motif_min_af),
                      policy='Original candidate rows only; PASS = DB OR short-read OR NN OR enabled m6A/m5C/pseU motif; WT reference flags only; enabled motif rules also apply to IVT')
        (out / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as error:
        (out / 'FAILED.txt').write_text(str(error) + '\n')
        raise
    finally:
        database.close()
        database_path.unlink(missing_ok=True)
