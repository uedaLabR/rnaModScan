"""Prepare unlabeled DRS candidates using the same BED reader/AF definition."""
import csv
import json
from collections import Counter
from contextlib import ExitStack
from itertools import groupby
from pathlib import Path

from utils.io.bed import ALL_MODIFICATION_CODES, CALL_COUNT_FIELDS, bedmethyl_rows, passes_thresholds
from utils.references.catalog import MODIFICATIONS
from utils.pileup.candidates import validate_options
from utils.pileup.provenance import read_pileup_log
from utils.io.artifacts import file_info
from prediction.confidence import BASE_FIELDS, confidence_fields

FIELDS = ['row_id', 'sample_id', 'chrom', 'start', 'end', 'strand', 'modification',
          'total_reads', 'modified_reads', 'af', 'label', 'label_reason', *CALL_COUNT_FIELDS,
          *BASE_FIELDS]


def candidate_rows_with_call_counts(rows):
    """Join eight winning-state call counts at one coordinate and strand.

    The BED reader validates coordinate order and uniqueness. Only the current
    coordinate is buffered. An absent channel remains unknown, not zero.
    """
    for _, coordinate_rows in groupby(rows, key=lambda row: (row['chrom'], row['start'])):
        coordinate_rows = list(coordinate_rows)
        counts_by_strand = {}
        for row in coordinate_rows:
            modification = ALL_MODIFICATION_CODES.get(row['mod_code'])
            if modification is not None:
                counts = counts_by_strand.setdefault(row['strand'], {})
                counts['call_count_' + modification] = row['modified_reads']
        for row in coordinate_rows:
            if row['modification'] is not None:
                yield dict(row, **counts_by_strand.get(row['strand'], {}))


def prepare_prediction_sites(bedmethyl, pileup_log, sample_id, out, thresholds, flank=9):
    """Threshold-existing modkit calls only. Do not create candidates from a DB."""
    if not sample_id or any(c.isspace() for c in sample_id):
        raise ValueError('Use a nonempty sample ID without whitespace')
    if not thresholds or any(m not in MODIFICATIONS for m in thresholds):
        raise ValueError('Specify thresholds for supported target modifications')
    for setting in thresholds.values():
        validate_options(setting['min_depth'], setting['min_modified'], setting['min_af'], 0)
    source = file_info(bedmethyl)
    upstream = read_pileup_log(pileup_log)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE.txt').write_text('Candidate preparation not complete.', encoding='utf-8')
    counts = Counter()
    try:
        with ExitStack() as stack:
            writers = {}
            for target in thresholds:
                directory = out / target
                directory.mkdir()
                handle = stack.enter_context((directory / 'sites.tsv').open('w', encoding='utf-8', newline=''))
                writer = csv.DictWriter(handle, FIELDS, delimiter='\t')
                writer.writeheader()
                writers[target] = writer
            def counted_rows():
                for row in bedmethyl_rows(bedmethyl):
                    counts['input_rows'] += 1
                    yield row

            for row in candidate_rows_with_call_counts(counted_rows()):
                target = row['modification']
                if target not in thresholds or not passes_thresholds(row, **thresholds[target]):
                    continue
                record = {key: row[key] for key in FIELDS if key in row}
                record.update(confidence_fields(row))
                record.update(row_id=f"{sample_id}|{row['chrom']}:{row['start']}:{row['strand']}:{target}",
                              sample_id=sample_id, label='', label_reason='unlabeled_prediction_candidate')
                writers[target].writerow(record)
                counts[target] += 1
        if not counts['input_rows'] or file_info(bedmethyl) != source:
            raise ValueError('Input BED is empty or changed while being read')
        with (out / 'candidate_counts.tsv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['sample_id', 'modification', 'candidate_sites'])
            writer.writerows((sample_id, target, counts[target]) for target in thresholds)
        report = dict(status='complete', sample_id=sample_id, source=source, upstream=upstream,
                      thresholds=thresholds, window_flank=flank, counts=dict(counts),
                      labels='none; external overlap is applied only AFTER independent NN scoring',
                      af_definitions=dict(af='N_mod / (N_valid_cov + N_fail + N_diff + N_nocall); unchanged candidate/NN AF',
                                          af_pass='N_mod / N_valid_cov; confidence-passing coverage; audit only'))
        (out / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as exc:
        (out / 'FAILED.txt').write_text(str(exc), encoding='utf-8')
        raise
