"""Streaming, strand-specific candidate filtering; no training labels assigned."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

from utils.io.bed import TARGET_CODES as MODS, bedmethyl_rows, parse_bedmethyl_row as parse_row
FIELDS = ['sample_id', 'chrom', 'start', 'end', 'strand', 'modification',
          'mod_code', 'total_reads', 'modified_reads', 'af', 'valid_coverage',
          'canonical_reads', 'other_mod_reads', 'deletions', 'failed_calls',
          'different_base_reads', 'no_calls']


def validate_options(min_depth, min_modified, min_af, min_ml):
    if min_depth < 1 or min_modified < 1:
        raise ValueError('Read count thresholds must be positive')
    if not math.isfinite(min_af) or not 0 <= min_af <= 1:
        raise ValueError('AF must be between 0 and 1')
    if not 0 <= min_ml <= 255:
        raise ValueError('ML must be between 0 and 255')



def filter_bedmethyl(source, out, sample, min_depth=10, min_modified=3,
                     min_af=0.1, min_ml=125, provenance=None):
    validate_options(min_depth, min_modified, min_af, min_ml)
    if not sample or any(x.isspace() for x in sample):
        raise ValueError('sample_id must be nonempty and contain no whitespace')
    source, out = Path(source), Path(out)
    if not source.is_file():
        raise FileNotFoundError(source)
    out.mkdir(parents=True, exist_ok=False)
    counts, observed = Counter(), Counter()
    try:
        with (out/'candidates.tsv').open('w', newline='', encoding='utf-8') as tsv, \
                (out/'candidates.bed').open('w', encoding='utf-8') as bed:
            writer = csv.DictWriter(tsv, fieldnames=FIELDS, delimiter='\t')
            writer.writeheader()
            for row in bedmethyl_rows(source):
                observed[row['mod_code']] += 1
                counts['input_rows'] += 1
                if row['modification'] is None:
                    counts['unsupported_mod_rows'] += 1
                    continue
                if (row['total_reads'] < min_depth or row['modified_reads'] < min_modified
                        or row['af'] < min_af):
                    counts['filtered_rows'] += 1
                    continue
                counts[row['modification']] += 1
                counts['candidate_rows'] += 1
                row['sample_id'] = sample
                writer.writerow(row)
                bed.write(f"{row['chrom']}\t{row['start']}\t{row['end']}\t{sample}|{row['modification']}\t0\t{row['strand']}\n")
        if not counts['input_rows']:
            raise ValueError('No bedMethyl data rows; check modkit output')
        if not set(observed).intersection(MODS):
            raise ValueError('No supported modification codes present')
        report = dict(sample_id=sample, source=str(source.resolve()),
                      thresholds=dict(min_total_reads=min_depth, min_modified_reads=min_modified,
                                      min_af=min_af, min_ml=min_ml, probability=min_ml/256),
                      af_definition='N_mod / (N_valid_cov + N_fail + N_diff + N_nocall)',
                      call_definition='modkit winning-state calls, not all raw ML >= threshold',
                      quality_verification='run by wrapper' if provenance else 'user asserted upstream threshold; bedMethyl has no per-read ML',
                      counts=dict(counts), observed_codes=dict(observed),
                      missing_modifications=[v for k,v in MODS.items() if k not in observed],
                      provenance=provenance)
        (out/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return report
    except Exception as exc:
        (out/'FAILED.txt').write_text(str(exc), encoding='utf-8')
        raise

