"""Summarize saved non-DRACH sequences by distance from the DRACH consensus.

Only the previous motif-analysis TSVs are read. No genome, BAM, NN model or
prediction is opened or changed.
"""
import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.io.excel import verify_workbook, write_workbook

GROUPS = ('GLORI_all', 'NN_all', 'GLORI_only', 'NN_only', 'shared')
ALLOWED = ('AGT', 'AG', 'A', 'C', 'ACT')
MOTIFS = tuple(''.join(bases) for bases in itertools.product(*ALLOWED))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tsv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def write_tsv(path, headers, rows):
    with Path(path).open('x', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(headers)
        writer.writerows(rows)


def site_groups(row):
    glori, nn = row['GLORI'] == '1', row['NN'] == '1'
    if glori:
        yield 'GLORI_all'
    if nn:
        yield 'NN_all'
    yield 'shared' if glori and nn else 'GLORI_only' if glori else 'NN_only'


def distances_from_sequences(path):
    """Validate stored distances by two independently expressed definitions."""
    counts = Counter()
    seen = set()
    rows = read_tsv(path)
    for row in rows:
        key = (row['chrom'], row['start'], row['strand'])
        if key in seen:
            raise ValueError('Duplicate saved genomic site')
        seen.add(key)
        kmer = row['central_5mer']
        if len(kmer) != 5 or set(kmer) - set('ACGT') or kmer[2] != 'A':
            raise ValueError('Expected a complete candidate-centered reference-A 5-mer')
        consensus_distance = sum(base not in allowed for base, allowed in zip(kmer, ALLOWED))
        enumerated_distance = min(sum(a != b for a, b in zip(kmer, motif)) for motif in MOTIFS)
        if consensus_distance != enumerated_distance or consensus_distance != int(row['DRACH_distance']):
            raise ValueError('Stored distance differs from independent recomputation')
        if not 1 <= consensus_distance <= 4:
            raise ValueError('Non-DRACH reference-A sites must have distances 1 to 4')
        for group in site_groups(row):
            counts[group, consensus_distance] += 1
    return counts, len(rows)


def summarize(counts):
    rows = []
    for group in GROUPS:
        n = sum(counts[group, distance] for distance in range(6))
        one, two, three, four = (counts[group, distance] for distance in range(1, 5))
        rows.append([group, n, one, 100 * one / n, two, 100 * two / n,
                     three, 100 * three / n, four, 100 * four / n,
                     one + two, 100 * (one + two) / n,
                     three + four, 100 * (three + four) / n])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'paper/data/hela_non_drach_motifs_20261108')
    parser.add_argument('--mettl16', type=Path, default=ROOT / 'paper/data/mettl16_screen_20261109/table_S19.tsv')
    parser.add_argument('--output', type=Path, default=ROOT / 'paper/data/drach_distance_extension_20261110')
    args = parser.parse_args()
    source_files = [args.source / 'site_sequences.tsv', args.source / 'drach_distance.tsv', args.mettl16]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_files}
    if args.output.exists():
        raise FileExistsError('Preserve an existing result folder; choose another --output')
    counts, n_unique = distances_from_sequences(source_files[0])
    previous = read_tsv(source_files[1])
    for row in previous:
        if counts[row['group'], int(row['DRACH_distance'])] != int(row['count']):
            raise ValueError('Per-site count differs from original distance summary')
    summaries = summarize(counts)
    headers = ['group', 'sites', 'distance_1_count', 'distance_1_percent',
               'distance_2_count', 'distance_2_percent', 'distance_3_count',
               'distance_3_percent', 'distance_4_count', 'distance_4_percent',
               'distance_at_most_2_count', 'distance_at_most_2_percent',
               'distance_at_least_3_count', 'distance_at_least_3_percent']
    distance_headers = ['group', 'DRACH_distance', 'count', 'known_5mer_sites', 'fraction']
    distance_rows = [[row['group'], int(row['DRACH_distance']), int(row['count']),
                      int(row['known_5mer_sites']), float(row['fraction'])] for row in previous]
    mettl16 = read_tsv(source_files[2])
    mettl16_headers = list(mettl16[0])
    mettl16_rows = [[row[key] if key == 'group' else float(row[key]) if 'percent' in key
                     else int(row[key]) for key in mettl16_headers] for row in mettl16]
    definitions = [
        ['Scope', 'HeLa non-DRACH m6A site sets, raw NN before reference/DRACH rescue; saved sequence TSVs only.'],
        ['Distance', 'Minimum Hamming substitutions in the central genomic-strand-oriented 5-mer to any allowed DRACH motif.'],
        ['IUPAC', 'DNA alphabet: D=A/G/T; R=A/G; central A; C; H=A/C/T. RNA uses U for T.'],
        ['Maximum distance', 'The center is reference A, so four remaining positions can violate DRACH: maximum distance 4, not 3.'],
        ['Cumulative at most 2', 'Distance 1 plus distance 2. Distance 0 is absent because the input was selected as non-DRACH.'],
        ['At least 3', 'Distance 3 plus distance 4; this is a sequence-distance category, not proof of a distinct biological motif.'],
        ['Denominator', 'All sites in the stated group have complete A/C/G/T central 5-mers. GLORI_all and NN_all overlap; only/shared subsets partition them.'],
        ['Verification', 'Every saved site distance recomputed by IUPAC membership and explicit minimum over all 18 DRACH 5-mers, then original count TSV checked.'],
        ['Interpretation', 'Descriptive sequence composition only. No matched null, enrichment test, independent chemical accuracy or RNA-structure analysis.'],
        ['METTL16', 'Previous exact canonical UACAGAGAA and expanded UACAGARAA nonamer matches copied unchanged for numerical context. Candidate A is position 4.'],
        ['METTL16 limitation', 'No match in NN-retained sites was observed. No formal enrichment test was performed; no absence-of-enrichment or enzyme-specificity claim follows.'],
    ]
    sheets = [dict(name='Definitions', headers=['item', 'definition'], rows=definitions, widths=[27, 100]),
              dict(name='Distance_summary', headers=headers, rows=summaries),
              dict(name='Distance_counts', headers=distance_headers, rows=distance_rows),
              dict(name='METTL16_previous_S19', headers=mettl16_headers, rows=mettl16_rows)]
    args.output.mkdir(parents=True)
    for sheet in sheets:
        write_tsv(args.output / (sheet['name'] + '.tsv'), sheet['headers'], sheet['rows'])
    book = args.output / 'HeLa_DRACH_distance_extension_20261110.xlsx'
    write_workbook(book, sheets, float_format='0.0000')
    restored = verify_workbook(book)
    if any(restored[sheet['name']] != [sheet['headers'], *sheet['rows']] for sheet in sheets):
        raise ValueError('Workbook native cells did not roundtrip exactly')
    for path in source_files:
        if sha256(path) != hashes[str(path.relative_to(ROOT))]:
            raise ValueError('Source changed during read-only analysis')
    report = dict(status='complete', source_sha256=hashes, source_files_unchanged=True,
                  unique_site_sequences_verified=n_unique, allowed_DRACH_5mers=len(MOTIFS),
                  summaries=[dict(zip(headers, row)) for row in summaries],
                  maximum_distance_with_central_A=4, no_statistical_enrichment_test=True,
                  models_inputs_old_predictions_unchanged=True, workbook=book.name,
                  workbook_sha256=sha256(book), native_cells_roundtrip_exact=True,
                  generator_sha256=sha256(Path(__file__)))
    (args.output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'README.txt').write_text(
        'Read-only extension of existing HeLa non-DRACH motif summaries, 2026-11-10.\n'
        'Distance_summary.tsv and the native Excel workbook give distances 1/2/3/4 and cumulative at-most-2/at-least-3 counts.\n'
        'The central A is fixed, but all four flanking positions can depart from the degenerate DRACH consensus.\n'
        'A distance of at least 3 is a sequence-distance category, not evidence for an entirely separate biological motif.\n'
        'METTL16_previous_S19 copies the previous sequence screen. Zero NN matches is not a formal absence-of-enrichment test.\n'
        'Original source SHA256s and all 56,979 per-site distances were verified; no BAM/genome/model or existing output changed.\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
