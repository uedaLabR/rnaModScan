"""Screen saved, strand-oriented non-DRACH contexts for METTL16 nonamers."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.io.excel import write_workbook, verify_workbook

SOURCE = ROOT / 'paper/data/hela_non_drach_motifs_20261108/site_sequences.tsv'
OUTPUT = ROOT / 'paper/data/mettl16_screen_20261109'
MOTIFS = {'canonical': {'UACAGAGAA'}, 'UACAGARAA': {'UACAGAGAA', 'UACAGAAAA'}}
GROUPS = ('GLORI_all', 'NN_all', 'GLORI_only', 'NN_only', 'shared')


def write_rows(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def groups(row):
    glori, nn = row['GLORI'] == '1', row['NN'] == '1'
    return ([name for name, flag in [('GLORI_all', glori), ('NN_all', nn)] if flag]
            + ['shared' if glori and nn else 'GLORI_only' if glori else 'NN_only'])


def main():
    if (OUTPUT / 'run.json').exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with SOURCE.open() as handle:
        records = list(csv.DictReader(handle, delimiter='\t'))
    for row in records:
        sequence = row['rna_11nt']
        if len(sequence) != 11 or sequence[5] != 'A':
            raise ValueError('Expected an RNA-oriented, A-centred 11nt window')
        # The fourth base of UAC[m6A]GARAA aligns to the candidate at index5.
        row['candidate_nonamer_RNA'] = sequence[2:11]
    summary, hits = [], []
    for group in GROUPS:
        selected = [row for row in records if group in groups(row)]
        complete = [row for row in selected if all(base in 'ACGU' for base in row['candidate_nonamer_RNA'])]
        for name, motifs in MOTIFS.items():
            found = [row for row in complete if row['candidate_nonamer_RNA'] in motifs]
            summary.append(dict(group=group, motif=name, total_sites=len(selected),
                complete_nonamers=len(complete), motif_sites=len(found),
                motif_percent=100*len(found)/len(complete) if complete else None,
                motif_sites_with_recorded_DRS_depth_at_least_10=sum(
                    bool(row['dorado_depth']) and int(row['dorado_depth']) >= 10 for row in found),
                motif_sites_with_saved_NN_score=sum(bool(row['nn_score']) for row in found)))
    for row in records:
        if row['candidate_nonamer_RNA'] in MOTIFS['UACAGARAA']:
            hits.append({**row, 'canonical_nonamer': int(row['candidate_nonamer_RNA'] == 'UACAGAGAA')})
    write_rows(OUTPUT / 'summary.tsv', summary, list(summary[0]))
    write_rows(OUTPUT / 'motif_sites.tsv', hits, list(records[0]) + ['canonical_nonamer'])
    methods = [
        ['Scope', 'HeLa non-DRACH GLORI and original-target-qualified raw NN site sets; no rescue'],
        ['Coordinates', 'Saved genome BED0 coordinates; negative-strand contexts already reverse complemented'],
        ['Exact motif', 'UAC[m6A]GAGAA, with candidate at the fourth nucleotide'],
        ['Expanded motif', 'UAC[m6A]GARAA, R=A/G; includes the reported MAT2A variant'],
        ['Denominator', 'Complete candidate-aligned 9nt windows within each non-DRACH set'],
        ['Interpretation', 'Sequence screen only; secondary structure and METTL16 dependence were not assessed'],
        ['Coverage', 'Recorded DRS depth and saved NN score are distinct audit fields; absent depth is not zero coverage'],
        ['Primary source', 'Pendleton et al. 2017, https://doi.org/10.1016/j.cell.2017.05.003'],
        ['Expanded sequence source', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10578476/'],
    ]
    sheets = [
        dict(name='Summary', headers=list(summary[0]), rows=[[row[key] for key in summary[0]] for row in summary]),
        dict(name='Motif_sites', headers=list(hits[0]), rows=[[row[key] for key in hits[0]] for row in hits]),
        dict(name='Definitions', headers=['Item', 'Definition'], rows=methods),
    ]
    book = OUTPUT / 'METTL16_sequence_screen_20261109.xlsx'
    write_workbook(book, sheets)
    restored = verify_workbook(book, sheet_names=[sheet['name'] for sheet in sheets])
    if restored != {sheet['name']: [sheet['headers']] + sheet['rows'] for sheet in sheets}:
        raise ValueError('Workbook cell roundtrip mismatch')
    run = dict(status='complete', source=str(SOURCE.relative_to(ROOT)),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        source_rows=len(records), unique_motif_hit_sites=len(hits), summary=summary,
        scope='Descriptive sequence screen; no BAM scan, NN inference or model training',
        worksheet_names=[sheet['name'] for sheet in sheets], workbook_sha256=hashlib.sha256(book.read_bytes()).hexdigest())
    (OUTPUT / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    (OUTPUT / 'README.txt').write_text('\n'.join(name + ': ' + value for name, value in methods) + '\n')
    print(json.dumps(run, indent=2))


def append_dorado_table():
    """Add the saved pre-filter screen without scanning any sequence or BAM again."""
    source = ROOT / 'paper/data/hela_feature_patterns_20261109/mettl16_centered_motif_summary.json'
    counts = json.loads(source.read_text())
    run_path = OUTPUT / 'run.json'
    run = json.loads(run_path.read_text())
    if 'dorado_table_added' in run:
        raise FileExistsError('Dorado table already added')
    book = OUTPUT / 'METTL16_sequence_screen_20261109.xlsx'
    original = verify_workbook(book)
    summary = run['summary']
    table = [dict(group='Dorado before filtering', sites=counts['denominator'],
        canonical_sites=counts['exact_UACAGAGAA'], expanded_sites=counts['relaxed_UACAGARAA'])]
    for group in GROUPS:
        exact = next(row for row in summary if row['group'] == group and row['motif'] == 'canonical')
        expanded = next(row for row in summary if row['group'] == group and row['motif'] == 'UACAGARAA')
        table.append(dict(group=group, sites=exact['total_sites'],
            canonical_sites=exact['motif_sites'], expanded_sites=expanded['motif_sites']))
    for row in table:
        row['canonical_percent'] = 100 * row['canonical_sites'] / row['sites']
        row['expanded_percent'] = 100 * row['expanded_sites'] / row['sites']
    sheets = [dict(name=name, headers=rows[0], rows=rows[1:]) for name, rows in original.items()]
    sheets.append(dict(name='TableS19', headers=list(table[0]),
        rows=[[row[key] for key in table[0]] for row in table]))
    archive = OUTPUT / 'archive_initial_GLORI_NN'
    archive.mkdir(exist_ok=False)
    book.rename(archive / book.name)
    (archive / 'run_before_dorado_table.json').write_text(json.dumps(run, indent=2) + '\n')
    write_workbook(book, sheets)
    restored = verify_workbook(book)
    expected = {sheet['name']: [sheet['headers']] + sheet['rows'] for sheet in sheets}
    if restored != expected:
        raise ValueError('Dorado-extended workbook roundtrip mismatch')
    run['dorado_table_added'] = dict(source=str(source.relative_to(ROOT)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), rows=table,
        scope='Saved candidate screen; no additional BAM, sequence or Parquet scan')
    run['worksheet_names'] = list(restored)
    run['workbook_sha256'] = hashlib.sha256(book.read_bytes()).hexdigest()
    run['initial_workbook_preserved'] = str((archive / book.name).relative_to(ROOT))
    run_path.write_text(json.dumps(run, indent=2) + '\n')
    write_rows(OUTPUT / 'table_S19.tsv', table, list(table[0]))
    print(json.dumps(run['dorado_table_added'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--append-dorado', action='store_true')
    if parser.parse_args().append_dorado:
        append_dorado_table()
    else:
        main()
