"""Prepare attached Cell Genomics Table S1 as positive-evidence BEDs (stdlib)."""
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.references.genome import IndexedFasta
from utils.references.catalog import sha256

SOURCE = ROOT / 'data/reference_sites_GRCh38/pus7/mmc2.xlsx'
OUTPUT = ROOT / 'data/reference_sites_GRCh38/pus7'
FASTA = Path('/share/reference/hg38.fa')
CATALOG = ROOT / 'data/references/catalog.json'
PAPER_URL = 'https://www.cell.com/cell-genomics/fulltext/S2666-979X(26)00194-1'
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def read_sheet(archive, sheet_name):
    """Read cached scalar cells; never evaluate workbook formulas or external links."""
    strings_xml = ET.fromstring(archive.read('xl/sharedStrings.xml'))
    strings = [''.join(node.itertext()) for node in strings_xml.findall('m:si', NS)]
    workbook = ET.fromstring(archive.read('xl/workbook.xml'))
    relationships = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
    targets = {node.get('Id'): node.get('Target') for node in relationships}
    relation_key = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
    sheet = next(node for node in workbook.findall('m:sheets/m:sheet', NS)
                 if node.get('name') == sheet_name)
    target = targets[sheet.get(relation_key)]
    member = target.lstrip('/') if target.startswith('/') else 'xl/' + target
    document = ET.fromstring(archive.read(member))
    rows = []
    for row in document.findall('m:sheetData/m:row', NS):
        cells = {}
        for cell in row.findall('m:c', NS):
            column = ''.join(c for c in cell.get('r') if c.isalpha())
            value = cell.find('m:v', NS)
            if cell.get('t') == 'inlineStr':
                cells[column] = ''.join(cell.find('m:is', NS).itertext())
            elif value is not None:
                cells[column] = strings[int(value.text)] if cell.get('t') == 's' else value.text
        rows.append((int(row.get('r')), cells))
    headers = rows[0][1]
    return [(number, {headers[column]: value for column, value in cells.items()})
            for number, cells in rows[1:] if cells]


def load_reference_sets():
    """Only load Ψ references; intersections flag potential training/rescue leakage."""
    references = {}
    config = json.loads(CATALOG.read_text())
    for source in config['sources']:
        if source['modification'] != 'pseU' or source.get('status', 'ready') != 'ready':
            continue
        path = (CATALOG.parent / source['path']).resolve()
        sites = set()
        with path.open() as handle:
            for line in handle:
                if not line.strip() or line.startswith(('#', 'track', 'browser')):
                    continue
                fields = line.split()
                sites.add((fields[0], int(fields[1]), fields[5]))
        references[source['id']] = sites
    return references


def prepare():
    outputs = [OUTPUT / name for name in ('S1A.bed', 'S1B.bed', 'union.bed', 'audit.tsv', 'provenance.json')]
    for path in outputs:
        if path.exists():
            raise FileExistsError(path)
    references = load_reference_sets()
    fasta = IndexedFasta(FASTA)
    sets = {}
    audit = []
    try:
        with zipfile.ZipFile(SOURCE) as archive:
            for sheet in ('S1A', 'S1B'):
                sites = set()
                for row_number, row in read_sheet(archive, sheet):
                    dependent = row.get('PUS7_dependent', row.get('PUS7.dependent'))
                    if dependent != 'Yes':
                        raise ValueError(f'Unexpected dependence at {sheet}:{row_number}')
                    chrom, start, strand = row['chr'], int(row['pos']) - 1, row['strand']
                    if start < 0 or strand not in ('+', '-'):
                        raise ValueError(f'Invalid locus at {sheet}:{row_number}')
                    motif = fasta.motif(chrom, start, strand).replace('T', 'U')
                    if motif != row['motif'] or motif[2] != 'U':
                        raise ValueError(f'GRCh38 motif mismatch at {sheet}:{row_number}')
                    key = chrom, start, strand
                    sites.add(key)
                    matches = [name for name, reference in references.items() if key in reference]
                    audit.append(dict(sheet=sheet, excel_row=row_number, chrom=chrom,
                                      source_pos_1based=row['pos'], start=start, end=start+1,
                                      strand=strand, gene=row['gene'], motif=motif,
                                      pus7_dependent=dependent, existing_reference_sources=';'.join(matches),
                                      source_row_json=json.dumps(row, ensure_ascii=False, sort_keys=True)))
                sets[sheet] = sites
    finally:
        fasta.close()
    sets['union'] = sets['S1A'] | sets['S1B']
    for name, sites in sets.items():
        with (OUTPUT / f'{name}.bed').open('w') as handle:
            for chrom, start, strand in sorted(sites):
                handle.write(f'{chrom}\t{start}\t{start+1}\tpseU_PUS7_{name}\t0\t{strand}\n')
    with (OUTPUT / 'audit.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(audit)
    provenance = dict(paper_url=PAPER_URL, source_file=SOURCE.name, source_sha256=sha256(SOURCE),
                      cell='HepG2', modification='pseU', role='positive_evidence',
                      assembly='GRCh38', coordinates='BED0',
                      coordinate_evidence='All source motifs match strand-oriented hg38 at pos-1; pos fails all rows.',
                      fasta=str(FASTA), fasta_index_sha256=sha256(str(FASTA)+'.fai'),
                      catalog_sha256=sha256(CATALOG),
                      counts={name: len(sites) for name, sites in sets.items()},
                      shared_sites=len(sets['S1A'] & sets['S1B']),
                      existing_reference_overlap={name: len(sets['union'] & sites)
                                                  for name, sites in references.items()},
                      limitations=[
                          'HepG2 evidence is not cell-matched to HEK293T or HeLa.',
                          'S1A is direct RNA mismatch/calibration evidence; S1B uses Dorado.',
                          'PUS7 dependence does not mean every row was independently chemically validated.',
                          'Used as positive evidence for HEK293T/HeLa by explicit user request; source cell remains HepG2.',
                          'Not an independent holdout unless overlapping training groups and rescue evidence are excluded.',
                          'Non-overlap or absent coverage is not negative evidence.'])
    provenance['bed_sha256'] = {name: sha256(OUTPUT / f'{name}.bed') for name in sets}
    (OUTPUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    prepare()
