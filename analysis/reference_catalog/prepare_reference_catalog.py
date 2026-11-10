"""One-time reference setup; edit SOURCE_DIR then run with Python, no CLI."""
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.references.rmbase import import_rmbase3
from utils.references.catalog import sha256

SOURCE_DIR = ROOT / 'source'
REFERENCE_DIR = ROOT / 'data/references'
RMBASE_DIR = ROOT / 'references/rmbase3'
SHORT_READ_DIR = ROOT / 'data/reference_sites_GRCh38'
CATALOG = REFERENCE_DIR / 'catalog.json'


def short_read_sources():
    sources = []
    for cell in ('HeLa', 'HEK293T'):
        methods = {'m6A': ['GLORI'], 'm5C': ['m5C-TAC-seq'], 'inosine': ['Slic-seq'], 'pseU': ['BID-seq']}
        if cell == 'HeLa':
            methods['inosine'].append('BACS')
            methods['pseU'].append('BACS')
        else:
            methods['pseU'].append('PRAISE')
        for mod, assays in methods.items():
            for method in assays:
                filemod = 'psi' if mod == 'pseU' else mod
                sources.append(dict(id=f'{cell}.{mod}.{method}', kind='short_read', cell=cell,
                                    modification=mod, method=method, assembly='GRCh38', coordinates='BED0',
                                    path=Path(os.path.relpath(SHORT_READ_DIR / 'by_method' / f'{cell}_{filemod}_{method}_GRCh38.bed', REFERENCE_DIR)).as_posix(),
                                    provenance=Path(os.path.relpath(SHORT_READ_DIR / 'sources.json', REFERENCE_DIR)).as_posix()))
    return sources


def pus7_sources():
    """User-approved cross-cell positives; retain actual assay and source cell."""
    directory = ROOT / 'data/reference_sites_GRCh38/pus7'
    sources = []
    for cell in ('HeLa', 'HEK293T'):
        for sheet in ('S1A', 'S1B'):
            sources.append(dict(
                id=f'{cell}.pseU.PUS7_{sheet}', kind='short_read', cell=cell,
                source_cell='HepG2', cross_cell_mapping='explicit_user_request',
                evidence_type='nanopore_direct_rna',
                evidence_bucket_note='short_read is the existing experimental-positive bucket; this source is not short-read',
                modification='pseU', method=f'PUS7_{sheet}',
                assembly='GRCh38', coordinates='BED0',
                path=Path(os.path.relpath(directory / f'{sheet}.bed', REFERENCE_DIR)).as_posix(),
                sha256=sha256(directory / f'{sheet}.bed'),
                provenance=Path(os.path.relpath(directory / 'provenance.json', REFERENCE_DIR)).as_posix()))
    return sources


def prepare():
    if CATALOG.exists():
        raise FileExistsError(CATALOG)
    for source in short_read_sources() + pus7_sources():
        path = REFERENCE_DIR / source['path']
        if not path.is_file():
            raise FileNotFoundError(path)
    if RMBASE_DIR.exists():
        if (RMBASE_DIR / 'FAILED.txt').exists():
            raise ValueError('Incomplete RMBase import; choose a new output directory')
        report = json.loads((RMBASE_DIR / 'import.json').read_text(encoding='utf-8'))
        if report['version'] != '3.0' or report['assembly'] != 'GRCh38':
            raise ValueError('Expected RMBase 3.0 GRCh38')
        for item in report['files']:
            if sha256(RMBASE_DIR / item['bed']) != item['bed_sha256']:
                raise ValueError('RMBase BED checksum mismatch')
    else:
        report = import_rmbase3(SOURCE_DIR, RMBASE_DIR)
    sources = short_read_sources() + pus7_sources()
    for item in report['files']:
        mod = item['modification']
        sources.append(dict(id=f'RMBase3.{mod}', kind='database', database='RMBase', version='3.0',
                            cell='*', modification=mod, assembly='GRCh38', original_assembly='hg38',
                            coordinates='BED0', path=Path(os.path.relpath(RMBASE_DIR / item['bed'], REFERENCE_DIR)).as_posix(),
                            sha256=item['bed_sha256'], provenance=Path(os.path.relpath(RMBASE_DIR / 'import.json', REFERENCE_DIR)).as_posix()))
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    CATALOG.write_text(json.dumps(dict(assembly='GRCh38', sources=sources), indent=2), encoding='utf-8')
    for item in report['files']:
        print(f'{item["modification"]}: {item["unique_sites"]:,} RMBase sites')


if __name__ == '__main__':
    prepare()
