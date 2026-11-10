"""Read-only verification of uploaded prepared references; no pileup scan."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'analysis/reference_catalog/server_reference_manifest.json'
sys.path.insert(0, str(ROOT))
from utils.references.catalog import ReferenceCatalog, sha256


def check_files(root, manifest):
    root = Path(root).resolve()
    if manifest['schema_version'] != 1 or manifest['assembly'] != 'GRCh38':
        raise ValueError('Unsupported reference bundle')
    covered = set()
    for item in manifest['files']:
        path = (root / item['path']).resolve()
        path.relative_to(root)
        if path in covered:
            raise ValueError(f'Duplicate bundle entry: {path}')
        covered.add(path)
        if not path.is_file():
            raise FileNotFoundError(f'Missing uploaded file: {path}')
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError(f'Checksum mismatch: {path}; upload the matching bundle file')
    catalog_path = (root / manifest['catalog']).resolve()
    catalog_path.relative_to(root)
    if catalog_path not in covered:
        raise ValueError('Catalog is not included in verified files')
    config = json.loads(catalog_path.read_text(encoding='utf-8-sig'))
    for source in config['sources']:
        if source.get('status', 'ready') != 'ready':
            raise ValueError(f"Reference is unavailable: {source['id']}")
        for key in ('path', 'provenance'):
            if source.get(key):
                resolved = (catalog_path.parent / source[key]).resolve()
                if resolved not in covered:
                    raise ValueError(f'Reference dependency is not verified: {resolved}')
    catalog = ReferenceCatalog(catalog_path)
    if len(catalog.sources) != manifest['source_count']:
        raise ValueError('Unexpected reference source count')
    for source in catalog.sources:
        print(f"{source['id']}: {source['unique_sites']:,} sites")
    print(f'OK: {len(catalog.sources)} reference sources; hashes and BED format verified.')
    print('Pileup and genome were not read.')
    return catalog


if __name__ == '__main__':
    check_files(ROOT, json.loads(MANIFEST.read_text(encoding='utf-8')))
