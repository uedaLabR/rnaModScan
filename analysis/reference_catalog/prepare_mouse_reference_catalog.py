"""One-time mm10 RMBase 3.0 preparation from the bundled source archives."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.references.rmbase import prepare_rmbase3_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'source')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/reference_sites_mm10/rmbase3')
    parser.add_argument('--catalog', type=Path, default=ROOT / 'data/references/catalog.mm10.json')
    args = parser.parse_args()
    report = prepare_rmbase3_catalog(args.source, args.out, args.catalog, genome='mm10')
    for item in report['files']:
        print(f"{item['modification']}: {item['unique_sites']:,} RMBase mm10 sites")
    print(f'Catalog: {args.catalog}')


if __name__ == '__main__':
    main()
