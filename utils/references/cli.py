"""Build/check portable positive-site references without running training."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build', help='Import preselected positive BED/TSV/CSV site lists')
    build.add_argument('--manifest', type=Path, required=True, help='YAML/JSON source descriptions and column mappings')
    build.add_argument('--output', type=Path, required=True, help='New directory for catalog.json, normalized BEDs and provenance')
    check = commands.add_parser('check', help='Check catalog coordinates, sources, files and checksums')
    check.add_argument('--catalog', type=Path, required=True)
    check.add_argument('--training', action='store_true', help='Allow the explicitly declared training assembly/source set')
    args = parser.parse_args(argv)
    try:
        from utils.references.import_catalog import build_catalog, check_catalog
        if args.command == 'build':
            report = build_catalog(args.manifest, args.output)
        else:
            report = check_catalog(args.catalog, for_training=args.training)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
