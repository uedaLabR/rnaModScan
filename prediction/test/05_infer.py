"""Read prepared Parquet inputs and write neural scores only."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rnamodscan.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['predict', '--stage', 'infer', *sys.argv[1:]]))
