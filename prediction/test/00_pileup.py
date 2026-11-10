"""Run optional modkit pileup using prediction YAML settings."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rnamodscan.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['predict', '--stage', 'pileup', *sys.argv[1:]]))
