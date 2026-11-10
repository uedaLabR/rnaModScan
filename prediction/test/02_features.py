"""Run the predict features step with a YAML configuration."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rnamodscan.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['predict', '--stage', 'features', *sys.argv[1:]]))
