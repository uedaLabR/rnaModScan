"""Select training sites before expensive BAM feature extraction."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rnamodscan.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['train', '--stage', 'select', *sys.argv[1:]]))
