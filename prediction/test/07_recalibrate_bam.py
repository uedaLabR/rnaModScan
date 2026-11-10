"""Write a new BAM with rejected target ML bytes set to zero."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rnamodscan.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['predict', '--stage', 'bam', *sys.argv[1:]]))
