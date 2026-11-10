"""Create HEK293T train/validation plus a separate HeLa analysis list."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rnamodscan.cli import main

CONFIG_PATH = ROOT / 'training/config.yaml'

if __name__ == '__main__':
    raise SystemExit(main(['train', '--stage', 'cell_dataset', '--config', str(CONFIG_PATH), *sys.argv[1:]]))
