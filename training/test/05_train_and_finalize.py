"""Run after 00_prepare.py has completed, using the same YAML and output_dir."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.run_training import main

CONFIG_PATH = ROOT / 'training/config.yaml'
STAGES = ('fit', 'final')

if __name__ == '__main__':
    main(CONFIG_PATH, STAGES)
