"""Run after all four bedMethyl files are complete; no overlap rerun is needed."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.run_training import main

CONFIG_PATH = ROOT / 'training/config.yaml'
STAGES = ('sites', 'select', 'features', 'annotation', 'split')

if __name__ == '__main__':
    main(CONFIG_PATH, STAGES)
