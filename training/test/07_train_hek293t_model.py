"""HEK293T-only model; HeLa is excluded from fitting and model selection."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.cell_trial import main


if __name__ == '__main__':
    raise SystemExit(main())
