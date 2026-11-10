"""Train/evaluate adenine fold 0 only; reuse existing feature Parquet and splits."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.fold_trial import main

if __name__ == '__main__':
    raise SystemExit(main())
