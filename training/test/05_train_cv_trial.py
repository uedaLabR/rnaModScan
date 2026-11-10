"""Start all five learning-rate trial folds and automatic validation summaries."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.cv_trial import main

if __name__ == '__main__':
    raise SystemExit(main())
