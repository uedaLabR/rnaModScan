#!/usr/bin/env bash
# Fresh integer Parquet; four CPU workers, up to 16 groups / 10 Mb per fetch.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
exec "$PROJECT_ROOT/.venv_cuda112/bin/python" \
  "$PROJECT_ROOT/prediction/test/00_run_sample.py" \
  --config "$PROJECT_ROOT/prediction/config_hela_wt_v2.yaml" --background "$@"
