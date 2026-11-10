#!/usr/bin/env bash
# Reuse completed HeLa inputs; measure only the public CPU inference stage.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_ROOT"
exec "$PROJECT_ROOT/.venv_cuda112/bin/python" \
  "$PROJECT_ROOT/analysis/runtime/benchmark_inference.py" "$@"
