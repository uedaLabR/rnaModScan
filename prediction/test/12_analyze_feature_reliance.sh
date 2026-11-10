#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$project_dir/.venv_cuda112/bin/python" -B "$project_dir/analysis/feature_reliance/run.py" "$@"
