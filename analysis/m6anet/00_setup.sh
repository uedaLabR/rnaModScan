#!/usr/bin/env bash
# Install isolated CPU tools only when the user explicitly runs this file.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG_FILE=${M6ANET_CONFIG:-$PROJECT_ROOT/analysis/m6anet/config.sh}
source "$CONFIG_FILE"
SETUP_MODE=${1:-all}
case "$SETUP_MODE" in
  all|--python-only) ;;
  *) echo 'Usage: bash analysis/m6anet/00_setup.sh [--python-only]' >&2; exit 2 ;;
esac
trap 'echo "Setup stopped at line $LINENO. Fix the reported error, then rerun this setup." >&2' ERR
CONDA_EXE=${CONDA_EXE:-/home/ueda/anaconda3/bin/conda}
if ! command -v "$CONDA_EXE" >/dev/null 2>&1; then
  echo 'conda is required for this setup. Use your Miniforge/Miniconda conda path as CONDA_EXE.' >&2
  echo 'Example: CONDA_EXE=/path/to/miniforge3/bin/conda bash analysis/m6anet/00_setup.sh' >&2
  exit 1
fi
ensure_conda_environment() {
  local environment=$1
  shift
  local action=create
  if [[ -f "$environment/conda-meta/history" ]]; then
    action=install
  elif [[ -e "$environment" ]]; then
    echo "Not a conda environment: $environment. Choose a new M6ANET_ENV path in config.sh." >&2
    return 1
  fi
  "$CONDA_EXE" "$action" -y -p "$environment" --override-channels -c conda-forge -c bioconda "$@"
}
if [[ "$SETUP_MODE" == all ]]; then
  ensure_conda_environment "$TOOLS_ENV" \
    'python=3.10' 'samtools=1.20' 'minimap2=2.28' 'gffread=0.12.7' 'f5c=1.5' 'slow5tools=0.8.0'
elif [[ ! -x "$TOOLS_ENV/bin/python" ]]; then
  echo 'Missing tools environment; run setup without --python-only first.' >&2
  exit 1
fi
ensure_conda_environment "$M6ANET_ENV" 'python=3.8' pip
"$M6ANET_ENV/bin/python" -m pip install --upgrade pip
"$M6ANET_ENV/bin/python" -m pip install 'numpy==1.23.5' 'pandas==1.5.3' 'setuptools<81'
"$M6ANET_ENV/bin/python" -m pip install 'torch==1.6.0+cpu' -f https://download.pytorch.org/whl/torch_stable.html
"$M6ANET_ENV/bin/python" -m pip install 'm6anet==2.1.0'
"$TOOLS_ENV/bin/python" -m venv "$BLUECRAB_ENV"
"$BLUECRAB_ENV/bin/python" -m pip install --upgrade pip
"$BLUECRAB_ENV/bin/python" -m pip install blue-crab
"$M6ANET_ENV/bin/python" -m pip check
"$BLUECRAB_ENV/bin/python" -m pip check
"$M6ANET_ENV/bin/m6anet" inference --help
"$BLUECRAB_ENV/bin/blue-crab" p2s --help

"$CONDA_EXE" list -p "$TOOLS_ENV" --explicit > "$PROJECT_ROOT/analysis/m6anet/tools_explicit.txt"
echo 'Setup complete. Run: bash analysis/m6anet/01_run_hela.sh --check'
