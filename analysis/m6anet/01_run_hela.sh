#!/usr/bin/env bash
# Explicit stages; no stage is launched by sourcing this file or by --check.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG_FILE=${M6ANET_CONFIG:-$PROJECT_ROOT/analysis/m6anet/config.sh}
source "$CONFIG_FILE"
export PATH="$TOOLS_ENV/bin:$M6ANET_ENV/bin:$BLUECRAB_ENV/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
STAGE=${1:---check}
TOOLS=(samtools minimap2 gffread slow5tools f5c m6anet blue-crab)
check_inputs() {
  local missing=0 tool path
  for tool in "${TOOLS[@]}"; do
    if command -v "$tool" >/dev/null 2>&1; then
      command -v "$tool"
    else
      echo "Missing tool: $tool" >&2
      case "$tool" in
        m6anet) echo "  Expected: $M6ANET_ENV/bin/m6anet (Python 3.8)" >&2 ;;
        blue-crab) echo "  Expected: $BLUECRAB_ENV/bin/blue-crab (Python >=3.10)" >&2 ;;
      esac
      missing=1
    fi
  done
  for path in "$BAM" "$GENOME" "$ANNOTATION"; do
    if [[ ! -f "$path" ]]; then echo "Missing input: $path" >&2; missing=1; fi
  done
  if [[ ! -d "$POD5_DIR" ]]; then echo "Missing POD5 directory: $POD5_DIR" >&2; missing=1; fi
  echo "Output: $OUT"
  if (( missing )); then
    echo 'If the five BAM/signal tools are installed, finish setup with: bash analysis/m6anet/00_setup.sh --python-only' >&2
  fi
  return "$missing"
}
if [[ "$STAGE" == '--check' ]]; then check_inputs; exit; fi
case "$STAGE" in reference|fastq|signal|align|index|eventalign|dataprep|infer|all) ;; *) echo 'Stages: --check, reference, fastq, signal, align, index, eventalign, dataprep, infer, all' >&2; exit 2;; esac
check_inputs
mkdir -p "$OUT/logs"
exec 9>"$OUT/.workflow.lock"
flock -n 9 || { echo 'Another m6Anet workflow is running.' >&2; exit 1; }
run_stage() {
  local stage=$1
  if [[ -e "$OUT/$stage.complete" ]]; then echo "Already complete: $stage"; return; fi
  echo "Starting $stage at $(date -Is)"
  case "$stage" in
    reference)
      [[ ! -e "$OUT/reference" ]] || { echo 'Partial reference output exists.' >&2; return 1; }
      mkdir "$OUT/reference"
      gffread "$ANNOTATION" -g "$GENOME" -w "$OUT/reference/transcripts.softmasked.fa"
      awk '/^>/ {print;next} {print toupper($0)}' "$OUT/reference/transcripts.softmasked.fa" > "$OUT/reference/transcripts.fa"
      curl -fL --retry 3 https://raw.githubusercontent.com/hasindu2008/f5c/v1.3/test/rna004-models/rna004.nucleotide.5mer.model -o "$OUT/reference/rna004.nucleotide.5mer.model"
      sha256sum "$GENOME" "$ANNOTATION" "$OUT/reference/transcripts.fa" "$OUT/reference/rna004.nucleotide.5mer.model" > "$OUT/reference/sha256.txt"
      ;;
    fastq)
      [[ ! -e "$OUT/reads.fastq" ]] || { echo 'Partial FASTQ exists.' >&2; return 1; }
      samtools fastq -@ 2 -n -F 3840 "$BAM" > "$OUT/reads.fastq"
      ;;
    signal)
      [[ ! -e "$OUT/reads.blow5" ]] || { echo 'Signal output exists without complete marker; inspect it before resuming.' >&2; return 1; }
      "$BLUECRAB_ENV/bin/python" "$PROJECT_ROOT/analysis/m6anet/convert_pod5_files.py" \
        --pod5-dir "$POD5_DIR" --output-dir "$OUT" --workers "$SIGNAL_PROCESSES"
      local signal_temp="$OUT/reads.partial.$$.blow5"
      slow5tools merge "$OUT/blow5_parts_verified" -o "$signal_temp" -t "$THREADS"
      slow5tools quickcheck "$signal_temp"
      mv "$signal_temp" "$OUT/reads.blow5"
      ;;
    align)
      [[ -e "$OUT/reference.complete" && -e "$OUT/fastq.complete" ]]
      [[ ! -e "$OUT/transcriptome.bam" ]] || { echo 'Partial transcriptome BAM exists.' >&2; return 1; }
      minimap2 -ax map-ont -uf --secondary=no -t "$THREADS" "$OUT/reference/transcripts.fa" "$OUT/reads.fastq" \
        | samtools view -u -F 2308 - \
        | samtools sort -@ 2 -o "$OUT/transcriptome.bam" -
      samtools index -@ 2 "$OUT/transcriptome.bam"
      ;;
    index)
      [[ -e "$OUT/signal.complete" && -e "$OUT/fastq.complete" ]]
      f5c index --slow5 "$OUT/reads.blow5" "$OUT/reads.fastq"
      ;;
    eventalign)
      [[ -e "$OUT/align.complete" && -e "$OUT/index.complete" ]]
      [[ ! -e "$OUT/eventalign.tsv" ]] || { echo 'Partial eventalign exists.' >&2; return 1; }
      f5c eventalign --rna --disable-cuda=yes --min-mapq "$MIN_MAPQ" -t "$THREADS" \
        -b "$OUT/transcriptome.bam" -r "$OUT/reads.fastq" -g "$OUT/reference/transcripts.fa" \
        --slow5 "$OUT/reads.blow5" --kmer-model "$OUT/reference/rna004.nucleotide.5mer.model" \
        --signal-index --scale-events --summary "$OUT/eventalign.summary.tsv" -o "$OUT/eventalign.tsv"
      ;;
    dataprep)
      [[ -e "$OUT/eventalign.complete" && ! -e "$OUT/dataprep" ]]
      m6anet dataprep --eventalign "$OUT/eventalign.tsv" --out_dir "$OUT/dataprep" --n_processes "$THREADS"
      ;;
    infer)
      [[ -e "$OUT/dataprep.complete" && ! -e "$OUT/inference" ]]
      m6anet inference --input_dir "$OUT/dataprep" --out_dir "$OUT/inference" \
        --pretrained_model HEK293T_RNA004 --device cpu --seed "$SEED" \
        --n_processes "$THREADS" --num_iterations "$ITERATIONS"
      ;;
  esac
  touch "$OUT/$stage.complete"
  echo "Completed $stage at $(date -Is)"
}
if [[ -e "$OUT/config_snapshot.sh" ]]; then
  cmp -s "$CONFIG_FILE" "$OUT/config_snapshot.sh" || { echo 'Config changed; choose a new output directory.' >&2; exit 1; }
else
  cp "$CONFIG_FILE" "$OUT/config_snapshot.sh"
fi
"$M6ANET_ENV/bin/python" -m pip freeze > "$OUT/logs/m6anet_packages.txt"
"$BLUECRAB_ENV/bin/python" -m pip freeze > "$OUT/logs/bluecrab_packages.txt"
f5c --version > "$OUT/logs/f5c_version.txt" 2>&1
samtools --version > "$OUT/logs/samtools_version.txt" 2>&1
minimap2 --version > "$OUT/logs/minimap2_version.txt" 2>&1
if [[ "$STAGE" == all ]]; then
  for stage in reference fastq signal align index eventalign dataprep infer; do run_stage "$stage" >> "$OUT/logs/$stage.log" 2>&1; done
else
  run_stage "$STAGE" >> "$OUT/logs/$STAGE.log" 2>&1
fi
