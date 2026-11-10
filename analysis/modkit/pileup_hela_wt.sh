#!/bin/bash
set -euo pipefail

BAM="/mnt/ssdnas/dorado/riboML/rna004_130bps_sup@v6.0.0/HeLa/PAU09364_pass_c7fa45d4/bam_pass/PAU09364_pass_c7fa45d4.sorted.bam"
REF="/share/reference/hg38.fa"
MODKIT="/home/ueda/RNA004/modScore/modkit/target/release/modkit"
OUT="/home/ueda/RNA004/modScore/result/rna004_130bps_sup_v6.0.0/HeLa_WT_pass_only_modkit"
THREADS=8

[[ -r "$BAM" ]] || { echo "Missing BAM: $BAM" >&2; exit 1; }
[[ -r "$BAM.bai" || -r "${BAM%.bam}.bai" || -r "$BAM.csi" ]] || { echo "Missing BAM index: $BAM" >&2; exit 1; }
[[ -r "$REF" && -r "$REF.fai" ]] || { echo "Missing FASTA or index: $REF" >&2; exit 1; }
[[ -x "$MODKIT" ]] || { echo "Missing executable: $MODKIT" >&2; exit 1; }

# A new directory prevents overwriting an earlier or currently running pileup.
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
"$MODKIT" --version > "$OUT/modkit_version.txt"
cp -- "${BASH_SOURCE[0]}" "$OUT/pileup_command.sh"

# Preserve the original script's automatic confidence thresholds and all modification codes.
echo "Starting pileup: $BAM"
"$MODKIT" pileup \
    "$BAM" \
    "$OUT/all_modifications.bed" \
    --ref "$REF" \
    --threads "$THREADS" \
    --log-filepath "$OUT/pileup.log"

echo "Finished: $OUT/all_modifications.bed"
