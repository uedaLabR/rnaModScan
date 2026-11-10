#!/bin/bash
set -euo pipefail

BAM="/mnt/ssdnas/dorado/riboML/rna004_130bps_sup@v6.0.0/HEK293T_IVT/PAW01579_75c9b793_949d4b52/bam/PAW01579_75c9b793_949d4b52.sorted.bam"
REF="/share/reference/hg38.fa"
MODKIT="/home/ueda/RNA004/modScore/modkit/target/release/modkit"
OUT="/home/ueda/RNA004/modScore/result/rna004_130bps_sup_v6.0.0/HEK293T_IVT_modkit"
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
"$MODKIT" pileup \
    "$BAM" \
    "$OUT/all_modifications.bed" \
    --ref "$REF" \
    --threads "$THREADS" \
    --log-filepath "$OUT/pileup.log"

echo "Finished: $OUT/all_modifications.bed"
