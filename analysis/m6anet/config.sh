# HeLa WT / RNA004. Paths are deliberately independent of training_v10 outputs.
BAM='/mnt/ssdnas/dorado/riboML/rna004_130bps_sup@v6.0.0/HeLa/PAU09364_pass_c7fa45d4/bam_pass/PAU09364_pass_c7fa45d4.sorted.bam'
POD5_DIR='/mnt/ssdnas/riboML/HeLa/pod5_pass'
GENOME='/share/reference/hg38.fa'
ANNOTATION='/mnt/ssdnas07/pipeline/rna_modscore_v2/source/gencode.v44.annotation.gff3'
OUT="$PROJECT_ROOT/results/m6anet_HeLa_WT_v1"
THREADS=8
SIGNAL_PROCESSES=2
MIN_MAPQ=20
ITERATIONS=1000
SEED=42
# Dedicated environments; never install into .venv310 or .venv_cuda112.
TOOLS_ENV="$PROJECT_ROOT/.m6anet_tools"
# m6anet 2.1.0 requires Python <3.9 and torch==1.6.0; blue-crab needs Python >=3.10.
M6ANET_ENV="$PROJECT_ROOT/.m6anet_py38"
BLUECRAB_ENV="$PROJECT_ROOT/.venv_bluecrab"
