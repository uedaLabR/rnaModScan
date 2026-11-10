"""rnaModScan: edit server paths, then run python analysis/pileup_overlap/run_overlap.py."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from rnamodscan.analysis.call_summary import summarize_calls
from rnamodscan.analysis.overlap_plots import generate_venn_plots, require_plotting

# Completed modkit outputs (WT uses pass reads; IVT uses its existing BAM).
PILEUP_DIR = Path('/home/ueda/RNA004/modScore/result/rna004_130bps_sup_v6.0.0')
REFERENCE_FASTA = Path('/share/reference/hg38.fa')
REFERENCE_CATALOG = ROOT / 'data/references/catalog.json'
# Output tables for all four samples are saved together here. Use a new name for a rerun.
OUTPUT_DIR = Path('/home/ueda/PycharmProjects/rnaModScan/results/pileup_overlap_v2')
UPSTREAM_MIN_ML = None  # Read each sample's automatic thresholds from pileup.log.

SAMPLES = [
    dict(enabled=True, sample_id='HEK293T_WT', cell='HEK293T', reference_cell='HEK293T',
         condition='WT', bedmethyl=PILEUP_DIR / 'HEK293T_WT_pass_only_modkit/all_modifications.bed',
         pileup_log=PILEUP_DIR / 'HEK293T_WT_pass_only_modkit/pileup.log'),
    dict(enabled=True, sample_id='HEK293T_IVT', cell='HEK293T', reference_cell='HEK293T',
         condition='IVT', bedmethyl=PILEUP_DIR / 'HEK293T_IVT_modkit/all_modifications.bed',
         pileup_log=PILEUP_DIR / 'HEK293T_IVT_modkit/pileup.log'),
    dict(enabled=True, sample_id='HeLa_WT', cell='HeLa', reference_cell='HeLa',
         condition='WT', bedmethyl=PILEUP_DIR / 'HeLa_WT_pass_only_modkit/all_modifications.bed',
         pileup_log=PILEUP_DIR / 'HeLa_WT_pass_only_modkit/pileup.log'),
    dict(enabled=True, sample_id='HeLa_IVT', cell='HeLa', reference_cell='HeLa',
         condition='IVT', bedmethyl=PILEUP_DIR / 'HeLa_IVT_modkit/all_modifications.bed',
         pileup_log=PILEUP_DIR / 'HeLa_IVT_modkit/pileup.log'),
]
for sample in SAMPLES:
    sample['dorado_model_path_hint'] = 'rna004_130bps_sup@v6.0.0'
    sample['dorado_version'] = 'NOT_RECORDED'
    version_file = sample['bedmethyl'].parent / 'modkit_version.txt'
    sample['modkit_version'] = version_file.read_text().strip() if version_file.is_file() else 'NOT_RECORDED'

THRESHOLDS = [dict(name='depth10_mod3_af0.1', min_depth=10, min_modified=3, min_af=0.1)]
WRITE_CALLED_SITES = False  # False: small aggregate tables only; True: optional compressed site list.
COMPRESS_CALLED_SITES = True
PROGRESS_EVERY = 1000000
WRITE_VENN_PLOTS = False  # Tables first; plotting can be run later with plot_overlap.py.
VENN_FORMATS = ('png', 'svg', 'pdf')
VENN_DPI = 250


def main():
    selected = [{k: v for k, v in sample.items() if k != 'enabled'}
                for sample in SAMPLES if sample['enabled']]
    if not selected:
        raise ValueError('Enable at least one sample')
    for sample in selected:
        if sample['bedmethyl'] is None:
            raise ValueError(f"Set bedmethyl path for {sample['sample_id']}")
        if sample.get('pileup_log') is None:
            raise ValueError(f"Set pileup_log path for {sample['sample_id']}; do not assume fixed ML=125")
    if WRITE_VENN_PLOTS:
        require_plotting()  # Fail before reading a large pileup if matplotlib is missing.
    print('rnaModScan pileup overlap analysis', flush=True)
    print('Samples: ' + ', '.join(s['sample_id'] for s in selected), flush=True)
    print(f'DRACH FASTA: {REFERENCE_FASTA}', flush=True)
    print(f'Output: {OUTPUT_DIR}', flush=True)
    report = summarize_calls(selected, THRESHOLDS, REFERENCE_FASTA, REFERENCE_CATALOG,
                             OUTPUT_DIR, UPSTREAM_MIN_ML,
                             write_called_sites=WRITE_CALLED_SITES,
                             compress_called_sites=COMPRESS_CALLED_SITES,
                             progress_every=PROGRESS_EVERY)
    print(f"Aggregation complete: {report['elapsed_seconds'] / 60:.1f} min", flush=True)
    if WRITE_VENN_PLOTS:
        generate_venn_plots(OUTPUT_DIR, formats=VENN_FORMATS, dpi=VENN_DPI)


if __name__ == '__main__':
    main()
