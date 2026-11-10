"""Replot existing overlap summaries without reading pileup or genome; no CLI."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from rnamodscan.analysis.overlap_plots import generate_venn_plots

# Edit these paths; select a NEW OUTPUT_DIR when regenerating existing figures.
SUMMARY_DIR = Path('/home/ueda/PycharmProjects/rnaModScan/results/pileup_overlap_v2')
OUTPUT_DIR = SUMMARY_DIR / 'venn'
FORMATS = ('png', 'svg', 'pdf')
DPI = 250


if __name__ == '__main__':
    generate_venn_plots(SUMMARY_DIR, OUTPUT_DIR, formats=FORMATS, dpi=DPI)
