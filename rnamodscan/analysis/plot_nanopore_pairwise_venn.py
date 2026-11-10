"""Pairwise Venn diagrams: Nanopore vs short-read and Nanopore vs RMBase."""

import csv
import hashlib
import json
import re
from pathlib import Path

# ============================================================
# EDIT THESE PATHS
# ============================================================
SUMMARY_DIR = Path("/path/to/summary_dir")
OUTPUT_DIR = SUMMARY_DIR / "venn_pairwise"

TABLE_NAME = "reference_overlap_summary.tsv"
RUN_JSON_NAME = "run.json"

FORMATS = ("png", "svg", "pdf")
DPI = 250
# ============================================================

CATEGORIES = ("short_read_only", "both", "database_only", "neither")
IDENTITY = ("sample_id", "threshold", "modification", "motif_group")

# Two additional columns are required in reference_overlap_summary.tsv:
#   short_read_sites : total number of short-read reference sites
#   database_sites   : total number of database/RMBase reference sites
#
# Existing columns are still used to obtain overlap with Nanopore:
#   Nanopore �� short-read = short_read_only + both
#   Nanopore �� database   = database_only + both


def require_plotting():
    try:
        import matplotlib
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.patches import Circle
    except ImportError as exc:
        raise RuntimeError(
            "Pairwise Venn plotting requires matplotlib."
        ) from exc
    return matplotlib, Figure, FigureCanvasAgg, Circle


def read_partitions(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")

        required = set(
            IDENTITY
            + CATEGORIES
            + (
                "cell",
                "reference_cell",
                "condition",
                "call_sites",
                "reference_status",
                "min_depth",
                "min_modified",
                "min_af",
                "short_read_sites",
                "database_sites",
            )
        )

        if not required.issubset(reader.fieldnames or []):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(
                "Missing columns in reference_overlap_summary.tsv: "
                + ", ".join(missing)
            )

        rows = []
        seen = set()

        for row in reader:
            key = tuple(row[k] for k in IDENTITY)

            if not all(key) or key in seen:
                raise ValueError(f"Duplicate or empty plot identity: {key}")

            seen.add(key)

            for field in ("call_sites", "short_read_sites", "database_sites"):
                row[field] = int(row[field])
                if row[field] < 0:
                    raise ValueError(f"{field} must be nonnegative")

            if row["reference_status"] == "ready":
                for field in CATEGORIES:
                    row[field] = int(row[field])
                    if row[field] < 0:
                        raise ValueError(f"{field} must be nonnegative")

SUMMARY_DIR = Path("/������/reference_overlap_summary.tsv��run.json�������f�B���N�g��")
OUTPUT_DIR = SUMMARY_DIR / "venn_pairwise"

if __name__ == "__main__":
    print(f"Summary directory : {SUMMARY_DIR}", flush=True)
    print(f"Output directory  : {OUTPUT_DIR}", flush=True)

    generate_venn_plots(
        SUMMARY_DIR,
        OUTPUT_DIR,
        formats=("png", "svg", "pdf"),
        dpi=250,
    )