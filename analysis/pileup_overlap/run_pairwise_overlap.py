#!/usr/bin/env python3
"""
Pairwise RNA modification site overlap analysis.

For each RNA modification:
  - DRS (Dorado/modkit pileup) vs short-read reference sites
  - DRS (Dorado/modkit pileup) vs RMBase 3.0

m6A is reported twice using the SAME m6A call threshold:
  1) all m6A sites
  2) DRACH-only m6A sites

The DRACH restriction is applied symmetrically to DRS, short-read, and RMBase.

No command-line arguments are required.
Edit the CONFIGURATION section below, then run:

    python -B analysis/pileup_overlap/run_pairwise_overlap.py
"""

from __future__ import annotations

import csv
import sys
import json
import math
import re
from collections import Counter
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from utils.io.bed import bedmethyl_rows, load_bed_sites, open_text, parse_bedmethyl_row

# Genome FASTA used only for DRACH classification.
# A matching .fai file is required.
REFERENCE_FASTA = Path("/share/reference/hg38.fa")

# Use a NEW output directory for each final run.
OUTPUT_DIR = Path("/home/ueda/PycharmProjects/rnaModScan/results/pairwise_overlap_v3")

# Four completed samples. WT uses pass reads only.
SAMPLES = [
    dict(enabled=True,
         sample_id='HEK293T_WT', cell='HEK293T', condition='WT',
         bedmethyl=Path('/home/ueda/RNA004/modScore/result/rna004_130bps_sup_v6.0.0/HEK293T_WT_pass_only_modkit/all_modifications.bed')),
    dict(enabled=True,
         sample_id='HEK293T_IVT', cell='HEK293T', condition='IVT',
         bedmethyl=Path('/home/ueda/RNA004/modScore/result/rna004_130bps_sup_v6.0.0/HEK293T_IVT_modkit/all_modifications.bed')),
    dict(enabled=True,
         sample_id='HeLa_WT', cell='HeLa', condition='WT',
         bedmethyl=Path('/home/ueda/RNA004/modScore/result/rna004_130bps_sup_v6.0.0/HeLa_WT_pass_only_modkit/all_modifications.bed')),
    dict(enabled=True,
         sample_id='HeLa_IVT', cell='HeLa', condition='IVT',
         bedmethyl=Path('/home/ueda/RNA004/modScore/result/rna004_130bps_sup_v6.0.0/HeLa_IVT_modkit/all_modifications.bed')),
]

# ONE threshold set per modification.
# m6A-all and m6A-DRACH use exactly the same m6A threshold.
THRESHOLDS = {
    "m6A": dict(min_depth=10, min_modified=3, min_af=0.10),
    "m5C": dict(min_depth=10, min_modified=3, min_af=0.10),
    "inosine": dict(min_depth=10, min_modified=3, min_af=0.10),
    "pseU": dict(min_depth=10, min_modified=3, min_af=0.10),
}

# Prepared short-read reference BED directory.
# All method-specific BEDs for the same cell/modification are unioned.
SHORT_READ_DIR = ROOT / "data/reference_sites_GRCh38/by_method"

# RMBase 3.0 normalized BED directory.
RMBASE_DIR = ROOT / "references/rmbase3"

# User-approved HepG2 PUS7 positives, applied explicitly to both cells.
# These enter the experimental-reference bucket; the assay itself is Nanopore.
PUS7_REFERENCE_DIR = ROOT / "data/reference_sites_GRCh38/pus7"

WRITE_PLOTS = True
PLOT_FORMATS = ("png", "svg", "pdf")
PLOT_DPI = 250

PROGRESS_EVERY = 1_000_000

# ============================================================
# CONSTANTS
# ============================================================

MOD_CODE_TO_NAME = {
    "a": "m6A",
    "m": "m5C",
    "17596": "inosine",
    "17802": "pseU",
}

SHORT_READ_FILE_MOD = {
    "m6A": "m6A",
    "m5C": "m5C",
    "inosine": "inosine",
    "pseU": "psi",
}

RMBASE_FILES = {
    "m6A": "m6A.GRCh38.bed",
    "m5C": "m5C.GRCh38.bed",
    "inosine": "inosine.GRCh38.bed",
    "pseU": "pseU.GRCh38.bed",
}

OUTPUT_GROUPS = [
    ("m6A", "all"),
    ("m6A", "DRACH"),
    ("m5C", "all"),
    ("inosine", "all"),
    ("pseU", "all"),
]

DRACH_RE = re.compile(r"^[AGT][AG]AC[ACT]$")
DNA_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


# ============================================================
# FASTA ACCESS
# ============================================================

class IndexedFasta:
    """Small cached random-access reader for a FASTA with samtools .fai."""

    def __init__(self, fasta_path: Path, chunk_size: int = 1_000_000):
        self.path = Path(fasta_path)
        self.fai_path = Path(str(self.path) + ".fai")
        if not self.path.is_file():
            raise FileNotFoundError(f"Reference FASTA not found: {self.path}")
        if not self.fai_path.is_file():
            raise FileNotFoundError(
                f"FASTA index not found: {self.fai_path}\n"
                f"Create it with: samtools faidx {self.path}"
            )

        self.index = {}
        with self.fai_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                c = line.rstrip("\n").split("\t")
                if len(c) < 5:
                    raise ValueError(f"Invalid .fai row: {line.rstrip()}")
                self.index[c[0]] = {
                    "length": int(c[1]),
                    "offset": int(c[2]),
                    "line_bases": int(c[3]),
                    "line_width": int(c[4]),
                }

        self.handle = self.path.open("rb")
        self.chunk_size = int(chunk_size)
        self.cache_key = None
        self.cache_start = 0
        self.cache_seq = ""

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _fetch_exact(self, chrom: str, start: int, end: int) -> str | None:
        meta = self.index.get(chrom)
        if meta is None:
            return None
        if start < 0 or end > meta["length"] or start >= end:
            return None

        line_bases = meta["line_bases"]
        line_width = meta["line_width"]

        first_line = start // line_bases
        last_base = end - 1
        last_line = last_base // line_bases

        file_start = (
            meta["offset"]
            + first_line * line_width
            + (start % line_bases)
        )
        file_end = (
            meta["offset"]
            + last_line * line_width
            + (last_base % line_bases)
            + 1
        )

        self.handle.seek(file_start)
        raw = self.handle.read(file_end - file_start)
        seq = raw.replace(b"\n", b"").replace(b"\r", b"").decode("ascii").upper()

        if len(seq) < end - start:
            return None
        return seq[: end - start]

    def _get_chunk(self, chrom: str, pos: int) -> tuple[int, str] | None:
        meta = self.index.get(chrom)
        if meta is None or pos < 0 or pos >= meta["length"]:
            return None

        chunk_id = pos // self.chunk_size
        key = (chrom, chunk_id)
        if key == self.cache_key:
            return self.cache_start, self.cache_seq

        # Add a small flank so 5-mer motifs at chunk boundaries remain cached.
        start = max(0, chunk_id * self.chunk_size - 4)
        end = min(meta["length"], (chunk_id + 1) * self.chunk_size + 4)
        seq = self._fetch_exact(chrom, start, end)
        if seq is None:
            return None

        self.cache_key = key
        self.cache_start = start
        self.cache_seq = seq
        return start, seq

    def motif5(self, chrom: str, start: int, strand: str) -> str | None:
        """Return RNA-orientation 5-mer centered on the BED start base."""
        if strand not in ("+", "-"):
            return None

        left = start - 2
        right = start + 3
        meta = self.index.get(chrom)

        if meta is None or left < 0 or right > meta["length"]:
            return None

        cached = self._get_chunk(chrom, start)
        if cached is None:
            return None

        chunk_start, seq = cached
        a = left - chunk_start
        b = right - chunk_start

        if a < 0 or b > len(seq):
            motif = self._fetch_exact(chrom, left, right)
        else:
            motif = seq[a:b]

        if motif is None or len(motif) != 5:
            return None

        motif = motif.upper()
        if strand == "-":
            motif = motif.translate(DNA_COMPLEMENT)[::-1]
        return motif

    def is_drach(self, chrom: str, start: int, strand: str) -> bool:
        motif = self.motif5(chrom, start, strand)
        return bool(motif and DRACH_RE.fullmatch(motif))


# ============================================================
# INPUT READERS
# ============================================================


def validate_thresholds():
    expected = set(MOD_CODE_TO_NAME.values())
    if set(THRESHOLDS) != expected:
        raise ValueError(
            "THRESHOLDS must contain exactly: "
            + ", ".join(sorted(expected))
        )

    for mod, cfg in THRESHOLDS.items():
        if cfg["min_depth"] < 1 or cfg["min_modified"] < 1:
            raise ValueError(f"{mod}: read-count thresholds must be positive")
        if not math.isfinite(cfg["min_af"]) or not 0 <= cfg["min_af"] <= 1:
            raise ValueError(f"{mod}: min_af must be between 0 and 1")



def read_drs_calls(bedmethyl: Path, fasta: IndexedFasta):
    """Read the modkit pileup once and return threshold-passing DRS site sets."""
    bedmethyl = Path(bedmethyl)
    if not bedmethyl.is_file():
        raise FileNotFoundError(f"modkit pileup not found: {bedmethyl}")

    calls = {mod: set() for mod in THRESHOLDS}
    m6a_drach = set()

    observed_codes = Counter()
    input_rows = 0
    supported_rows = 0
    passed_rows = Counter()

    for row in bedmethyl_rows(bedmethyl):
        input_rows += 1
        observed_codes[row["mod_code"]] += 1

        if PROGRESS_EVERY and input_rows % PROGRESS_EVERY == 0:
            print(
                f"  DRS pileup: {input_rows:,} rows processed",
                flush=True,
            )

        mod = row["modification"]
        if mod is None:
            continue

        supported_rows += 1
        cfg = THRESHOLDS[mod]
        if (
            row["total_reads"] < cfg["min_depth"]
            or row["modified_reads"] < cfg["min_modified"]
            or row["af"] < cfg["min_af"]
        ):
            continue

        site = (row["chrom"], row["start"], row["strand"])
        calls[mod].add(site)
        passed_rows[mod] += 1

        if mod == "m6A" and fasta.is_drach(*site):
            m6a_drach.add(site)

    return {
        "all": calls,
        "m6A_DRACH": m6a_drach,
        "input_rows": input_rows,
        "supported_rows": supported_rows,
        "observed_codes": dict(observed_codes),
        "passed_rows": dict(passed_rows),
    }


def read_bed_sites(path: Path, fasta: IndexedFasta | None = None):
    """Read BED6-like single-base sites; optionally also build a DRACH subset."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    print(f"  Loading {path.name}...", flush=True)
    all_sites = load_bed_sites(path)
    print(f"    {len(all_sites):,} unique sites loaded", flush=True)
    drach_sites = None
    if fasta is not None:
        drach_sites = set()
        # Sequential loci reuse the genome chunk instead of random set-order reads.
        for index, site in enumerate(sorted(all_sites), 1):
            if fasta.is_drach(*site):
                drach_sites.add(site)
            if index % 100_000 == 0:
                print(f"    DRACH: {index:,}/{len(all_sites):,} sites", flush=True)
        print(f"    DRACH complete: {len(drach_sites):,} sites", flush=True)

    return all_sites, drach_sites


def method_name_from_path(path: Path, cell: str, file_mod: str):
    name = path.name
    prefix = f"{cell}_{file_mod}_"
    suffix = "_GRCh38.bed"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix):-len(suffix)]
    return path.stem


def load_short_read_references(cell: str, fasta: IndexedFasta):
    """Load method-specific BEDs and union them by modification."""
    result = {}
    method_sets = {}

    for mod, file_mod in SHORT_READ_FILE_MOD.items():
        paths = sorted(SHORT_READ_DIR.glob(f"{cell}_{file_mod}_*_GRCh38.bed"))

        extra_methods = {}
        if mod == "pseU" and cell in ("HEK293T", "HeLa"):
            for sheet in ("S1A", "S1B"):
                path = PUS7_REFERENCE_DIR / f"{sheet}.bed"
                paths.append(path)
                extra_methods[path] = f"PUS7_{sheet}"

        union_all = set()
        union_drach = set() if mod == "m6A" else None
        per_method = {}

        for path in paths:
            all_sites, drach_sites = read_bed_sites(
                path,
                fasta=fasta if mod == "m6A" else None,
            )
            method = extra_methods.get(path) or method_name_from_path(path, cell, file_mod)
            per_method[method] = {
                "path": str(path.resolve()),
                "all": all_sites,
                "DRACH": drach_sites if mod == "m6A" else None,
            }
            union_all.update(all_sites)
            if mod == "m6A":
                union_drach.update(drach_sites)

        result[mod] = {
            "paths": [str(p.resolve()) for p in paths],
            "all": union_all,
            "DRACH": union_drach if mod == "m6A" else None,
            "available": bool(paths),
        }
        method_sets[mod] = per_method

    return result, method_sets


def load_rmbase_references(fasta: IndexedFasta):
    result = {}

    for mod, filename in RMBASE_FILES.items():
        path = RMBASE_DIR / filename

        if not path.is_file():
            result[mod] = {
                "path": str(path.resolve()),
                "all": set(),
                "DRACH": set() if mod == "m6A" else None,
                "available": False,
            }
            continue

        all_sites, drach_sites = read_bed_sites(
            path,
            fasta=fasta if mod == "m6A" else None,
        )

        result[mod] = {
            "path": str(path.resolve()),
            "all": all_sites,
            "DRACH": drach_sites if mod == "m6A" else None,
            "available": True,
        }

    return result


# ============================================================
# OVERLAP CALCULATION
# ============================================================

def group_sites(container, modification: str, motif_group: str):
    if modification == "m6A" and motif_group == "DRACH":
        return container["DRACH"]
    return container["all"]


def pairwise_counts(drs_sites: set, ref_sites: set):
    overlap = drs_sites & ref_sites
    drs_only = drs_sites - ref_sites
    ref_only = ref_sites - drs_sites
    union = drs_sites | ref_sites

    return {
        "drs_sites": len(drs_sites),
        "reference_sites": len(ref_sites),
        "overlap_sites": len(overlap),
        "drs_only": len(drs_only),
        "reference_only": len(ref_only),
        "union_sites": len(union),
        "jaccard": len(overlap) / len(union) if union else None,
        "drs_supported_pct": (
            100.0 * len(overlap) / len(drs_sites) if drs_sites else None
        ),
        "reference_recovered_pct": (
            100.0 * len(overlap) / len(ref_sites) if ref_sites else None
        ),
    }


def fmt_float(value):
    return "" if value is None else f"{value:.6f}"


# ============================================================
# PLOTTING
# ============================================================

def require_matplotlib():
    try:
        import matplotlib
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.patches import Circle
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install with: "
            "python -m pip install matplotlib"
        ) from exc

    return matplotlib, Figure, FigureCanvasAgg, Circle


def draw_pairwise_panel(ax, title, left_label, right_label, counts, Circle):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(title, fontsize=13, weight="bold", pad=10)

    ax.add_patch(
        Circle(
            (4.0, 3.1),
            2.0,
            facecolor="#4E9AC7",
            edgecolor="#2B698F",
            alpha=0.38,
            linewidth=1.5,
        )
    )
    ax.add_patch(
        Circle(
            (6.0, 3.1),
            2.0,
            facecolor="#EDA860",
            edgecolor="#A46121",
            alpha=0.38,
            linewidth=1.5,
        )
    )

    ax.text(2.7, 5.35, left_label, ha="center", fontsize=11, weight="bold")
    ax.text(7.3, 5.35, right_label, ha="center", fontsize=11, weight="bold")

    ax.text(
        3.0, 3.15, f"{counts['drs_only']:,}",
        ha="center", va="center", fontsize=17, weight="bold"
    )
    ax.text(
        5.0, 3.15, f"{counts['overlap_sites']:,}",
        ha="center", va="center", fontsize=17, weight="bold"
    )
    ax.text(
        7.0, 3.15, f"{counts['reference_only']:,}",
        ha="center", va="center", fontsize=17, weight="bold"
    )

    ax.text(3.0, 2.55, "only", ha="center", fontsize=9)
    ax.text(5.0, 2.55, "overlap", ha="center", fontsize=9)
    ax.text(7.0, 2.55, "only", ha="center", fontsize=9)

    drs_pct = counts["drs_supported_pct"]
    ref_pct = counts["reference_recovered_pct"]

    footer = (
        f"DRS: {counts['drs_sites']:,} sites"
        f"  |  overlap: {counts['overlap_sites']:,}"
        f"  |  reference: {counts['reference_sites']:,} sites"
    )
    ax.text(5.0, 0.75, footer, ha="center", fontsize=8.5, color="#455767")

    pct_text = (
        f"Overlap = {drs_pct:.1f}% of DRS; {ref_pct:.1f}% of reference"
        if drs_pct is not None and ref_pct is not None
        else "Overlap percentage: n/a"
    )
    ax.text(5.0, 0.38, pct_text, ha="center", fontsize=8.5, color="#455767")


def make_pairwise_figure(
    sample,
    modification,
    motif_group,
    short_counts,
    rmbase_counts,
    short_available,
    rmbase_available,
    plotting,
):
    matplotlib, Figure, FigureCanvasAgg, Circle = plotting

    fig = Figure(figsize=(13.5, 6.7), facecolor="white")
    FigureCanvasAgg(fig)

    ax1 = fig.add_axes([0.04, 0.16, 0.44, 0.70])
    ax2 = fig.add_axes([0.52, 0.16, 0.44, 0.70])

    if short_available:
        draw_pairwise_panel(
            ax1,
            "DRS vs short-read",
            "DRS",
            "Short-read",
            short_counts,
            Circle,
        )
    else:
        ax1.axis("off")
        ax1.text(
            0.5, 0.55,
            "DRS vs short-read\n\nNo short-read reference BED found",
            transform=ax1.transAxes,
            ha="center", va="center", fontsize=13
        )

    if rmbase_available:
        draw_pairwise_panel(
            ax2,
            "DRS vs RMBase 3.0",
            "DRS",
            "RMBase",
            rmbase_counts,
            Circle,
        )
    else:
        ax2.axis("off")
        ax2.text(
            0.5, 0.55,
            "DRS vs RMBase 3.0\n\nRMBase reference BED not found",
            transform=ax2.transAxes,
            ha="center", va="center", fontsize=13
        )

    group_label = (
        "m6A - DRACH"
        if modification == "m6A" and motif_group == "DRACH"
        else modification
    )
    if modification == "m6A" and motif_group == "all":
        group_label = "m6A - all sites"

    cfg = THRESHOLDS[modification]

    fig.text(
        0.5,
        0.965,
        f"{sample['sample_id']}  |  {group_label}",
        ha="center",
        va="top",
        fontsize=16,
        weight="bold",
    )

    fig.text(
        0.5,
        0.920,
        f"{sample['cell']}  |  {sample['condition']}  |  "
        f"depth >= {cfg['min_depth']}  |  modified reads >= {cfg['min_modified']}  |  "
        f"AF >= {cfg['min_af']}",
        ha="center",
        va="top",
        fontsize=10,
        color="#455767",
    )

    note = "Exact site match: chromosome + BED start + strand."
    if modification == "m6A" and motif_group == "DRACH":
        note += " DRACH filtering is applied to DRS, short-read, and RMBase."
    fig.text(
        0.5,
        0.065,
        note + " Circle areas are schematic, not proportional.",
        ha="center",
        fontsize=8.5,
        color="#455767",
    )

    return fig


# ============================================================
# OUTPUT
# ============================================================

SUMMARY_FIELDS = [
    "sample_id",
    "cell",
    "condition",
    "modification",
    "motif_group",
    "comparison",
    "reference_label",
    "reference_status",
    "min_depth",
    "min_modified",
    "min_af",
    "drs_sites",
    "reference_sites",
    "overlap_sites",
    "drs_only",
    "reference_only",
    "union_sites",
    "jaccard",
    "drs_supported_pct",
    "reference_recovered_pct",
]


METHOD_FIELDS = [
    "sample_id",
    "cell",
    "condition",
    "modification",
    "motif_group",
    "method",
    "reference_path",
    "min_depth",
    "min_modified",
    "min_af",
    "drs_sites",
    "method_sites",
    "overlap_sites",
    "drs_only",
    "method_only",
    "jaccard",
]


def write_tsv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))


# ============================================================
# MAIN
# ============================================================

def main():
    validate_thresholds()

    selected = [s for s in SAMPLES if s.get("enabled")]
    if not selected:
        raise ValueError("Enable at least one sample in SAMPLES")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}\n"
            "Choose a new OUTPUT_DIR for this run."
        )

    OUTPUT_DIR.mkdir(parents=True)
    plots_dir = OUTPUT_DIR / "venn"
    if WRITE_PLOTS:
        plots_dir.mkdir()

    print("RNA modification pairwise overlap", flush=True)
    print(f"Project root : {ROOT}", flush=True)
    print(f"Genome       : {REFERENCE_FASTA}", flush=True)
    print(f"Short-read   : {SHORT_READ_DIR}", flush=True)
    print(f"RMBase       : {RMBASE_DIR}", flush=True)
    print(f"Output       : {OUTPUT_DIR}", flush=True)

    plotting = require_matplotlib() if WRITE_PLOTS else None

    summary_rows = []
    method_rows = []
    run_samples = []

    try:
        with IndexedFasta(REFERENCE_FASTA) as fasta:
            # RMBase is cell-independent, so load it once.
            print("Loading RMBase reference sites...", flush=True)
            rmbase = load_rmbase_references(fasta)

            for sample in selected:
                sample_id = sample["sample_id"]
                cell = sample["cell"]
                condition = sample["condition"]
                bedmethyl = Path(sample["bedmethyl"])

                print("", flush=True)
                print(f"Sample: {sample_id}", flush=True)
                print(f"  DRS pileup: {bedmethyl}", flush=True)

                print(f"  Loading short-read references for {cell}...", flush=True)
                short_refs, short_methods = load_short_read_references(cell, fasta)

                print("  Reading DRS/modkit pileup...", flush=True)
                drs_result = read_drs_calls(bedmethyl, fasta)

                run_sample = {
                    "sample_id": sample_id,
                    "cell": cell,
                    "condition": condition,
                    "bedmethyl": str(bedmethyl.resolve()),
                    "input_rows": drs_result["input_rows"],
                    "supported_rows": drs_result["supported_rows"],
                    "observed_codes": drs_result["observed_codes"],
                    "passed_rows": drs_result["passed_rows"],
                    "short_read_sources": {
                        mod: short_refs[mod]["paths"]
                        for mod in THRESHOLDS
                    },
                }
                run_samples.append(run_sample)

                for modification, motif_group in OUTPUT_GROUPS:
                    cfg = THRESHOLDS[modification]

                    if modification == "m6A" and motif_group == "DRACH":
                        drs_sites = drs_result["m6A_DRACH"]
                        short_sites = short_refs["m6A"]["DRACH"]
                        db_sites = rmbase["m6A"]["DRACH"]
                    else:
                        drs_sites = drs_result["all"][modification]
                        short_sites = short_refs[modification]["all"]
                        db_sites = rmbase[modification]["all"]

                    short_available = short_refs[modification]["available"]
                    db_available = rmbase[modification]["available"]

                    short_counts = (
                        pairwise_counts(drs_sites, short_sites)
                        if short_available
                        else None
                    )
                    db_counts = (
                        pairwise_counts(drs_sites, db_sites)
                        if db_available
                        else None
                    )

                    label = (
                        f"{modification}-{motif_group}"
                        if motif_group != "all"
                        else modification
                    )

                    print(
                        f"  {label}: DRS={len(drs_sites):,}",
                        flush=True,
                    )
                    if short_counts is not None:
                        print(
                            f"    short-read: ref={short_counts['reference_sites']:,}, "
                            f"overlap={short_counts['overlap_sites']:,}, "
                            f"DRS-only={short_counts['drs_only']:,}, "
                            f"short-only={short_counts['reference_only']:,}",
                            flush=True,
                        )
                    else:
                        print("    short-read: unavailable", flush=True)

                    if db_counts is not None:
                        print(
                            f"    RMBase: ref={db_counts['reference_sites']:,}, "
                            f"overlap={db_counts['overlap_sites']:,}, "
                            f"DRS-only={db_counts['drs_only']:,}, "
                            f"RMBase-only={db_counts['reference_only']:,}",
                            flush=True,
                        )
                    else:
                        print("    RMBase: unavailable", flush=True)

                    for comparison, ref_label, available, counts in [
                        (
                            "short_read",
                            "Short-read union",
                            short_available,
                            short_counts,
                        ),
                        (
                            "rmbase",
                            "RMBase 3.0",
                            db_available,
                            db_counts,
                        ),
                    ]:
                        base = {
                            "sample_id": sample_id,
                            "cell": cell,
                            "condition": condition,
                            "modification": modification,
                            "motif_group": motif_group,
                            "comparison": comparison,
                            "reference_label": ref_label,
                            "reference_status": "ready" if available else "unavailable",
                            "min_depth": cfg["min_depth"],
                            "min_modified": cfg["min_modified"],
                            "min_af": cfg["min_af"],
                        }

                        if counts is None:
                            base.update({
                                "drs_sites": len(drs_sites),
                                "reference_sites": "",
                                "overlap_sites": "",
                                "drs_only": "",
                                "reference_only": "",
                                "union_sites": "",
                                "jaccard": "",
                                "drs_supported_pct": "",
                                "reference_recovered_pct": "",
                            })
                        else:
                            base.update({
                                "drs_sites": counts["drs_sites"],
                                "reference_sites": counts["reference_sites"],
                                "overlap_sites": counts["overlap_sites"],
                                "drs_only": counts["drs_only"],
                                "reference_only": counts["reference_only"],
                                "union_sites": counts["union_sites"],
                                "jaccard": fmt_float(counts["jaccard"]),
                                "drs_supported_pct": fmt_float(
                                    counts["drs_supported_pct"]
                                ),
                                "reference_recovered_pct": fmt_float(
                                    counts["reference_recovered_pct"]
                                ),
                            })

                        summary_rows.append(base)

                    # Also save method-specific short-read overlap statistics.
                    for method, info in short_methods[modification].items():
                        if modification == "m6A" and motif_group == "DRACH":
                            method_sites = info["DRACH"]
                        else:
                            method_sites = info["all"]

                        counts = pairwise_counts(drs_sites, method_sites)
                        method_rows.append({
                            "sample_id": sample_id,
                            "cell": cell,
                            "condition": condition,
                            "modification": modification,
                            "motif_group": motif_group,
                            "method": method,
                            "reference_path": info["path"],
                            "min_depth": cfg["min_depth"],
                            "min_modified": cfg["min_modified"],
                            "min_af": cfg["min_af"],
                            "drs_sites": counts["drs_sites"],
                            "method_sites": counts["reference_sites"],
                            "overlap_sites": counts["overlap_sites"],
                            "drs_only": counts["drs_only"],
                            "method_only": counts["reference_only"],
                            "jaccard": fmt_float(counts["jaccard"]),
                        })

                    if WRITE_PLOTS:
                        fig = make_pairwise_figure(
                            sample=sample,
                            modification=modification,
                            motif_group=motif_group,
                            short_counts=short_counts,
                            rmbase_counts=db_counts,
                            short_available=short_available,
                            rmbase_available=db_available,
                            plotting=plotting,
                        )

                        stem = (
                            f"{safe_name(sample_id)}__"
                            f"{safe_name(modification)}__"
                            f"{safe_name(motif_group)}"
                        )

                        try:
                            for fmt in PLOT_FORMATS:
                                fig.savefig(
                                    plots_dir / f"{stem}.{fmt}",
                                    format=fmt,
                                    dpi=PLOT_DPI,
                                    facecolor="white",
                                )
                        finally:
                            fig.clear()

        write_tsv(
            OUTPUT_DIR / "pairwise_overlap_summary.tsv",
            SUMMARY_FIELDS,
            summary_rows,
        )

        write_tsv(
            OUTPUT_DIR / "short_read_method_overlap.tsv",
            METHOD_FIELDS,
            method_rows,
        )

        report = {
            "status": "complete",
            "assembly": "GRCh38",
            "coordinate_match": "chromosome + BED0 start + strand",
            "m6A_groups": ["all", "DRACH"],
            "m6A_drach_rule": "DRACH filtering applied symmetrically to DRS, short-read, and RMBase",
            "thresholds": THRESHOLDS,
            "reference_fasta": str(REFERENCE_FASTA.resolve()),
            "short_read_dir": str(SHORT_READ_DIR.resolve()),
            "additional_positive_evidence": {
                "path": str(PUS7_REFERENCE_DIR.resolve()),
                "source_cell": "HepG2", "applies_to": ["HEK293T", "HeLa"],
                "assay": "Nanopore direct RNA; S1B Dorado",
                "note": "Experimental-reference bucket includes user-approved cross-cell PUS7 sites; not exclusively short-read",
            },
            "rmbase_dir": str(RMBASE_DIR.resolve()),
            "samples": run_samples,
            "outputs": {
                "summary": "pairwise_overlap_summary.tsv",
                "method_summary": "short_read_method_overlap.tsv",
                "plots": "venn/" if WRITE_PLOTS else None,
            },
        }

        (OUTPUT_DIR / "run.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("", flush=True)
        print("Complete.", flush=True)
        print(
            f"Summary: {OUTPUT_DIR / 'pairwise_overlap_summary.tsv'}",
            flush=True,
        )
        print(
            f"Method table: {OUTPUT_DIR / 'short_read_method_overlap.tsv'}",
            flush=True,
        )
        if WRITE_PLOTS:
            print(f"Plots: {plots_dir}", flush=True)

    except BaseException as exc:
        (OUTPUT_DIR / "FAILED.txt").write_text(str(exc), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
