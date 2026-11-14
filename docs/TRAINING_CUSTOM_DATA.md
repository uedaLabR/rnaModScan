# Train with custom references and another species

Training accepts selected positive genomic sites from GLORI, Slic-seq, RMBase
or another explicitly documented assay/database. This imports reference
evidence into the existing m6A/inosine/background model; it does not replace
the nanopore sequencing inputs or change the network architecture.

## Required evidence and sequencing inputs

| Available result | Use |
|---|---|
| GLORI or another m6A assay | Positive m6A sites for the appropriate cell/tissue |
| Slic-seq or another inosine assay | Positive inosine sites for the appropriate cell/tissue |
| RMBase or another database | Separate positive source entries for each target |
| Several assays/databases | Import together, retaining individual source IDs |
| Only one target's reference | Can prepare labels, but cannot fit both modification classes |

The current joint model needs positive examples for **both m6A and inosine**,
plus negative examples from the established WT/IVT protocol. For each cell key,
provide one WT and corresponding IVT BAM with Dorado MM/ML tags and indexes,
their modkit bedMethyl and matching logs, and a softmasked FASTA/FAI. More cell
pairs can be added to YAML. Duplicate WT/IVT replicate records under one cell
key are not currently supported; use an explicit reviewed sample/cell mapping.

Reference sites alone cannot train the model. Unmatched reference sites and
missing IVT coverage are unknown, never automatic negatives. The current
workflow does not accept a positive-only assay as a substitute for the IVT
control or as evidence for a modification it did not measure. m5C and
pseudouridine use known-site/motif retention without NN training. Their
known-site route keeps the depth ≥10, modified reads ≥3 and AF ≥0.10 candidate
defaults. Only the motif route adds depth ≥20 and AF ≥0.40, controlled by
`filtering.auxiliary_motif_min_depth` and `filtering.auxiliary_motif_min_af`.
m6A/inosine and training thresholds are unchanged.

## Import positive-site files

First select positive sites using criteria appropriate to your assay/database.
The importer does not choose significance or stoichiometry cutoffs. Export
Excel sheets to genomic TSV/CSV, and convert transcript coordinates to your
genomic assembly before import. Raw files with undocumented columns are not
automatically interpreted.

```bash
mkdir -p configs/local
cp training/examples/reference_sources.yaml configs/local/reference_sources.yaml
```

Edit the manifest's assembly, species, cell/tissue and source paths. Paths are
relative to the manifest. Every source needs a citation or source URL, including
unpublished results. An m6A table with `chromosome`, `position`, `strand` headers
and 1-based positions can be combined with an inosine BED6:

```yaml
schema_version: 1
assembly: GRCh38
species: Homo sapiens
sources:
  - id: tissue.GLORI
    kind: short_read
    cell: my_tissue
    modification: m6A
    path: positive_m6a.tsv
    format: tsv
    coordinates: one_based_position
    columns: {chrom: chromosome, position: position, strand: strand}
    citation: Your GLORI study and positive-site selection
  - id: tissue.Slic-seq
    kind: short_read
    cell: my_tissue
    modification: inosine
    path: positive_inosine.bed
    format: bed
    coordinates: BED0
    citation: Your Slic-seq study and positive-site selection
```

Put those files beside the copied manifest or supply explicit paths. Column
names are case-sensitive. Plain and gzip-compressed BED/TSV/CSV are supported.

| Coordinate setting | Required column keys | Meaning |
|---|---|---|
| `BED0` | `chrom`, `start`, `end`, `strand` | 0-based half-open; `end=start+1` |
| `one_based_position` | `chrom`, `position`, `strand` | Single 1-based position |
| `one_based_closed` | `chrom`, `start`, `end`, `strand` | Single-base 1-based closed interval |

For CSV/TSV, map keys to exact header names. For BED, map them to 1-based column
numbers; the default is chrom/start/end/strand in columns 1/2/3/6, using BED0.
A one-based position 101 becomes BED `start=100`, `end=101`. A `+` or `-` strand
is mandatory. There is no strand inference, chromosome renaming or liftover.
Duplicate coordinates within a source are collapsed and counted; separate
sources retain their identities.

```bash
rnamodscan references build --manifest configs/local/reference_sources.yaml \
  --output data/training_references/my_study
rnamodscan references check \
  --catalog data/training_references/my_study/catalog.json --training
```

The destination must be new. It contains `catalog.json`, normalized `sites/*.bed`,
`provenance/*.json`, `SHA256SUMS` and `import_run.json`. Provenance records source
file hashes, column mappings, coordinate conversion and citations. The importer
validates staged files and refuses to overwrite existing outputs. Original
assay files are preserved.

## Database-only or mixed references

An RMBase or other database may replace or supplement assay positive maps.
Create separate entries for m6A and inosine. For a preselected single-base BED:

```yaml
  - id: RMBase3.m6A
    kind: database
    database: RMBase
    version: "3.0"
    cell: "*"
    modification: m6A
    path: rmbase_m6a.bed.gz
    format: bed
    coordinates: BED0
    source_url: https://rna.sysu.edu.cn/rmbase3/download.php
```

Place this in the manifest's `sources` list and add the corresponding inosine
entry. Other databases use their actual name, version and citation. `cell: "*"`
is permitted for database-wide evidence, but does not make it cell-specific.
Experimental `short_read` sources require an explicit cell/tissue.

A raw multi-column download is not automatically a correctly mapped single-base
BED: select the target modification and specify the columns and assembly.
Human HEK293T/HeLa training can instead reuse the included
`data/references/hg38/catalog.json`, without the original RMBase archives or
large provenance TSVs.

## Initialize and run training

```bash
rnamodscan train init --output configs/local/train.yaml --cell my_tissue \
  --catalog data/training_references/my_study/catalog.json \
  --fasta /absolute/path/to/matching.softmasked.fa \
  --input-dir /absolute/path/to/WT_IVT_inputs --run-dir results/my_training
```

Edit the generated sample paths, or set filename stems using `--wt-id` and
`--ivt-id`. Sample `reference_cell` must match the experimental catalog scope;
use `--reference-cell` when this mapping differs explicitly from `--cell`.
`selection.source_dir: null` builds new lists. A same-assembly GTF/GENCODE-style
GFF3 is optional reporting annotation; `references.gtf: null` skips it in the
full workflow. It never becomes a neural input.

```bash
rnamodscan train --config configs/local/train.yaml --dry-run
rnamodscan train check --config configs/local/train.yaml
# Starts the actual extraction, cross-validation and final fitting:
rnamodscan train --config configs/local/train.yaml
```

`init` writes only new settings; `check` is read-only. The full command stops
before running stages if preflight fails. Enough independent locus groups and
all three training classes are required for grouped five-fold splitting. The
nanopore candidates must actually overlap the positive references. Missing
classes in the globally selected dataset stop before expensive BAM feature
extraction. Each training fold must also contain all three classes; very small
or spatially clustered datasets can still fail that check after splitting.

Held-out results are saved in `models/adenine/fold_i/test/metrics.json`.
Checkpoints and development histories use validation data; test labels do not
select checkpoints. Final refitting always requires all five completed folds.
It uses their selected epoch schedule by default; `final_epochs` overrides only
the epoch count, not that prerequisite. It
exports `final_model/adenine/{model.keras,run.json}` with history and training
IDs, without assigning a new held-out score to the full-data fit. Existing
human weights and completed analyses are never replaced. See the
[training guide](../training/README.md) for environment and stage details.

## Another species or assembly

1. Obtain matching-species WT/IVT data, positive evidence for both targets, and a
   softmasked FASTA/FAI. The BAM, site maps and optional annotation must agree
   on assembly and chromosome names.
2. Declare the actual manifest assembly/species, for example `assembly: mm39`
   and `species: Mus musculus`. Import/check the catalog, then initialize the
   training YAML using it. Training accepts custom experimental maps, including
   mouse assays, and retains assembly provenance in new indexes/models.
3. Train in a new directory and inspect held-out results in that species. The
   human model's performance does not establish transfer to another species.

Copy the prediction template and set these fields to use your new weights:

```yaml
genome: other
targets: [m6A, inosine]
model_root: /absolute/path/to/my_training/final_model
model_directories: {}
references:
  fasta: /absolute/path/to/matching.softmasked.fa
  gtf: null
  catalog: null
filtering:
  reference_only_targets: []
  m6a_drach_keep: false
  m6a_mettl16_keep: false
  m5c_nsun6_keep: false
  m5c_nsun2_keep: false
  pseu_trub1_keep: false
  pseu_pus7_keep: false
```

This is a fragment to combine with sample paths and remaining fields from
`prediction/config.yaml`. Select `mm10` only for mm10, `hg38` only for the
human GRCh38 workflow, and `other` for a custom assembly such as mm39. `other`
intentionally disables DB/short-read rescue even if a catalog is supplied.
For this custom-assembly prediction recipe, leave optional annotation off with
`gtf: null`. Disable motif-retention rules unless justified for that species
and experiment. The four m5C/Ψ keys matter only when those targets are included;
`other` still permits motif-only retention when a rule is enabled. See the
[output motif codes](OUTPUT_FLAGS.md#motif-codes). Set CPU/GPU devices as in the
[prediction guide](../prediction/README.md).

| Context | Reference behavior |
|---|---|
| Training | Explicitly declared matching-assembly experimental/database sources |
| Prediction `hg38` | Human DB and explicit matching-cell experimental maps |
| Prediction `mm10` | Mouse RMBase 3.0 only |
| Prediction `other` | No DB or short-read files opened |

Reference overlap is operational evidence, not independent chemical validation.
Do not assess performance using the same maps that supplied training or rescue
labels as though they were independent. The modification classes, 18 read
channels and 23,731-parameter architecture remain unchanged.
