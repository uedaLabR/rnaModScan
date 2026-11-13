# rnaModScan

rnaModScan provides **deep learning-based filtering of RNA modification calls in
nanopore direct RNA sequencing**. It combines Dorado modification calls and
per-read confidence scores (MM/ML tags), reference sequence and read features with a shared
neural network classifier for **m6A, inosine and false-positive candidates**.
m5C and pseudouridine (`pseU`) are supported through reference overlap only.

The workflow starts from a sorted, indexed BAM and modkit bedMethyl candidates.
For each original candidate, database overlap, matching-cell experimental
reference overlap, the neural-network decision and the m6A DRACH and canonical
METTL16 motif rules are recorded separately. Their OR determines retention.
Reference-based retention applies to WT samples. The input BAM is preserved; writing a
new filtered BAM is optional.

## Installation

The included HEK293T + HeLa deployment model was trained with **Python 3.10 on Linux
x86_64, TensorFlow 2.10.1 and Keras 2.10.0**. Use a separate environment:

```bash
git clone https://github.com/uedaLabR/rnaModScan.git
cd rnaModScan
python3.10 -m venv .venv_predict
source .venv_predict/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r prediction/requirements-tf210.txt
python -m pip install -e .
python -m pip check
python -m rnamodscan --help
python -m rnamodscan predict --help
```

The pinned requirements record the working environment; a fresh installation
has not yet been validated. The same TensorFlow package supports CPU inference
with `--device cpu`. GPU use also requires a compatible NVIDIA driver, CUDA 11.2
and cuDNN 8.1; see the [TensorFlow compatibility table](https://www.tensorflow.org/install/source#gpu).

The current saved model cannot be loaded with TensorFlow 2.19/Keras 3. The
`training` and `prediction-cpu` extras in `pyproject.toml`, and the top-level
`requirements.txt`, select a newer TensorFlow. Use the pinned installation above
for the existing model, without combining those extras with it.

For configuration inspection and CLI help without an NN runtime, install only
the base package with `python -m pip install -e .`. The full workflow additionally
needs the libraries in the pinned requirements. External `modkit` is needed only
when `modkit.run_pileup: true`; existing bedMethyl and pileup logs can be reused.

## Docker: CPU or NVIDIA GPU

[Dockerfile.cpu](Dockerfile.cpu) and [Dockerfile.gpu](Dockerfile.gpu) package the
prediction workflow with the included final model and human reference BED/JSON
files. Both target Linux x86_64 containers and preserve TensorFlow 2.10.1/Keras
2.10.0 compatibility. Sample inputs and the matching softmasked genome are mounted
separately. The GPU image includes CUDA/cuDNN and needs a compatible NVIDIA driver
and NVIDIA Container Toolkit on the host.

```bash
docker build --platform linux/amd64 -f Dockerfile.cpu -t rnamodscan:cpu .
docker run --rm rnamodscan:cpu predict --help

# NVIDIA GPU host:
docker build --platform linux/amd64 -f Dockerfile.gpu -t rnamodscan:gpu .
docker run --rm --gpus all --entrypoint python rnamodscan:gpu \
  /opt/rnaModScan/docker/smoke_test.py --device gpu
```

See the [container guide](docker/README.md) for smoke tests, an example YAML,
read-only input mounts, CPU/GPU prediction and output paths. The images reuse
existing modkit outputs; external modkit and Dorado are not bundled. Docker
builds/container GPU execution were not run on the preparation host because its
daemon was inaccessible; a fresh isolated CPU installation, saved-model smoke
test and upstream manifest checks passed. Public repository release is planned for the week beginning
**16 November 2026**; it is currently private. These commands build locally and
do not assume published container images.

## Required inputs

Before running prediction, provide:

- A coordinate-sorted, indexed BAM containing Dorado MM/ML modification tags.
- bedMethyl and its matching modkit pileup log, or an installed modkit executable
  to generate them.
- A **softmasked FASTA and FAI matching the BAM assembly**. Lowercase `acgt`
  supplies repeat features; chromosome names and coordinates must agree.
- A compatible final-model bundle containing `model.keras` and `run.json`.
- The reference files listed in the selected catalog, when references are enabled.

The pretrained deployment model is included under
[`models/pooled_HEK293T_HeLa_20261111/`](models/pooled_HEK293T_HeLa_20261111/README.md).
It contains the original `adenine/model.keras` weights (about 0.49 MiB),
`adenine/run.json` and `SHA256SUMS`. It was fitted on all 548,735 labelled
HEK293T/HeLa WT/IVT observations using the five-epoch schedule selected during
development. No retraining or separate weight download is required for prediction.
Check the distributed files from the repository root:

```bash
(cd models/pooled_HEK293T_HeLa_20261111 && sha256sum -c SHA256SUMS)
```

The model loader also checks the input schema, class order and weight checksum.
The manuscript's HeLa analysis used separate HEK293T-only weights; those
performance results are not an independent evaluation of the pooled deployment
model. The prepared human reference BEDs and provenance JSON are also included
as described below. Supply your own matching genome FASTA/FAI and sample inputs;
analysis outputs are not distributed with the code.

## Prediction

Create a local copy of the template; `configs/local/` is ignored by Git:

```bash
mkdir -p configs/local
cp prediction/config.yaml configs/local/predict.yaml
```

Edit that file. Relative paths are resolved **from the YAML file's directory**;
update them after copying. In particular, set:

| Setting | Value to provide |
|---|---|
| `output_dir` | A new directory for this analysis |
| `sample.bam` | BAM with index |
| `sample.bedmethyl`, `sample.pileup_log` | Matching modkit outputs, unless pileup is enabled |
| `sample.sample_id`, `sample.cell`, `sample.reference_cell`, `sample.condition` | Explicit sample metadata; condition is `WT` or `IVT` |
| `references.fasta` | Matching softmasked genome FASTA with FAI |
| `references.catalog` | Matching reference catalog, or `null` for the genome default |
| `references.gtf` | Optional GENCODE GTF/GFF3 for output annotation |
| `model_root` | Directory containing `adenine/model.keras` and `adenine/run.json` |

For the configuration copied to `configs/local/predict.yaml`, point to the
included deployment model with:

```yaml
model_root: ../../models/pooled_HEK293T_HeLa_20261111
model_directories: {}
```

The original `prediction/config.yaml` already selects this model with a path
relative to its own directory. Existing local configurations retain their
explicit model selections.

HEK293 and HEK293T are distinct reference-cell values. Original site AF, call
counts and provenance are retained. ML confidence thresholds, site AF thresholds
and NN scores have separate meanings.

The public prediction and Docker templates enable two independent m6A retention
rules:

```yaml
filtering:
  reference_only_targets: [m5C, pseU]
  m6a_drach_keep: true
  m6a_mettl16_keep: true
```

The latter retains original m6A candidates matching RNA `UAC[m6A]GAGAA`
(DNA `TACAGAGAA`, candidate A at position 4) in the candidate's strand orientation,
even when the neural network rejects them. Only this exact nine-base motif is
used; the broader `UAC[m6A]GARAA` pattern is not retained by this rule.
Candidates must first pass the normal depth, modified-read-count and AF criteria.
The motif rule neither adds missing sites nor creates m6A calls from inosine-only
candidates. Like DRACH retention, it applies to original m6A candidates in WT
and IVT samples. Set `m6a_mettl16_keep: false` to disable it; an omitted key
defaults to `false` for existing configurations. This additional retention policy
does not change the trained network or the reported classifier-only results.

```bash
# Inspect configuration without opening BAM or starting inference.
rnamodscan predict --config configs/local/predict.yaml --device cpu --dry-run

# Run preparation, inference and filtering.
rnamodscan predict --config configs/local/predict.yaml --device cpu --no-recalibrate

# Alternatively, use an available GPU, falling back to CPU.
rnamodscan predict --config configs/local/predict.yaml --device auto --no-recalibrate
```

Choose one full-run command per new output directory. Use `--recalibrate` in place
of `--no-recalibrate` to also write a new BAM with rejected target ML values set
to zero. The original BAM is not overwritten.

CPU preparation, inference and filtering can instead run as separate phases:

```bash
rnamodscan predict --config configs/local/predict.yaml --stage prepare
rnamodscan predict --config configs/local/predict.yaml --stage infer --device cpu
rnamodscan predict --config configs/local/predict.yaml --stage filter --no-recalibrate
```

### Read the output flags

The principal output is `<output_dir>/predictions/bedmethyl/filtered.bed`. It keeps
the original bedMethyl columns and appends five columns explaining why each
original candidate was retained:

| Added column | Meaning |
|---|---|
| `flag_db` | `1`: matches an applicable database site in a WT sample |
| `flag_short_read` | `1`: matches an applicable published experimental site in a WT sample |
| `flag_nn` | `1`: the neural network classifier predicts the candidate's original modification |
| `motif_keep` | `0`: neither enabled motif rule applied; `1`: DRACH; `2`: canonical METTL16 motif |
| `filter_pass` | `1`: at least one retention rule passed; only these rows are exported |

Several flags can support the same candidate. A zero indicates that the
corresponding retention rule did not apply; it does not establish that the site
is unmodified. In particular, `motif_keep=0` can reflect a disabled motif rule,
and the candidate can still pass through another flag. Comment headers explain
the columns in combined and modification-specific BED files. Detailed TSVs keep
the two motif flags separately and include rejected candidates and source IDs.
See the [output flag guide and worked example](docs/OUTPUT_FLAGS.md) and the
[prediction guide](prediction/README.md).

## Genome and reference preparation

| `genome` | Database overlap | Experimental reference overlap |
|---|---|---|
| `hg38` | Human RMBase 3.0 | Explicit matching-cell references |
| `mm10` | Mouse RMBase 3.0 only | Disabled |
| `other` | Disabled | Disabled |

Set `genome` in YAML or override it with `--genome`. A matching softmasked FASTA
and FAI are required during preparation for every genome, including workflows
using only reference-based filtering. No liftover or chromosome renaming is
performed. For `other`, m5C and pseudouridine return no sites because references
are disabled. Cross-species NN performance has not been evaluated.

The four mouse RMBase source archives are included. Generate their ignored BED
and provenance files once after cloning:

```bash
python analysis/reference_catalog/prepare_mouse_reference_catalog.py
```

Prepared human references are included in
[`data/references/hg38/`](data/references/hg38/README.md): 17 compressed BEDs,
three original provenance JSON files, a portable catalog and checksums. They
cover human RMBase 3.0 and the configured HEK293T/HeLa experimental references.
The complete runtime data occupy about 12.3 MiB; the largest BED is about 6.8 MiB.
The existing reader opens `.bed.gz` directly, so no extraction or reference
rebuilding is needed after cloning. Verify the bundle from the repository root:

```bash
(cd data/references/hg38 && sha256sum -c SHA256SUMS)
```

For `genome: hg38`, `references.catalog: null` selects the included
`data/references/hg38/catalog.json`. A supplied catalog can override it. The
legacy `data/references/catalog.json` and original imported files are preserved
for existing local analyses; that legacy catalog is not portable on its own.
The human bundle is not used for `mm10` or `other`. See
[reference data](source/README.md) and the
[reference preparation guide](analysis/reference_catalog/README.md).

In particular, `source/hg38.m6A.tar.gz` is the RMBase source archive used to
generate the human m6A BED; it is obtained separately when importing references.
It is unnecessary when using the included prepared references. Prediction uses
the prepared BED and the catalog's provenance JSON. The large
`references/rmbase3/m6A.provenance.tsv` is a detailed import audit and is not
required by the prediction runtime. It remains excluded from Git. Neither the
archive nor the detailed TSV is part of the model weights.

## Training and development

You can train new weights using your own **GLORI, Slic-seq, RMBase or other
positive-site results**. GLORI supplies m6A examples; Slic-seq supplies inosine
examples; a database or another assay can supply either target. The current
network jointly classifies m6A, inosine and background, so it needs positive
examples for **both modifications** and paired WT/IVT sequencing inputs. A
positive-site file alone cannot replace the tagged BAMs and negative control.
m5C and pseudouridine remain reference-only targets.
Sources can be mixed: for example, use GLORI m6A sites with database inosine
sites, or Slic-seq inosine sites with database m6A sites.

Create a reference catalog and a fresh configuration, then check the inputs:

```bash
mkdir -p configs/local
# Edit this manifest's assembly, cell/tissue and positive-site file paths.
cp training/examples/reference_sources.yaml configs/local/reference_sources.yaml
rnamodscan references build --manifest configs/local/reference_sources.yaml \
  --output data/training_references/my_study
rnamodscan references check \
  --catalog data/training_references/my_study/catalog.json --training

rnamodscan train init --output configs/local/train.yaml --cell my_tissue \
  --catalog data/training_references/my_study/catalog.json \
  --fasta /absolute/path/to/softmasked.fa \
  --input-dir /absolute/path/to/WT_IVT_inputs --run-dir results/my_training
# Edit the generated sample filenames before checking:
rnamodscan train --config configs/local/train.yaml --dry-run
rnamodscan train check --config configs/local/train.yaml
# Starts extraction and fitting; run only after the check succeeds:
rnamodscan train --config configs/local/train.yaml
```

For human HEK293T/HeLa data, the included `data/references/hg38/catalog.json`
can be used directly instead of building a catalog. `train init` does not start
training or overwrite an existing configuration. It sets
`selection.source_dir: null` and selects a fresh output directory.
`train check` is read-only. The prediction Docker images do not include the
training package; use the environment in the [training guide](training/README.md).

### Adapting to another species or assembly

Build a catalog declaring your actual assembly/species and supply matching
WT/IVT BAMs, modkit outputs and a softmasked FASTA/FAI. Training accepts custom
experimental or database sources, including mouse assays. Columns and
coordinate conventions must be explicit: no liftover, chromosome renaming or
human-label transfer is performed. Gene annotation is optional and is not an
NN input.

After cross-validation, new deployment weights are written to
`<output_dir>/final_model/adenine/`. Select them explicitly in prediction with
`model_root: /absolute/path/to/your_run/final_model`. Included human weights are
never replaced. Prediction reference policies remain unchanged: `mm10` uses
mouse RMBase only; `other` uses the network without database/short-read rescue.
See the [other-species walkthrough](docs/TRAINING_CUSTOM_DATA.md#another-species-or-assembly)
for prediction YAML and motif-rule settings. Cross-species performance of the
included human model has not been established.

The [custom-data guide](docs/TRAINING_CUSTOM_DATA.md) covers column mappings,
database-only references, controls, outputs and deployment. References construct
operational weak labels: an absent match is never made a negative, and overlap
is not independent chemical validation. The 23,731-parameter model architecture
is unchanged.
The [NN design](docs/NN_DESIGN.md), [label preparation](docs/TRAINING_SET_PREPARATION.md)
and [candidate extraction](docs/CANDIDATE_EXTRACTION.md) document the scientific rules.

```text
rnamodscan/    Command-line entry point
training/     Label preparation, grouped CV and model fitting
prediction/   Candidate preparation, inference, filtering and optional BAM output
utils/        Shared NN, feature extraction, reference and IO utilities
tests/        Automated tests with synthetic inputs
analysis/     Reference preparation and analysis utilities
source/       Reference provenance, checksums and mouse source archives
docs/         Design and workflow documentation
```

Virtual environments, credentials, large generated data, training outputs, local
manuscript files and manuscript-generation scripts are excluded from Git. The
verified deployment bundle above is the explicitly included model. Source code is
distributed under the [MIT License](LICENSE); third-party reference datasets
retain their own terms and attribution.
