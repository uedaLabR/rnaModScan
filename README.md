# rnaModScan

rnaModScan filters RNA modification candidates from Oxford Nanopore direct RNA
sequencing. It combines Dorado MM/ML calls, reference sequence and read features
with a shared neural network for **m6A, inosine and false-positive candidates**.
m5C and pseudouridine (`pseU`) are supported through reference overlap only.

The workflow starts from a sorted, indexed BAM and modkit bedMethyl candidates.
For each original candidate, database overlap, matching-cell experimental
reference overlap, the neural-network decision and the m6A DRACH rule are
recorded separately. Their OR determines retention. Reference rescue applies to
WT samples. The input BAM is preserved; writing a new filtered BAM is optional.

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

The principal output is `<output_dir>/predictions/bedmethyl/filtered.bed`, with
original bedMethyl columns followed by `flag_db`, `flag_short_read`, `flag_nn`,
`flag_drach` and `filter_pass`. Modification-specific files, retained-site tables,
exclusion audits and summaries are also saved. See the
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

Training uses paired WT/IVT samples and experimental references to construct weak
labels, followed by grouped cross-validation and a joint adenine model. The
current model has 23,731 parameters and combines reference sequence/repeat
features with read features. Reference overlap and weak-label metrics are not
independent measurements of chemical accuracy.

Use a local copy of `training/config.yaml`, replace all study-specific paths,
set `selection.source_dir: null` for new site-list construction, and select a new
`output_dir`. Start by inspecting the configuration:

```bash
cp training/config.yaml configs/local/train.yaml
# Edit configs/local/train.yaml before running:
rnamodscan train --config configs/local/train.yaml --dry-run
```

The [training guide](training/README.md) describes the stages and dependencies.
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
