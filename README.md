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

The current HEK293T model was trained and used with **Python 3.10 on Linux
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

**Model weights, human reference datasets and analysis outputs are not included
in this Git repository. A public pretrained-model download has not yet been
provided.** The model bundle must be supplied separately or produced by training.
Renaming a model file does not establish compatibility: the loader checks its
input schema, class order and checksum.

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

Human archives and imported reference BED/provenance files are excluded from
Git. The included human catalog records the original data sources, relative
paths and checksums; it is not a bundle of those datasets. Prepare the files or
provide an assembly-matched catalog before using `hg38`. See
[reference data](source/README.md) and the
[reference preparation guide](analysis/reference_catalog/README.md).

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

Virtual environments, credentials, large generated data, models, local manuscript
files and manuscript-generation scripts are excluded from Git. Source code is
distributed under the [MIT License](LICENSE); third-party reference datasets
retain their own terms and attribution.
