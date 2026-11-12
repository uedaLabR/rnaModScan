# Training

The implementation lives in `training/`; shared NN, BAM, reference and IO code
lives in `utils/`. The public entry point is `rnamodscan train`. Numbered scripts
under `training/test/` expose individual development stages; automated tests
live in the root `tests/` directory.

## Environment and inputs

Use the Python 3.10/TensorFlow 2.10.1 environment in the
[installation guide](../README.md#installation) for the current model lineage.
Training curves additionally require matplotlib; the working environment used
`matplotlib==3.8.4`:

```bash
python -m pip install matplotlib==3.8.4
```

`requirements-nn.txt` and the `training` extra select a newer TensorFlow and
should be used in a separate environment. Existing TF2.10 models cannot be
loaded directly by TensorFlow 2.19/Keras 3.

Create a fresh local configuration. The existing `training/config.yaml` is a
study-specific development record and is preserved:

```bash
mkdir -p configs/local
rnamodscan train init --output configs/local/train.yaml --cell HEK293T \
  --catalog data/references/hg38/catalog.json \
  --fasta /absolute/path/to/softmasked/hg38.fa \
  --input-dir /absolute/path/to/WT_IVT_inputs --run-dir results/my_training
```

For custom references or another species, pass your catalog instead. See the
[custom-data guide](../docs/TRAINING_CUSTOM_DATA.md) for GLORI, Slic-seq, RMBase
and arbitrary positive-site BED/TSV/CSV inputs. Paired WT/IVT BAMs, indexes,
bedMethyl/logs and an assembly-matched softmasked FASTA/FAI remain required.
Positive references must cover both m6A and inosine to fit the joint model.

`init` writes only a new YAML and refuses existing configurations/run outputs.
It sets `selection.source_dir: null`, five folds and a fresh run directory.
Default sample IDs/filename stems are `sample_WT` and `sample_IVT`; edit YAML
or pass `--wt-id`/`--ivt-id`. `--reference-cell` selects an explicit catalog
cell/tissue mapping. Paths are relative to the generated YAML. Add more WT/IVT
pairs with distinct cell keys if needed; the current pairing accepts one WT and
one IVT record per cell key, not duplicate replicate records under the same key.

Annotation is reporting-only: `references.gtf: null` skips it in the full
workflow. An optional same-assembly GTF or GENCODE-style GFF3 must contain
gene/transcript identifiers. Review `samples`, `cv` and `nn` before execution;
default training settings are starting values, not validated for every species.

## Stages

Inspect the plan before starting a new analysis:

```bash
rnamodscan train --config configs/local/train.yaml --dry-run
rnamodscan train check --config configs/local/train.yaml
```

The dry run opens no input data. The read-only `check` command validates paths,
indexes, dependency imports, catalog metadata, BAM/FAI contig lengths and output
collisions without scanning alignments or starting jobs. It returns status
`ready`/exit 0 or `not_ready`/exit 2 with errors. `--skip-dependencies` performs
only path/metadata checks; it does not establish runtime readiness. Preparation
validates reference contents/checksums and actual label classes.

The all-stage command performs preflight, then starts site construction,
selection, feature extraction, optional annotation, grouped CV splitting,
fitting and final-model export:

```bash
rnamodscan train --config configs/local/train.yaml
```

Individual stages can instead be selected with `--stage sites`, `select`,
`features`, `annotation`, `split`, `fit` or `final`. Use each stage with matching
inputs and a consistent output directory. Additional inspection stages are
listed by `rnamodscan train --help`.

Explicit `--stage annotation` still requires a GTF. Choose a new `output_dir`
for a separate full analysis. Missing classes stop before BAM feature extraction.
The final stage requires all five completed folds; with `final_epochs: null`
it selects the median best-validation epoch count. Inspect held-out results in
`models/adenine/fold_i/test/metrics.json` and training histories under `train/`.

Overlapping genomic windows are grouped across samples, cells, conditions,
strands and modifications before fold assignment. m6A and inosine candidates
form one adenine dataset with classes `m6A`, `inosine`, `false_positive`. Missing
reference overlap is not a negative label. Dual-positive sites are excluded and
audited; class weights are computed from training data only.

The default CV configuration pools the configured cells. For HEK293T-only
training with HeLa held separately, prepare the cell manifests with
`--stage cell_dataset` and review the dedicated
`training/test/07_train_hek293t_model.py --help` entry point. The all-stage command
does not select that separate workflow automatically.

Final prediction bundles contain `model.keras` and `run.json`, including the
input schema, class order, model configuration and checksum. The current model
uses `sequence_reads_repeat18_v1` and has 23,731 parameters. New weights are
written to `<output_dir>/final_model/adenine/`; set prediction `model_root` to
the parent `final_model` directory. They have no new held-out evaluation after
refitting; report the earlier CV test results. The pretrained human deployment
bundle is included separately; raw training datasets/completed CV runs remain
local archives. Existing models are not overwritten.

The CPU/GPU Dockerfiles are prediction-only images. For CPU training in a new
Python 3.10 environment, install `tensorflow-cpu==2.10.1`,
`-r docker/requirements-runtime.txt` and `matplotlib==3.8.4`, then install the
project with `python -m pip install --no-deps -e .`. For GPU training use
`tensorflow==2.10.1` instead, with the compatible host CUDA/cuDNN libraries.
Do not install both TensorFlow distributions in one environment or combine
them with the modern extras.

See [label preparation](../docs/TRAINING_SET_PREPARATION.md),
[candidate extraction](../docs/CANDIDATE_EXTRACTION.md),
[NN design](../docs/NN_DESIGN.md) and
[site Parquet](../docs/SITE_BAG_PARQUET.md) for the rules and formats. Weak-label
CV and reference overlaps must not be presented as independent chemical accuracy.
