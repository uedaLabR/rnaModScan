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

Create a local configuration and replace all study-specific input paths:

```bash
mkdir -p configs/local
cp training/config.yaml configs/local/train.yaml
```

Set a new `output_dir`, paired WT/IVT sample BAMs with indexes, bedMethyl and
matching pileup logs, a reference catalog, softmasked FASTA with FAI, and the
GENCODE GTF/GFF3 used for annotation. All paths are resolved from the YAML file.
For a new dataset, set `selection.source_dir: null`; the template otherwise
refers to a local, previously prepared site-list directory that is not included
in Git. Review `samples`, `cv`, `cell_dataset` and `nn` before execution.

## Stages

Inspect the plan before starting a new analysis:

```bash
rnamodscan train --config configs/local/train.yaml --dry-run
```

The all-stage command runs site construction, selection, feature extraction,
annotation, grouped CV splitting, fitting and final-model export:

```bash
rnamodscan train --config configs/local/train.yaml
```

Individual stages can instead be selected with `--stage sites`, `select`,
`features`, `annotation`, `split`, `fit` or `final`. Use each stage with matching
inputs and a consistent output directory. Additional inspection stages are
listed by `rnamodscan train --help`.

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
uses `sequence_reads_repeat18_v1` and has 23,731 parameters. Training datasets,
weights and completed development runs are not included in this repository.

See [label preparation](../docs/TRAINING_SET_PREPARATION.md),
[candidate extraction](../docs/CANDIDATE_EXTRACTION.md),
[NN design](../docs/NN_DESIGN.md) and
[site Parquet](../docs/SITE_BAG_PARQUET.md) for the rules and formats. Weak-label
CV and reference overlaps must not be presented as independent chemical accuracy.
