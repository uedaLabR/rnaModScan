# Prediction

Install the runtime described in the [main README](../README.md#installation).
The included pooled HEK293T + HeLa model requires TensorFlow 2.10.1/Keras 2.10.0 on Python 3.10.
The modern `prediction-cpu` extra is not compatible with that saved model.
The [deployment bundle](../models/pooled_HEK293T_HeLa_20261111/README.md) contains
the trained weights and runtime metadata. Prepared human reference BEDs and
provenance JSON are also [included](../data/references/hg38/README.md). Supply a
matching genome FASTA/FAI and your own sample inputs.

## Configure an analysis

Copy `config.yaml` to an ignored local configuration, then edit it:

```bash
mkdir -p configs/local
cp prediction/config.yaml configs/local/predict.yaml
```

Relative paths are resolved from the YAML file, so update them after copying.
Use a new `output_dir` and set the following fields:

| Field | Required input |
|---|---|
| `sample.bam` | Sorted BAM with MM/ML tags and its index |
| `sample.bedmethyl`, `sample.pileup_log` | bedMethyl and the log from the same modkit pileup |
| `sample.sample_id`, `cell`, `reference_cell`, `condition` | Explicit sample metadata; `WT` or `IVT` |
| `references.fasta` | Matching softmasked FASTA with FAI |
| `references.catalog` | Assembly-matched catalog, or `null` for the genome default |
| `references.gtf` | Optional GENCODE GTF/GFF3 for reporting |
| `model_root` | Parent directory of `adenine/model.keras` and `adenine/run.json` |

For `configs/local/predict.yaml`, select the included model with:

```yaml
model_root: ../../models/pooled_HEK293T_HeLa_20261111
model_directories: {}
```

The template at `prediction/config.yaml` selects the same bundle relative to its
own location. Explicit model paths in existing analysis configurations are kept.

With `modkit.run_pileup: false`, existing pileup outputs are read without
rethresholding. To create them from the BAM, install modkit separately and set
`modkit.run_pileup: true`. See the [modkit workflow](../docs/PREDICTION_MODKIT_GROUPS.md).

Use `genome: hg38`, `mm10` or `other`; `--genome` overrides YAML. Human analysis
uses human RMBase and explicitly matching-cell experimental references. Mouse
analysis uses RMBase 3.0 only. `other` disables both reference types and returns
no m5C/pseudouridine sites. Preparation always requires an assembly-matched
softmasked FASTA and FAI. There is no automatic liftover or chromosome renaming.
See [reference preparation](../source/README.md).

For human prediction, leave `references.catalog: null` to select the included
`data/references/hg38/catalog.json`. Its compressed BEDs are read directly; no
download of the original RMBase archive or manual decompression is required.
They preserve the original coordinates, cell scopes and source assignments.

## Run

```bash
rnamodscan predict --config configs/local/predict.yaml --device cpu --dry-run
rnamodscan predict --config configs/local/predict.yaml --device cpu --no-recalibrate
```

`python -m rnamodscan predict` is equivalent. `--device auto` uses a GPU if
available and otherwise CPU; `--device gpu` requires a GPU. `--cpu-threads 4`
sets TensorFlow intra-operation threads and does not change extraction workers.
The dry run reports planned paths and unset inputs without opening data files.

A full run performs CPU candidate/feature preparation, inference in a separate
process, and filtering. The phases can also run separately, using the same
configuration and one output directory:

```bash
rnamodscan predict --config configs/local/predict.yaml --stage prepare
rnamodscan predict --config configs/local/predict.yaml --stage infer --device cpu
rnamodscan predict --config configs/local/predict.yaml --stage filter --no-recalibrate
```

Inference reads only prepared NN inputs and the model. It does not reopen BAM,
FASTA or reference databases. Existing completed outputs are preserved; choose a
new output directory for a separate analysis.

## Decisions and outputs

m6A and inosine use a shared three-class softmax model. A candidate passes if any
of its independent evidence flags is set: database overlap, experimental
reference overlap, matching NN argmax, or DRACH for m6A. Reference flags apply
only to WT samples. m5C and pseudouridine are reference-only targets. A candidate
for another modification is never manufactured from an NN prediction.

Important files under `output_dir`:

- `predictions/bedmethyl/filtered.bed`: retained original bedMethyl rows, with
  `flag_db`, `flag_short_read`, `flag_nn`, `flag_drach`, `filter_pass` appended.
- `predictions/bedmethyl/<target>/filtered.bed`: modification-specific rows.
- `predictions/<target>/`: retained sites, exclusion audits and summaries.
- `scores/adenine/run.json`: inference runtime, TensorFlow version and device.
- `run.json`: workflow status and completed stages.

Original AF, counts and provenance are retained. ML confidence, site AF and NN
scores are separate quantities. Reference rescue and reference overlap do not
establish independent chemical accuracy.

BAM output is optional and disabled by default. Use `--recalibrate` with a full
run or the filter phase to write a new BAM in which rejected target ML values
are zeroed. The input BAM is preserved.

The HEK293T-only model and HeLa prediction run used in development are complete;
those local datasets and results are not distributed here. In the saved HeLa
benchmark, NN inference for 507,501 prepared sites took 26.28 minutes on CPU
versus 22.73 minutes on GPU. These are inference measurements; the full CPU
workflow time was estimated, not measured. Cross-species performance remains
unevaluated.

## Use your own trained model

See [custom-data and other-species training](../docs/TRAINING_CUSTOM_DATA.md).
Training exports `<training_output>/final_model/adenine/{model.keras,run.json}`;
set `model_root` to its parent `final_model` directory, or point
`model_directories.adenine` directly to `adenine`. Use the same compatible
TensorFlow runtime as training; the loader checks schema, class order and checksum.

Select prediction genome and references explicitly. For a custom assembly,
`other` disables DB/short-read rescue and requires your matching FASTA/FAI.
Use `targets: [m6A, inosine]`, `references.gtf: null` and disable
`filtering.m6a_drach_keep` unless appropriate to that species. The included
human weights are preserved and are not automatically replaced by training.
