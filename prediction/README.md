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
analysis uses RMBase 3.0 only. `other` disables both reference types, but enabled
motif rules can still retain m5C/pseudouridine candidates. Preparation requires an assembly-matched
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

m6A and inosine use a shared neural network classifier with three output classes:
m6A, inosine and false-positive candidates. The classifier supports retention
when its highest-scoring class matches the candidate's original modification.
A candidate passes if any of its independent evidence flags is set: database
overlap, experimental reference overlap, the classifier's decision or one of six
modification-specific motif rules. Reference flags apply only to WT samples.
m5C and pseudouridine have an auxiliary reporting function using known-site
overlap or enabled motif rules, without neural-network scores. Known-site
retention keeps the original depth ≥10, target-modified reads ≥3 and AF ≥0.10
defaults. Only the m5C/Ψ motif route requires depth ≥20 and AF ≥0.40, with the
same original modified-read requirement. m6A/inosine defaults are unchanged.
A candidate for another modification is never manufactured from an NN prediction.

Configure the extra m5C/Ψ motif gate under `filtering`:

```yaml
filtering:
  auxiliary_motif_min_depth: 20
  auxiliary_motif_min_af: 0.40
```

These defaults also apply when the keys are omitted. The final filter rechecks
the motif gate against original site counts, including for older saved routing.
`all_predictions.tsv` retains original candidates and reference-overlap facts.
For m5C/Ψ, `auxiliary_motif_eligible=0` disables motif flags/codes 3–6, while
known-site flags and PASS can still be 1 under the original candidate criteria.
For example, a known m5C candidate with 10 reads and 3 modified reads can pass
with `motif_keep=0`, even if its sequence also matches a motif.

The public template enables `filtering.m6a_drach_keep` and
`filtering.m6a_mettl16_keep`, plus the four m5C/Ψ rules below.
`m6a_mettl16_keep` retains an original m6A candidate when
its strand-oriented sequence is `UAC[m6A]GAGAA` (DNA `TACAGAGAA`, candidate
adenine at position 4), independently of the neural-network class. The expanded
`UAC[m6A]GARAA` motif is not included. Both motif rules apply to WT and IVT
m6A candidates that already passed the depth, modified-read-count and AF
thresholds; they do not generate missing sites or another modification's rows.
Set either key to `false` to disable that rule. An omitted
`m6a_mettl16_keep` defaults to `false`, preserving existing YAML behavior.
Sequence-based retention does not establish METTL16 dependence or chemical
accuracy. Previously reported NN-only results remain separate from this final
retention policy.

| Key under `filtering` | Motif code | Original target and RNA pattern |
|---|---|---|
| `m5c_nsun6_keep` | `3` | m5C, NSUN6-associated `[m5C]UCCA` |
| `m5c_nsun2_keep` | `4` | m5C, NSUN2-associated G-rich `[m5C]NGGG` |
| `pseu_trub1_keep` | `5` | pseudouridine, TRUB1-associated `GU[Ψ]CNANNC` |
| `pseu_pus7_keep` | `6` | pseudouridine, classic PUS7-associated `UG[Ψ]AR` |

These four keys are enabled in the supplied prediction/Docker templates and
default to `false` if omitted from older YAML. They apply to WT and IVT
candidates that passed the original candidate thresholds and the extra
m5C/Ψ motif depth/AF gate, using the matching strand-oriented FASTA. The legacy `reference_only_targets` name denotes the
NN-free branch; m5C/Ψ motif rules also apply there. These sequence-only rules
do not test secondary structure or confirm modification/writer dependence.
N={A,C,G,U}, R={A,G}; brackets locate the candidate. The broader PUS7
`UN[Ψ]AR` pattern is not used.

Important files under `output_dir`:

- `predictions/bedmethyl/filtered.bed`: retained original bedMethyl rows, with
  `flag_db`, `flag_short_read`, `flag_nn`, `motif_keep`, `filter_pass` appended.
  `motif_keep` is `0` for no motif retention, `1` for DRACH, `2` for canonical
  METTL16 and `3`–`6` for the m5C/Ψ motifs listed above. Comment headers describe the columns
  and codes in both combined and modification-specific BED files.
- `predictions/bedmethyl/<target>/filtered.bed`: modification-specific rows.
- `predictions/<target>/`: retained sites, exclusion audits and summaries.
- `scores/adenine/run.json`: inference runtime, TensorFlow version and device.
- `run.json`: workflow status and completed stages.

Original AF, counts and provenance are retained. ML confidence, site AF and NN
scores are separate quantities. Reference rescue and reference overlap do not
establish independent chemical accuracy.
Detailed TSV audits keep nine independent binary flags: DB, experimental
reference, NN, `flag_drach`, `flag_mettl16`, `flag_nsun6`, `flag_nsun2`,
`flag_trub1` and `flag_pus7`. The public BED combines the six motif flags
into the single `motif_keep` code, preserving five appended columns.

The [output flag guide](../docs/OUTPUT_FLAGS.md) explains every added column,
the difference between an inapplicable rule and a negative chemical result, and
how several flags can retain the same candidate. It includes a small bedMethyl
example.

BAM output is optional and disabled by default. Use `--recalibrate` with a full
run or the filter phase to write a new BAM in which rejected target ML values
are zeroed. Candidates retained by any enabled motif rule remain retained in
this BAM output. The input BAM is preserved.

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
`filtering.m6a_drach_keep` and `filtering.m6a_mettl16_keep` unless appropriate to
that species and experiment. If including m5C/Ψ targets, also review or disable
the four motif rules above for the species. The included human weights are preserved and are
not automatically replaced by training.
