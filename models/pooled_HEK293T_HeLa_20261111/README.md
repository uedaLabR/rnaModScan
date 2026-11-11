# Pooled HEK293T + HeLa deployment model

This bundle contains the completed 11 November 2026 deployment model. Cloning
the repository supplies the two runtime files without retraining:

```text
adenine/model.keras   Trained weights, 510,936 bytes
adenine/run.json      Runtime schema, class order, training metadata and checksum
SHA256SUMS           SHA-256 checksums for both files
```

Verify from this directory:

```bash
sha256sum -c SHA256SUMS
```

The weights have SHA-256
`b14c6836f2d07b49077cc8beab5ff4856bf9ce613f8004c1d2640b488ddea59e`.
Both runtime files are byte-identical copies of the verified training output.
Historical training paths in `run.json` record provenance; those directories are
not needed to load this portable bundle.

## Training and intended use

- All 548,735 labelled HEK293T/HeLa WT/IVT observations were used for deployment
  fitting: 194,592 m6A, 7,036 inosine and 347,107 false-positive/background labels.
- The model has 23,731 parameters and class order `m6A`, `inosine`, `false_positive`.
- Learning rate 0.0003; batch size 16; maximum 300 sampled reads per site; seed42.
- Five fixed epochs were selected as the median development-CV checkpoint epoch.
- Runtime: Python 3.10, TensorFlow 2.10.1 and Keras 2.10.0. Use the pinned
  installation in the [main README](../../README.md#installation).

This deployment refit used all labelled data and has no new held-out performance
estimate. Reported validation and HeLa comparisons use their earlier evaluated
models; the HeLa analysis used HEK293T-only weights. The original HEK293T-only
weights, configurations and results are preserved separately. Cross-species
performance remains unevaluated.

## Select the model

The prediction template already points here. If the YAML is at
`configs/local/predict.yaml`, use paths relative to that file:

```yaml
model_root: ../../models/pooled_HEK293T_HeLa_20261111
model_directories: {}
```

Alternatively, set `model_directories.adenine` to the absolute path of this
bundle's `adenine` directory. The loader verifies the model checksum and feature
schema. Prediction still requires the user's tagged BAM, bedMethyl/log inputs,
matching softmasked FASTA/FAI and configured reference data. m5C and pseudouridine
use reference overlap rather than this neural classifier.
