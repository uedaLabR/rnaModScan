# HEK293T evaluation checkpoint

This is the saved checkpoint used for the manuscript's HeLa analyses (Figures
4–5), packaged on 24 November 2026. The packaging date is not a new training date.
`adenine/model.keras` and `adenine/run.json` are byte-identical to the files in
the manuscript's `Supplementary_Software_and_Models.zip` and the original
`results/training_v10/cell_models/HEK293T/adenine/final_model/` output.

Verify both files from this directory:

```bash
sha256sum -c SHA256SUMS
```

Weight SHA-256:
`f7d4db59edc72f5df8ed361bfb1475a4d384b7a91cf06d06764dc0d55cd5af1d`.
The original `run.json` retains historical paths and checksums for provenance;
the original workstation directories are not required to load the bundle.

Weight fitting and epoch selection used HEK293T WT/IVT records: 325,971 labelled
records, eight refit epochs. This does not establish complete independence from
HeLa-derived information: architecture development used pooled cells and
RMBase teacher labels came from a pooled human catalogue. Classifier-only
metrics precede known-site/motif retention and must not be replaced with final
reporting PASS counts.

Use Python 3.10 with TensorFlow 2.10.1 and Keras 2.10.0, as documented in the
[installation instructions](../../README.md#installation). Select this checkpoint
explicitly with [the portable evaluation configuration](../../configs/HeLa_HEK293T_evaluation.yaml).
Provide the indicated genome/annotation, BAM and modkit inputs, choose a fresh
output directory, and adjust input paths before running. No inference or
analysis was repeated to prepare this bundle.

The default prediction configuration and Docker images continue to use the
separate [pooled deployment model](../pooled_HEK293T_HeLa_20261111/README.md).
To use the evaluation checkpoint in a container, mount this directory read-only
and set `model_directories.adenine` to its mounted `adenine` directory. No new
performance estimate is assigned to either checkpoint by this release.
