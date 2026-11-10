# Frozen-model feature reliance

This analysis asks how the existing classifier depends on its inputs. It does
not refit a model, revisit BAM, change labels or overwrite old predictions.
The default is every held-out test observation from fold0 and its existing best
checkpoint. Other folds can be selected with `--fold`.

```bash
bash prediction/test/12_analyze_feature_reliance.sh --check
bash prediction/test/12_analyze_feature_reliance.sh --background
```

The default new output is `results/feature_reliance_fold0_20261109`. All outputs
and launch records must be new. A detached launch survives terminal disconnects.
Its sibling `.launch.log`, and output `progress.json`, show progress. The old
training and HeLa analysis directories remain untouched.

Use a separate output for a small technical pilot:

```bash
bash prediction/test/12_analyze_feature_reliance.sh \
  --max-sites 512 --output results/feature_reliance_pilot_fold0_20261109 \
  --device gpu
```

`--max-sites` selects observations uniformly without labels and is explicitly
marked as a pilot. It is not a full-test performance estimate. CPU mode is
available with `--device cpu`; GPU mode requires a working GPU and does not
silently fall back. Default TensorFlow parallelism is four intra-operation
threads and one inter-operation thread, with GPU memory growth enabled.

Preparation reads existing site Parquet once to collect reference windows and
small empirical donor pools. Inference reads the same selected bags once and
scores baseline plus seven replacements at seeds17,43,101 (22 versions/site).
Input read masks, depths, source reservoirs and padding are retained. A larger
256-site CPU preparation batch amortizes pool collection, while source inference
batches have16sites. Compatible versions are stacked up to128sites per graph
call. Every call reserves one baseline anchor and fills unused slots with
baseline copies, preserving the same tensor dimensions and real-read count
across interventions. Anchor probabilities must remain within1e-6 and their
argmax classes must agree across all chunks and every site. Full-cohort padded
read tensors are never saved or held in memory.

The reference conditions exchange coherent overlapping3-mer windows, repeat
windows, or both together, within retained-depth bins. Read conditions replace
the eight ML channels jointly, or positive inverse-quality observations, using
small empirical pools stratified by observed base, offset and retained depth.
The primary `neighbor_ml` condition excludes the central candidate position.
`center_ml` and `all_ml` are separate diagnostics, so direct candidate confidence
is not confused with surrounding context. Unknown/absent/padded positions and
missing-quality zeros are protected as documented in `perturbations.py`.

`scores.npy` contains float32 probabilities in `[version,site,class]` order;
`sites.tsv` and `run.json` identify every input and version. The baseline is
recomputed in the same process. Its comparison with the earlier saved scores is
a diagnostic, not the baseline used to compute performance drops. Original
checkpoint, split, index and extraction-manifest SHA values are checked again
before completion. The protocol and relevant analysis/NN/reader source files
are copied into `code_snapshot/` and fingerprinted before computation. Every
condition must have a finite score for every site.

Metrics include operational-label log loss, multiclass accuracy/macroF1,
per-class precision/recall/F1, DRACH/non-DRACH m6A subsets, class changes, and the
signed change in true-label probability. Variation across three randomizations
is not variation across five CV folds. The score group changes are correlated
and can produce combinations absent from training: these are model-reliance
estimates under the named replacements, not causal biological importance.
They do not partition a total contribution into percentages summing to100%.
