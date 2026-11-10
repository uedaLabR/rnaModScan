This analysis describes neighboring features in the **existing HeLa WT input**;
it does not extract BAMs, rerun the model, alter predictions, or assess accuracy.

From the repository root, run with NumPy, PyArrow and Matplotlib installed:

```bash
PYTHONPATH=/tmp/rnamodscan-paper-deps .venv310/bin/python \
  analysis/hela_features/summarize_neighbor_features.py
```

The default inputs are the prepared integer bags and historical GPU scores under
`results/prediction_HeLa_WT_HEKonly_v2`. The script records every source file's
SHA256 and checks score/metadata keys, complete input coverage, the saved model
fingerprint, and the expected original-target-qualified retained counts. It
refuses to overwrite a previous scan. For another run, specify new `--output`
and `--figure-dir` paths. `--summarize-only` can generate summaries and figures
from the saved small NPZ summaries, provided its workbook does not already exist.

Outputs are under `paper/data/hela_feature_patterns_20261109` and
`paper/figures/hela_features_20261109`. The four-sheet Excel contains all numerical
figure sources and definitions. Per-site values are in `site_feature_metrics.tsv.gz`;
the original-target flag, genomic BED0 coordinate, motif and raw NN class accompany
each row. TSV files retain full numeric precision. Displayed box plots use medians,
25th/75th percentiles and 5th/95th-percentile whiskers, without outlier points.

Each candidate contributes one site mean, so deep sites do not receive greater
weight. Only the existing sampled real reads (at most 300) are used. The candidate
center is excluded; offsets are -9 to -1 and +1 to +9 in RNA 5′ to 3′ orientation.
ML means include missing-as-zero entries exactly as in the saved model inputs.
Canonical-reference summaries condition on reference A/C/G/T as appropriate,
which reduces but does not eliminate differences in sequence/read composition.
Actual quality is obtained directly from `quality_code - 1` for positive codes.
Mismatch fraction excludes missing observations and N. The optional high-score
fraction uses a fixed ML-byte cutoff of 128 solely as a descriptive convention;
it is not a modkit confidence threshold or site AF.

The groups are defined by the NN using the same features being summarized.
Differences therefore describe associations with its decisions; they establish
neither a branch's causal contribution nor biochemical specificity/sensitivity.
No ablation, independent validation or statistical significance test is supplied.
METTL16-like centered sequence counts are exploratory; a sequence occurrence
does not demonstrate the RNA structure required for substrate recognition.

To compare every modification channel from the completed summary TSV, run:

```bash
PYTHONPATH=/tmp/rnamodscan-paper-deps .venv310/bin/python \
  analysis/hela_features/summarize_all_modification_scores.py
```

This second script reads no Parquet, BAM or model. It writes a focused two-sheet
all-channel workbook and a three-panel heatmap under the same data/figure
directories. The previous four-sheet workbook is preserved. The comparison
uses group medians, IQRs, means and SDs without a significance test or causal
feature-attribution claim. All-channel values are conditioned on each channel's
canonical reference base; differing base opportunities prevent interpreting
cross-channel absolute score levels as relative biological abundances.
