# rnaModScanの集計と確認

- `hela_m6a/`: HeLaのGLORI・m6Anet・本法NNのDRACH別比較、座標変換、ベン図、累積Excel。本文m6Anet≥0.8、サプリ≥0.9。最終レスキューは説明文のみ。
- `pileup_overlap/`: WT/IVT bedMethylのcall・DRACH・short-read/RMBase一致集計と確認テスト。
- `reference_catalog/`: short-read BEDの再構築、RMBase取込、参照カタログ登録。
- 各工程の結果はそれぞれの `results/` 配下。学習とは独立した工程です。
- 共通処理は `rnamodscan/analysis/`、`rnamodscan/references/`、`rnamodscan/pileup/` に置きます。
- BED6/bedMethyl読込は `rnamodscan/io/bed.py` を学習準備・将来のfilteringと共用します。
- 学習候補のTSV・ラベル内訳の集計・8修飾BAM特徴のParquet化は `training/README.md` を参照してください。
- データ本体は従来の場所から直接参照し、ここへ複製しません。


## Frozen NN input-group sensitivity

`analysis/feature_reliance/README.md` describes the saved-model dependence analysis for reference sequence, repeat masks, neighboring modification scores and quality. It reads existing CV test bags, fixes weights and preserves masks/read sampling. The results are conditional perturbation sensitivity against operational labels, not causal importance or independent chemical accuracy. `prediction/test/12_analyze_feature_reliance.sh` supports checks and detached runs; use a fresh output directory.
