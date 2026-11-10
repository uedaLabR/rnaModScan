# 実行順（2026-11-06）

学習は[training/README](../training/README.md)、実解析は[prediction/README](../prediction/README.md)が入口です。

| 工程 | 入力 | 出力 | 主なPython |
| --- | --- | --- | --- |
| 部位一覧/選択 | modkit bedMethyl・参照catalog・WT/IVT | sites/selection TSV・集計 | training/site_lists.py, adenine_sites.py |
| 特徴抽出 | 選択TSV・BAM/BAI・softmasked FASTA/FAI | chr/strand別部位Parquet v3・軽い索引 | utils/features/bam.py, site_bags.py |
| 領域注釈（任意） | 索引・GENCODE GTF | 報告用gene/region | utils/features/position.py |
| CV/HEK単独分割 | 部位位置・sample | splits/cell_datasets | training/cross_validation.py, cell_datasets.py |
| fold学習 | 同じ部位Parquet・分割 | model/history/validation/test | training/fit.py |
| 最終モデル | 全選択Parquet・CV最良epoch | final_model/adenine | training/final_model.py |
| 実解析候補 | 既存またはmodkit生成bedMethyl | 候補TSV | prediction/candidates.py |
| 実解析抽出/準備 | 候補・BAM・softmasked FASTA | 同じParquet v3→prepared v9 | prediction/pipeline.py, site_bag_inputs.py |
| 推論 | prepared v9・最終モデル | scoreのみ | prediction/scoring.py |
| レスキュー/出力 | score・候補metadata・DB | 修飾別TSV/BED・任意ML0 BAM | prediction/adenine_filtering.py, bam_recalibration.py |

既存選択一覧からの再抽出は`training/test/00_rebuild_site_parquet.py --config training/config.yaml --background`です。
完了後、最初の評価は`training/test/05_train_fold0.py --config training/config.yaml --background`を使います。
抽出とNN学習は別処理です。新しい保存先training_v10へ出し、停止済みの旧runを再開しません。
mask別準備・NPY生成・RepeatMasker class注釈は不要です。FASTAの小文字情報を保持してください。
