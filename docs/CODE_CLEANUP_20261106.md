# コード整理（2026-11-06）

現行経路を「部位Parquet v3の4入力→3class NN→予測/レスキュー」にそろえました。
結果データ、停止済みrunのログ、原稿、論文の観測値、source snapshotは削除していません。

削除・整理した範囲:

- 全read/旧flat Parquetを読むNN loaderと、後段のNN用sampling/NPYキャッシュ。
- 一時的に導入したrepeat-mask別Parquetと、その追加準備script/row join。
- repeat class/familyをNNへ渡す注釈・語彙・設定。repeatはgenome FASTA caseから同時取得。
- 旧prepare_selected/resume/recover入口と古いconfig_v3/v4。
- 旧scaffold CLI/学習セット・二値用部位選択。必要な教師部位検証だけselection_validationへ分離。
- NNの二値sigmoid、参照CNN切替、全read入力None。整理時点では33,251パラメータの構造を維持。その後、同日のユーザー指定でread枝を16次元へ軽量化し、現行は23,731パラメータ。
- 上記の旧経路だけを検証していたテスト。学習ログ/指標/HEK単独/公開予測等は現行fixtureへ移植。

共通datasetは索引・CV照合・TensorFlow4入力の定義に縮小しました。
CIGAR/MM/逆鎖/splice、reservoir、メモリー解放、BAM並列処理の検証は維持します。
低levelのraw audit読戻しはテスト専用fixtureへ隔離し、現行NNの別入力経路には使いません。
analysis/のDorado/short-read/Alu比較は論文の観測値に必要なので保持します。

実行は[training/README](../training/README.md)、[prediction/README](../prediction/README.md)を参照してください。
