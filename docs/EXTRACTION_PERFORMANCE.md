# 現行の特徴抽出と速度（2026-11-06）

候補確定後にグループごとにBAMを取得します。複数BAMは独立sample workerで処理できます。
CIGAR索引はalignmentにつき各fetch内で一度、MM復号も最終採用readだけ各fetch内で一度行います。
最大300本を窓変換前のreservoirで選び、全readのeligible_depthは別に数えます。
FASTA窓取得時にsequenceとcase由来repeatを同時取得します。別mask/NPY準備やfit中の再encodeはありません。
完了bagを解放し、sample worker終了で参照/配列メモリーも解放します。

予測は中心距離10kbで両鎖をまとめてgroup化し、連続groupを最大group_batch_size・10 Mbの区間として1回fetchします。
単一groupの過長時も分割し、別fetchで再取得したalignmentは再計算して構いません。
区間外のread文脈を保持せず、完了部位から解放します。書き終えた部位のfeaturesもキャッシュしません。
group_workersによる並列化と整数保存/NumPy復元は[グループ抽出仕様](PREDICTION_MODKIT_GROUPS.md)を参照。Numbaは使用しません。

軽量モデルは23,731パラメータです。前の33,251パラメータモデルでは、A100/TF2.10の人工16sites×300readが約0.0454秒/batchでした。軽量モデルの速度・精度は未評価です。
この値はBAM・Parquet IO・validationを含みません。実データの全epoch速度や精度は未測定です。
最新の実行方法は[training/README](../training/README.md)です。旧resume/recovery/NPY入口は削除しました。
