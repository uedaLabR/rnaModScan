# HeLa処理時間・計算資源の記録

## 保存済みHeLa入力を使うCPU推論の実測

```bash
bash prediction/test/11_benchmark_hela_cpu.sh --check
bash prediction/test/11_benchmark_hela_cpu.sh --background
```

計測は同じ学習済みモデル・507,501部位・batch16を使う公開CLIの`--stage infer --device cpu`です。
TensorFlow intra-op4/inter-op1に設定し、GPUを非表示にします。Arrow等の補助threadもあるので、プロセス全体の4コア上限ではありません。
既存の`nn_inputs`全体を新しい計測フォルダから参照し、BAM走査や特徴抽出を繰り返しません。
新規出力へscoreだけ保存し、元の入力・GPU結果を変更しません。既存の計測結果への上書き・自動再開もしません。

2026-11-09のHeLa実測は完了しました。CPU推論26.2762分、過去GPU22.7266分（時間比1.1562倍）、CPU最大RSS0.6455GiBです。
全507,501行のNNクラス一致率は99.9943%で、29部位に境界付近の判定差がありました。
推論以外の過去CPU工程139.1262分を合算するとCPU全体は165.4024分ですが、これは全工程の再実測ではなく推定です。
Excel3シートとTSV、出典/照合記録は`paper/data/cpu_benchmark_20261109/`、全件独立監査は`analysis/runtime/cpu_full_benchmark_audit.json`です。

出力既定値は`results/benchmark_HeLa_CPU_20261109`です。
`run.json`が全体状態、`progress.json`が完了partition/部位数、`inference.log`が実際の公開CLIログ、
`resources.txt`がGNUtimeの実時間・CPU時間・最大RSSです。接続が切れても背景の計測は継続します。
進捗はParquetのfooterが書かれた完了partitionのみ数えるため、処理中の大きな染色体部分はまだ加算されません。
完了時に全GPU/CPU row_id、確率差、argmax一致率を比較して`class_disagreements.tsv`へ保存します。
異なる浮動小数点カーネルなのでbit一致を前提にしません。

GPU基準は既存HeLa v2の推論工程22.7266分です。両方とも起動・モデル/Parquet読込・復元・NN・score保存を含みます。
CPU側で再実行していない他の工程を合算する場合は、全工程の実測とはせず推定と明記します。
別日時の共有ホスト上の計測なので、厳密に隔離された速度比較ではありません。

完了後のExcel/TSV出力：

```bash
.venv310/bin/python analysis/runtime/summarize_inference_benchmark.py
```

未完了の計測を実測結果として出力せず、完了と全スコア照合を確認してから`paper/data/cpu_benchmark_20261109/`へ新規作成します。

## ジョブの資源監視

既存ジョブに変更を加えず、別プロセスで30秒ごとにprocess groupを観測します。
`monitor_jobs.py --job NAME:PGID --job NAME:PGID --output 新しい保存先`
をLinuxホスト上で実行します。対象がすべて終了すると監視も終了します。

`metadata.json`にCPU、搭載メモリ、GPU/driver、監視開始時刻、対象PGIDを保存。
`samples.jsonl`には各PIDのCPU累積時間、RSS、個別VmHWM、スレッド数、IOカウンタ、
GPU状態とGPU使用PID/メモリを記録します。CPU%は100%で1 CPU相当です。
GPU全体の使用率をそのまま対象ジョブの使用率とは見なしません。

途中開始なので、開始前に終了した子プロセスのCPU時間や、過去のジョブ全体の最大RAM/VRAMは復元できません。
監視後も短時間のピークは取り逃がすため「30秒間隔で観測した最大値」と明記します。
RSS合計は共有メモリを重複計上し得ます。IOはファイルシステムのキャッシュ等に影響され、保存ファイル容量とは異なります。

所要時間は既存のapplication logから工程ごとに復元できます。

- 本手法v2: `results/prediction_HeLa_WT_HEKonly_v2/execution.log`、`run.json`。2026-11-07 23:42:23 JST、PGID44603で新規開始。4 worker、最大16group/10 Mbの両鎖統合fetch、整数Parquet v4。監視PID44857、`results/runtime_HeLa_v2_20261107T144307Z`へ23:43:07 JSTから記録（最初の約44秒は欠測）。旧停止分と新規runの時間・観測値を混ぜない。
- 本手法: `results/prediction_HeLa_WT_HEKonly_v1/execution.log`のUTC Starting/Completed、`run.json`、特徴抽出の`progress.json`。
- m6Anet: `results/m6anet_HeLa_WT_v1/logs/*.log`のStarting/Completed。f5c、minimap2、slow5toolsには実時間・CPU時間・peak RAMを出す工程もあります。
- signal工程は破損POD5による停止・再開を含みます。最初から最後までの暦時間と実処理時間を区別し、再開で再利用した変換済みファイルを新規変換時間に混ぜません。
- 既存ログが明示しない待機時間と計算時間を分離できるとは限りません。ファイルmtimeは補助情報です。

論文では入力read数・候補数、ツール/version、CPU/GPU型番、並列数、工程別wall time、取得可能なCPU time、
RAM/VRAMの観測最大値、出力容量をまとめます。監視前欠測は欠測として残します。
両ジョブが同時実行され共有ストレージも使用しているため、厳密な速度比のベンチマークとは区別します。
本手法は既存Dorado/modkit出力から、m6Anetは信号変換から開始しており、比較の起点も明記します。
必要なら後日、同一資源・単独実行で`/usr/bin/time -v`等により再測定します。今回のジョブは再起動しません。
