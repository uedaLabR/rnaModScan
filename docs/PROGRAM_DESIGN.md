# 現行プログラム設計（2026-11-06）

training/に学習用、prediction/に実解析用、utils/に共通NN・BAM特徴・IO・参照処理を置きます。
番号はtest/の実行ファイルだけに付け、工程別の番号フォルダは作りません。

NNは参照3-mer＋binary repeatと、最大300readの18ch特徴を独立に処理する3classモデルです。
層と次元は[NN_DESIGN](NN_DESIGN.md)、保存型は[SITE_BAG_PARQUET](SITE_BAG_PARQUET.md)を参照してください。

- 部位候補はmodkit bedMethylから作り、既知根拠・IVT比較で既存の弱教師を付けます。
- 同一部位のm6A/inosineを統合し、両方正例は除外・記録します。m5C/psiはNNを使わず、既知部位または有効な修飾別モチーフ規則で保持します。既知部位は元の10 read・AF 0.10条件、モチーフ規則だけは20 read・AF 0.40を追加します（修飾readは元の3本以上）。
- 学習/実解析は同じ10kb隣接gapのgroup抽出を使います。BAM fetchは両鎖共通、候補照合は同鎖だけです。
- BAM走査はCIGAR索引を1alignmentにつき1回作り、全部位へ使います。MM復号は必要な時に最大1回。
- CIGARで求めた必要窓のquery位置集合だけにMLを保持し、重複する数値特徴は上限8192件の値cacheで再利用します。
- reservoirには共有リード参照を置き、部位の最終最大300本が確定してからMM/MLと特徴を作ります。
- DeferredReadはalignmentごとのCIGAR・必要窓の座標情報を共有し、最初の確定部位でMM/MLを1回だけ復号します。
- 一時採用後に全てのreservoirから外れたリードはMM/MLや配列・qualityへ変換しません。原depth/AF/call数は全read由来の値です。
- 同じFASTA窓のcaseからrepeatフラグを取得し、4つのNN入力を一つの部位Parquetへ保存します。
- 完了bagの共有リード参照を外し、他の未完了部位も参照しなくなったcontextを解放し、BAM workerは終了時にメモリーを解放します。全BAMのCIGAR cacheは持ちません。
- 学習は索引と保存済み数値を直接読みます。CVは重複窓を同groupにし、細胞/WT/IVTも跨いで漏洩を防ぎます。
- validationでepochを選び、testはその後に評価します。class weightはtrain件数だけから作ります。
- 実解析はprepare/infer/filterを分け、GPU processは準備済み4入力とモデルだけを読みます。
- DB照合はsourceごとの(chrom,start,strand) set、適用sourceは細胞/修飾別にcacheします。
- 詳細TSVはDB/short-read/NNと6種類のモチーフの独立9フラグを保持します。元BEDのPASS行にはDB/short-read/NNの3フラグ、6種類のモチーフ規則をまとめた`motif_keep`（0=なし、1=DRACH、2=代表METTL16、3=NSUN6、4=NSUN2関連G-rich、5=TRUB1、6=PUS7）とOR結果を追記し、コメントヘッダーで説明します。既存列は保持します。
- m5C/ψの追加モチーフゲートは`filtering.auxiliary_motif_min_depth`（既定20）/`auxiliary_motif_min_af`（既定0.40）で設定し、古いroutingでも最終filterで元countを再確認します。既知部位保持にはこの追加ゲートを適用しません。
- rescues/BAM書換えは染色体/strand単位に処理し、最終返却と除外auditを分けます。

全量再学習のモデルはhold-out評価には使いません。既知参照で作った教師に対する内部評価を独立実証と呼びません。
除去した試行コードの範囲は[整理記録](CODE_CLEANUP_20261106.md)にまとめます。
