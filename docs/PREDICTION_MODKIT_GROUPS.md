# 実解析：modkit pileupと候補グループ別BAM抽出

学習データ・ラベル・3クラスNNの設計は変更しない。実解析は次の順に行う。

1. 対象BAMをmodkitでpileupする。既存bedMethylを指定してこの工程を省くこともできる。
2. 総depth・修飾call数・AFで候補を抽出する。
3. m5C/ψは既知部位と一致する候補だけを最終返却する。BAM特徴・NNは使わない。
4. m6A/InosineはA部位の和集合を作り、候補グループごとにBAMを読む。
5. 既存と同じ19塩基・18readチャネル等の特徴を、染色体・strand別Parquetへ保存する。
6. 別プロセスで3クラスNNを実行し、DB/short-read/NN/DRACH/METTL16モチーフの独立フラグORによる最終判定を行う。
7. 必要なら、除外対象のMLを0にしたBAMを出力する。

## AFは低確信度readを分母に含める

```text
AF = N_mod / (N_valid_cov + N_fail + N_diff + N_nocall)
```

`N_mod`はmodkitがML確信度フィルターを適用して数えた対象修飾call。
分母は低確信度の`N_fail`、異なる塩基の`N_diff`、判定不能の`N_nocall`も含む。
削除readは従来どおり除外する。splice skipも塩基の観測として数えない。
depth≥10・対象修飾call≥3・AF≥0.1の既定と、学習時のAFを維持する。
AFは監査・出力用に保持し、NNには入力しない。

modkitのbedMethylの割合は`N_mod/N_valid_cov`で、上式とは異なる。
実解析出力ではこの値を監査専用の`af_pass`として併記するが、候補判定・NN入力には使わない。
`valid_coverage`・`failed_calls`・`different_base_reads`・`no_calls`も保存し、分母を追跡できる。
ゼロ有効depthの`af_pass`と相手修飾の欠測は空欄/nullであり、0ではない。

pileupは個々のreadのML値をbedMethylへ出力する処理ではない。
推定閾値は通常のpileupログ、採用数・不採用数はbedMethyl、NN用の元MLはBAMから取得する。
各readの値を別表へ出すmodkitの`extract`を、NN前の必須工程として追加する必要はない。

## modkitをCLI内から呼ぶ設定

`prediction/config.yaml`の主な設定は次のとおり。

```yaml
modkit:
  run_pileup: true
  executable: modkit  # PATHにない場合は実行ファイルの絶対パス
  threads: 4
  filter_threshold: null
  mod_thresholds: null
  include_bed: null
  seed: null
features:
  candidate_group_gap: 10000
  read_batch_size: 256
```

`run_pileup: true`なら`sample.bam`から`output_dir/pileup/all_modifications.bed`を生成し、
`sample.bedmethyl`・`sample.pileup_log`を指定せず後続処理へ渡せる。
`false`なら既存の両ファイルを指定する（現在の既定）。
`references.fasta`を設定した場合はpileupにも渡す。BAM/FASTAは索引を用意する。
実行ファイルのversion、正確なargv、入力情報、stdout/stderr、pileupログを保存する。
完了済み・途中出力を自動上書きしない。

`filter_threshold: null`はmodkitの自動推定を使う。
固定する場合は例`filter_threshold: {A: 0.8, C: 0.8}`や
`mod_thresholds: {a: 0.9, '17596': 0.9}`を明示する。これらはread confidenceでありsite AF閾値ではない。
`include_bed`は出力だけでなくmodkitの閾値推定対象も狭めるため、既定null。
再フィルター用`modkit modbam adjust-mods --filter-probs`は存在するが、この経路では先に適用しない。
`call-mods`はMLを0/100%へ変えるため、元MLを使うNN特徴抽出の前には使わない。
外部modkitに依存しない既存のquality分布推定コマンドも従来どおり使える。

```bash
python -m rnamodscan predict --config prediction/config.yaml --dry-run
python -m rnamodscan predict --config prediction/config.yaml
```

CPU/GPUを分ける場合は`--stage prepare`→`--stage infer`→`--stage filter`。
pileupだけ確認する場合は`prediction/test/00_pileup.py --config prediction/config.yaml`。
modkitを単独実行後、既存ファイルを指定して通常の予測を行っても同じ後続経路になる。

## 10 kbのグループとメモリ解放

同じ染色体で候補中心を座標順に並べ、隣接中心の距離が**10,000塩基以上**なら別グループにする。
9,999塩基なら同一グループ。同じ座標の両鎖・全NN対象を同じグループへ入れる。
両鎖を含む連続グループをまとめ、最初の候補～最後の候補までをBAM索引で1回fetchし、最大256件ずつ処理する。
`group_batch_size`で最大グループ数、`max_fetch_span`で最初の中心から最後の中心+1までの最大幅を指定する。
10 Mbの既定上限に達する前と染色体境界で分ける。単一グループが10 Mbを超える場合も候補中心で分割する。
区間の間隙も取得するが、鎖の合う候補と重ならないalignmentはCIGAR解析前に除外する。
全適格readを数えつつ、部位ごとのreservoir samplingで最大300本だけを共通Parquetに保持する。

長いalignment・長いintronを持つreadは、複数のグループでfetchされ得る。
各fetch内でCIGAR索引・MM/ML復号を共有し、取得区間の候補だけを処理する。
現在のfetchより後のreservoirやread文脈は作らず、fetch終了時に参照を解放する。
別区間で再fetchされたalignmentは、その区間内で再計算する。
read名で重複除外しないため、同名の別alignmentを失わない。

完了した部位のread bagとBAM読込batchは解放し、書き終えた部位のencoded featureもread文脈にキャッシュしない。
座標変換・MM/ML復号は関連alignmentにつき各fetch内で最大1回。
選択中のreadはMM/ML復号のため元BAMレコードを一時保持し、複数部位で同じ参照を共有する。
10 Mb全域の全readを一度に保持する設計ではない。未完了部位のreservoir、読込batch、Parquet書込bufferはメモリに存在する。
学習と実解析は同じgroup抽出を使う。比較用に従来走査も指定でき、採用readと特徴値は一致させる。

`features/samples/{sample}/candidate_groups.tsv`にグループ座標と候補数を保存する。
`progress.json`はグループ完了数・部位数・alignmentの処理回数を記録する。処理回数はグループ間の再取得分を含む。
グループ完了率を全BAM走査率や総処理時間の割合とみなさない。
`candidate_group_gap: null`なら比較確認用の従来の全BAM走査を選べる。
候補ごとの個別fetchやCIGAR解析には戻さない。

## 任意のグループ並列化

```yaml
features:
  sample_workers: 1
  group_workers: 4
  group_batch_size: 16
  max_fetch_span: 10000000
  bam_threads: 1
  candidate_group_gap: 10000
```

公開CLI `rnamodscan predict --config prediction/config.yaml`で読み込む設定。
学習・再抽出CLIも同じfeatures設定を使う。省略時の既定値はgroup_workers=1、group_batch_size=32、max_fetch_span=10000000。
配布YAMLではgroup_batch_size=16、HeLa v2ではgroup_workers=4を指定する。
sample並列とgroup並列の同時指定は拒否し、過剰な入れ子プロセスを作らない。

各spawnワーカーが自分のBAM/FASTAを開き、read情報は完了部位ごとに解放する。
連続するグループを上記の数・幅上限内で1タスク/1 fetchにまとめ、ファイル開閉・seek・小ファイル数を抑える。
参照カタログはワーカー内で再利用し、BAM/read/CIGARは他ワーカーと共有しない。
結果のテンソルはIPCで返さず、`sites/batch-XXXXXX/sites/chrom=.../strand=.../`の独立Parquetへ保存する。
完了順が前後しても、元のタスク番号順にmanifestとsummaryを結合し、座標・row_id順の共通indexを作る。
重み・NN・抽出ラベル・一様サンプリングは変更しない。失敗時は親出力をINCOMPLETE/FAILEDのままにする。

Numbaは採用しない。MM/ML解析は引き続きpysam/HTSlibを使う。
新Parquet v4は、float32のread_featuresの重複保存を廃し、元の整数に基づくread_inputsを保存する。
MLはuint8（0–255の1byte）、品質codeはuint16（Q+1、欠測0）、挿入長はuint32、塩基区分はuint8、
前後splice距離はuint16（clip1000、未観測1000）とする。監査用raw_readsのnull/0や元距離は保持する。
学習・推論ではArrow整数arrayからNumPyの参照表で18ch float32を復元し、元の入力と完全一致させる。
既存v3も読めるため、保存方式変更による再学習は不要。小数2桁への丸めやfloat16化は行わない。
ParquetはすでにZstd圧縮・dictionary encodingを使っているため、実ファイルの削減率は測定して報告する。
測定は`analysis/runtime/benchmark_group_extraction.py`、小区間の結果は`results/model_checks/group_acceleration_20261107/`を参照。
全データでの速度比は未測定。旧HeLaジョブはユーザー指示で停止し、旧出力は保存する。

## 最終出力と確認

元bedMethylのPASS行は`predictions/bedmethyl/filtered.bed`、修飾別は同階層の`{target}/filtered.bed`。元列に`flag_db, flag_short_read, flag_nn, motif_keep, filter_pass`を追記する。`motif_keep`は0=モチーフ保持なし、1=DRACH保持、2=代表METTL16モチーフ保持とし、各BEDのコメントヘッダーで説明する。詳細TSVでは`flag_drach`/`flag_mettl16`の二つのbinaryフラグを独立に保存する。
`filtering.m6a_mettl16_keep`が有効なら、対象Aが4番目のRNA鎖方向`UAC[m6A]GAGAA`に一致する元m6A候補を、NN判定と独立に保持する。拡張モチーフや新規候補行は加えない。
修飾別`predictions/{target}/kept_sites.tsv`が返却部位の詳細表。
m5C/ψはdepth・AFを通過し、既知部位に一致する行だけを含む。
`all_predictions.tsv`は除外理由も含む監査表として残し、BAMの除外mapにも使う。
m6A/Inosineの共通NN・独立フラグORの詳細は[3クラス設計](ADENINE_THREE_CLASS.md)を参照。

小規模テストで、従来走査との全特徴値の一致、10 kb境界、長いspliceを跨ぐ再計算、
fetch間のread/CIGAR参照解放、同名read、空部位、Parquet offset、両鎖の統合fetch、幅上限による分割、4並列での値・順序の一致を確認した。
modkitラッパーは模擬実行ファイルでCLI接続を検証した。このfetch変更後の全件速度・HeLa精度評価は未測定。実行状態と既存CV結果はHANDOFFと各runのprogress.jsonを参照。

公式資料：
[pileupとbedMethyl](https://nanoporetech.github.io/modkit/intro_pileup.html)、
[confidenceフィルター](https://nanoporetech.github.io/modkit/filtering.html)、
[詳細CLI](https://nanoporetech.github.io/modkit/advanced_usage.html)、
[call-mods](https://nanoporetech.github.io/modkit/intro_call_mods.html)、
[include-bed](https://nanoporetech.github.io/modkit/intro_include_bed.html)。
