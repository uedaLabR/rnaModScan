# 候補抽出と共通入力（2026-10-29）

2026-11-12追記：新規学習設定・事前確認・GLORI/Slic-seq/DB等の参照表取り込みは
[カスタムデータ学習ガイド](TRAINING_CUSTOM_DATA.md)にまとめています。
学習用カタログは任意の明示assemblyを扱えます。予測時の参照規則は以下のとおりです。

2026-11-06追記：予測では`genome: hg38/mm10/other`を選択可能。hg38は従来の参照、
mm10はRMBase3.0のみ、otherはDB/short-readを使用しない。以下のGRCh38記載はヒト学習データの仕様。
予測座標は入力BAM/bedMethyl/softmasked FASTAに一致するassemblyのBED0を保持し、assembly変換は行わない。

現行は候補閾値を維持してm6A/Inosineの和集合を作る。[統合A部位・3クラス仕様](ADENINE_THREE_CLASS.md)参照。

2026-11-05追記：候補の閾値・定義は以下を維持し、学習前にm6A/Inosineの部位選択を追加。
m5C/ψは既知部位との一致で返却する。[最新方針](TRAINING_PREDICTION_POLICY.md)参照。

学習の入口は `python training/test/01_sites.py --config training/config.yaml`。
設定はYAMLへ集約し、実装本体は`training/site_lists.py`、出力は`output_dir/sites/`です。
modkit生成済み18列bedMethylを読み、候補にBAMは使いません。
4対象はm6A (`a`)、m5C (`m`)、inosine (`17596`)、pseU/psi (`17802`)。
残る2Omeの4種は学習対象ではなく、BAM特徴として8種すべてを利用します。

共有読込は `utils/io/bed.py`。参照BED6もここで読み、GRCh38の0-based half-open、
単一塩基、明示した鎖を要求します。BEDのchr名は勝手に変換しません。
plain `.bed`、gzip `.gz`/`.bgz` を読み、bedMethylは染色体内の座標順・染色体ブロック連続性・
同一座標/鎖/修飾コードの重複を確認します。8種や未知のコードも読み捨てず `mod_code` に保持し、
4対象以外の `modification` はNoneにします。用途側で対象を選びます。

候補条件は総リード数≥10、対象修飾リード数≥3、AF≥0.1。

```text
total_reads = N_valid_cov + N_fail + N_diff + N_nocall
AF = N_mod / total_reads
N_valid_cov = N_mod + N_canonical + N_other_mod
```

削除リードは分母に含めません。bedMethylのpercent_modified列をそのままAFにはしません。
修飾リード数はmodkitのwinning-state callです。
集約BEDから個別ML閾値を再適用・検証できません。固定ML125を仮定せず、各pileup.logの
実際の塩基別閾値・コマンド・ログSHA256を保存します。

参照一致は同じchrom/start/strand/対象修飾で判定します。細胞対応はreference_cellで明示し、
HEK293をHEK293Tと暗黙に同一視しません。現在の自前サンプルはHEK293Tです。
RMBaseは既存カタログの3.0 GRCh38を使用し、全ヒト由来の根拠として細胞対応の実験手法と区別します。

実解析の一括CLI `python -m rnamodscan predict --config prediction/config.yaml`は、最初に同じBED読込・AF定義で未ラベル候補を作り、`output_dir/candidates/`へ保存します。
実装は`prediction/candidates.py`で、候補作成だけの確認は`prediction/test/01_candidates.py`から実行できます。
BAMはその後の特徴抽出にのみ使い、DBからDRS未観測候補を追加しません。参照一致によるレスキューはNN予測後の工程です。

任意の `python -m rnamodscan quality --config prediction/config.yaml`は、BAMのスコア分布とconfidence閾値候補を`quality/`へ保存します。
学習サンプルの集計は`--config training/config.yaml`を使います。
これは外部modkitを使わない独立集計で、既存bedMethylのwinning-state call数・AFを後から変更するものではありません。
異なるリードcall閾値でAFを作り直す場合はBAMからの再集計が別途必要です。実行順と設定項目は[実行手順](RUN_ORDER.md)を参照してください。
