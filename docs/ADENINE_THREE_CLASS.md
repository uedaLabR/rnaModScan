# A部位の統合データセットと3クラスNN（2026-11-05）


参照入力はFASTAからの19bool repeat情報も含み、3-merの順序付き3binaryをゼロ初期化の補助残差として配列枝へ渡します。割合は使いません。BAM抽出中に配列/read/repeatを同じ部位Parquet v3へ保存し、別準備は不要です。

学習設計は維持し、実解析に[任意modkit pileupと10 kbグループ別BAM抽出](PREDICTION_MODKIT_GROUPS.md)を追加した。
AFはconfidence failureも含む総read分母のまま。modkit標準比率af_passは監査専用。

## 判定単位とクラス

m6A・Inosine候補の和集合を、サンプル・GRCh38座標・鎖ごとに1件へまとめる。
WT/IVTとHEK293T/HeLaはそれぞれ別の観測として保持する。
同じ観測が両修飾の候補でも、BAMのread bagとNN入力は1件だけ。
内部モデル名と保存先は`adenine`で、生物学的な修飾名を追加する意味ではない。

クラスの順序は固定し、ラベルは0=m6A、1=Inosine、2=false_positive。
出力は3ノードのsoftmax、合計は1。これは候補に対する弱教師ラベルで学習したスコアで、
各RNA分子の修飾確率や、実解析集団で校正済みの真の確率を保証しない。
同一座標の全RNA分子で両修飾が生物学的に排他的だと示すものでもなく、この設計で返すクラスを1つ選ぶ。

## 教師ラベルと部位選択

既存の4対象の候補・ラベルを保存したまま、m6A/Inosineの`all_sites.tsv`を座標で結合する。
未ラベルの候補も結合時に残し、相手側AFの有無が教師ラベルの有無を示すことを避ける。

| 同じ観測の根拠 | 統合ラベル |
| --- | --- |
| m6Aだけ正例 | m6A |
| Inosineだけ正例 | Inosine |
| 両方が正例 | 学習から除外し`excluded_sites.tsv`へ記録 |
| IVT | false_positive（既存の運用上の負例定義） |
| WTで両対象とも明示的な負例 | false_positive |
| WTで片方が負例、もう片方が不明・未ラベル | 除外。欠測を陰性にしない |
| その他の未ラベルWT | 除外 |

元のIVT AF上書き規則は維持する。正例は既存手法/RMBaseとIVT比較で付けた弱いラベルである。
現在の正例はm6A 194,592件、Inosine 7,036件。同一サンプル・座標・鎖で両方正例は0件だった。
細胞が異なる同座標での違いは矛盾として除外せず、別観測として同じCV groupに入れる。

各細胞について、全正例・対応IVTを保持し、残りの適格false_positiveから正例合計と同数を
seed=42で無作為に追加する。統合後に抽出するため、同じFPを二重に選ばない。
選ばれた部位では最大300本を非復元無作為抽出し、その原特徴とNN数値を保存する。全適格read数とbedMethylの元AF/call数は別に保持する。m5C/ψは学習しない。

今回作成した統合一覧は、m6A 194,592件・Inosine 7,036件・FP 347,107件、合計548,735件。
FPは対応IVT 145,479件＋追加無作為抽出201,628件。根拠不足で学習対象外のWTは1,110,729件。
元の全候補一覧・論文用集計は保持している。

## NN入力と学習

NN入力はsequence_tokens・sequence_repeat_mask・read_features・read_maskの4つです。
19塩基窓の3-mer配列Encoderと、最大300本の18チャネルを処理する共有Conv1D1層＋read内Transformer1blockを独立した2枝とし、
配列64次元とmasked attentionで集約したread64次元を最後に結合します。
read枝にはMATCH/A/C/G/T/Nの6one-hotを含め、配列Encoderの特徴をCNNへ直接渡しません。

repeat class・GENCODE領域・depth/AFの部位統計はNNへ入力しません。Parquetと注釈表は保持し、
元AF、gene/region、repeat等は出力・監査用metadataとして残します。座標・参照一致・細胞・WT/IVTもNN入力外です。
マイナス鎖では配列を逆相補にし、全特徴をRNAの5′→3′へ揃えます。

Dense64→Dropout0.2→Dense32→3softmax、損失はSparseCategoricalCrossentropy。
train内の件数だけからN/(3×各class件数)のclass weightを計算し、全classがないtrainは停止します。
5-foldは重なる位置窓をgroup化し、同部位の全細胞・WT/IVT・鎖を同じsplitへ入れます。
validation/testは混同行列、各class precision/recall/F1、macro-F1、accuracyとDRACH別の値を保存。
これらは弱ラベルに対する内部評価です。最終モデルは全選択部位で再学習し、性能はCV testから報告します。

## 実解析での最終クラス

2026-11-06の最新指定で、旧「一つの最終修飾を選ぶ」方式を独立フラグのORに変更。

- `flag_db`: WT候補とDBの同座標・鎖・対象修飾一致。
- `flag_short_read`: WT候補と対応細胞の参照一致。
- `flag_nn`: 3クラスargmaxが元候補の修飾と一致。
- `flag_drach`: 保持設定が有効な元m6A DRACH候補。Inosineでは常に0。
- `flag_mettl16`: 保持設定が有効な元m6A候補で、RNA鎖方向の`UAC[m6A]GAGAA`に一致。Inosineでは常に0。

どれかが1なら`filter_pass=1`。m5C/ψのNN/DRACH/METTL16フラグは0。
IVTはDB/short-readフラグ0だが、元DRACH/METTL16モチーフ保持は同じ規則で適用する。
`filtering.m6a_mettl16_keep`はpublic/Docker設定で有効、既存YAMLで省略時はfalse。
対象Aは9塩基の4番目（DNA表現`TACAGAGAA`）で、拡張モチーフ`UAC[m6A]GARAA`は含めない。
元候補のdepth/AF/read条件を維持し、NNで除外された対象も追加規則で保持する。モチーフ一致だけではMETTL16依存性を証明しない。
NNは3クラスのままで、閾値とargmaxを混同しない。
同じ位置の両修飾がPASSしてもよい。元のdepth/AF条件を通過しない別修飾の行を生成しない。
`final_class`は部位auditでは返却修飾のセミコロン列、修飾別auditでは保持時の当該修飾（除外時は空欄）。
BAMは実際の候補について`final_keep=0`が確定した対象MLだけを0にする。

## 入出力と実行

既定のYAMLは`nn.task: adenine_three_class`。
`training/config.yaml`は`results/training_v10/`、`prediction/config.yaml`は`results/prediction_v8/`。
モデルは`final_model/adenine/`の1個。旧2値モデルの経路は削除し、現行は3クラスだけを使う。

```bash
python training/test/01_select_sites.py --config training/config.yaml
python training/test/00_prepare.py --config training/config.yaml
python training/test/05_train_and_finalize.py --config training/config.yaml
```

選択は`selection/adenine/training_sites.tsv`、件数は`selection/summary.tsv`、
除外理由と元IDは`selection/excluded_sites.tsv`、出典・class順序・seedは`selection/run.json`。
`features/index/adenine/`→`splits/adenine/fold_i/{train,validation,test}/`→
`models/adenine/fold_i/`→`final_model/adenine/`の順。
HEK293T単独学習用manifestも、同じ統合索引から`04_cell_datasets.py`で作る。

CPU準備→別プロセスGPU推論→CPU後処理という分離と、染色体・strand別Parquetを維持する。

```bash
python -m rnamodscan predict --config prediction/config.yaml --stage prepare
python -m rnamodscan predict --config prediction/config.yaml --stage infer
python -m rnamodscan predict --config prediction/config.yaml --stage filter --recalibrate
```

一括実行は従来どおり`python -m rnamodscan predict --config prediction/config.yaml`。
prepared cacheは`prediction_inputs_v9`、スコアは`scores/adenine/`に1部位1行。
`predictions/adenine/all_classifications.tsv`はNNの3スコア・NNクラス・既知の有無・最終クラス・理由を保存。
`class_counts.tsv`は同じ1部位1件の単位でNNクラス→最終クラスと理由を集計する。
従来の`predictions/{m6A,inosine}/all_predictions.tsv`にも、同じ3スコアと最終判定を保存する。
両ファイルは和集合の各位置を含み、元の対象候補だったかは`candidate_m6A/inosine`で区別する。
そのため修飾別の全予測表の件数と、元のDorado修飾別候補数は同じ分母ではない。
各対象の元AF・call数・出典は別列で保持し、欠測を0に補完しない。

BAM出力は最終クラスで除外となった対象のMLだけを0にする。NNが別の修飾を選んでも、
その修飾のread MLを新しく作成したり増やしたりしない。DoradoのAF/call数も書き換えない。
NNクラスと最終返却クラスが違う場合も記録される。

旧モデル・旧prepared cacheは新しい独立2枝へ流用しない。現在のjoint grouped CVは再利用できます。
旧2値dataset用CVは統合モデルには使いません。raw特徴Parquet v3は、
完了済みデータなら同座標・鎖・サンプルの片方のbagへ索引を張って再利用できる。
`feature_row_id`で元のread bag IDを保持する。中断したHeLa IVTは未完了のため再利用不可。
現在停止中の実データBAM抽出を自動再開しない。
