# 配列・リード特徴による部位判定NN（2026-11-06）

m6A・inosine・false_positiveを判定する3クラスsoftmaxです。参照配列とread特徴は独立した2枝で処理し、最後に結合します。repeat class・GENCODE領域・depth・AF・参照支持はNN入力に使わず、Parquetの報告用メタデータとして保持します。

## 入出力

| 入力 | 学習batchの形 | 内容 |
| --- | --- | --- |
| `sequence_tokens` | B × 17 int32 | RNA方向の19塩基参照配列から重複3-merを符号化 |
| `sequence_repeat_mask` | B × 19 bool | 同じRNA方向19塩基のbinary repeat情報（softmasked FASTA） |
| `read_features` | B × 300 × 19 × 18 float32 | 下表の18チャネル |
| `read_mask` | B × 300 bool | 実在するreadだけtrue |

Bの既定は16部位です。300本超はseed固定のreservoir samplingで最大300本を非復元無作為抽出し、300本未満は実readだけParquetに保存します。0埋めはbatch作成時のRAM上だけで行います。NN出力はB × 3で、順序はm6A・inosine・false_positiveです。

## 18 readチャネル

| 番号 | 特徴 | 変換 |
| --- | --- | --- |
| 1–8 | m6A、inosine、m5C、pseU、2OmeA/C/G/UのML | ML/255、欠測は0 |
| 9 | base quality | 1/(Q+1)、欠測は0 |
| 10 | その位置の直後にある挿入長 | log1p(長さ) |
| 11–16 | MATCH・A・C・G・T・N | 6項目のone-hot |
| 17–18 | 同じreadで検出したRNA 5′側・3′側splice接合部までのquery距離 | log1p(min(d,1000))/log1p(1000)、未観測はd=1000 |

通常塩基が参照ACGTと一致するときはMATCHだけを1にします。mismatchでは読まれたA/C/G/Tを、readがNなら参照塩基にかかわらずNを1にします。参照がNで一致を判断できない場合は、読まれた塩基を保持します。塩基がない位置（deletion・splice skip・未被覆）は6項目とも0です。独立したmismatchフラグは入力しません。Nと欠失・splice skipを同一視しません。

quality観測フラグ、alignment状態、MAPQ、splice観測フラグ、修飾の観測・適用・宣言フラグもNN入力から外します。これらの原値・欠測情報は`raw_reads`に残します。未観測spliceと1000塩基以上離れたspliceはNN上では同じ値1になります。これは簡略化に伴う意図的な情報集約です。

負鎖では参照窓とread塩基を逆相補、ML・品質等を逆順にし、全チャネルをRNA 5′→3′へそろえます。MATCHもこの方向にそろえた参照との比較で決めます。座標はGRCh38 BED0のままです。

## 独立した2枝と融合

| 枝 | 層と出力形（1部位または1read） |
| --- | --- |
| 参照配列 | 17 trimer → token/位置Embedding16＋repeat補助入力 → Transformer 1block（2head、FF32→16）→ center16＋mean16 → Dense64 |
| read | 19×18 → Conv1D16（kernel3、valid）→ 17×16 → 位置Embedding → Transformer 1block（2head、key_dim8、FF32→16）→ center16＋mean16 → Dense64 |
| 部位内read集約 | 実readの64次元 → Dense32 tanh → scalar attention → masked weighted mean64 |
| 最終融合 | 配列64＋read64＝128 → Dense64 → Dropout0.2 → Dense32 → Softmax3 |

read側の畳み込みは1層です。直後のTransformerは**1read内の17位置**を処理します。300read間にself-attentionをかける構造ではありません。Convの受容野は3塩基、Transformerにより19塩基窓全体の関係を扱います。参照配列Embeddingをread枝へ渡す接続はありません。

B部位の実readだけを`read_mask`でgatherし、共有read encoderを1回呼びます。部位へscatterで戻し、paddingを除いたread attentionで集約します。readの小分けループはありません。paddingはConv/Transformerへ入らず、attentionでも重み0になります。すべて0の特徴を持つ実readもmask=trueです。

既定の3クラスモデルは**23,731 trainable parameters**です。read encoderは5,488パラメータです。前の32次元read枝の33,251パラメータから縮小し、参照枝と融合部は維持しました。構造は小規模ですが、read数に比例する計算は残ります。実データの精度・実行速度はfold0で評価します。人工入力の計時を実データ全工程の速度と同一視しません。

## 3-merとbinary repeat情報

`references.fasta`に明示した参照の小文字acgtを1、大文字ACGTを0とします。N/nと染色体端paddingは0。マイナス鎖では19個のmaskを逆順にそろえます。FASTAを大文字化する前に取得し、配列を大文字化した結果が保存済み参照窓と一致することも確認します。細かいrepeat classは使いません。

19フラグを、配列と同じ重複3塩基で並べ、17×3のbinary値へ変換します。例えば`aAC`は配列トークン`AAC`とrepeat `[1,0,0]`です。平均・割合・多数決には変換しません。順序のある3値をbias無しDense16で射影し、token＋位置Embedding16へ残差として加えます。射影重みはすべて0で初期化し、学習開始時のrepeat寄与を0にします。

3-mer語彙は65のまま、repeatの状態ごとに別の配列トークンを作りません。repeatをattention maskやpadding maskにせず、全17トークン間のattentionを維持します。追加は48パラメータです。精度の非悪化を保証するものではなく、repeat有無のablationは今後validationで確認する計画です。

repeat flagは二本鎖RNA構造の推定そのものではありません。入力形式には種名や細かな分類を含めず、他生物種でもその種のsoftmasked FASTAから同じ形式を作れます。ヒトから他種への予測精度は未検証です。

## 学習・検証・互換性

教師ラベル・部位選択・位置group CV・class weightは[統合A部位設計](ADENINE_THREE_CLASS.md)と同じです。Adam/学習率0.001、最大20epoch、validation lossのpatience3が既定です。最終モデルは各foldの最良validation epochの中央値（切上げ）の回数で全選択データを再学習します。

モデルは`feature_set=sequence_reads_repeat18_v1`、共通部位Parquetは`site_bags_parquet_v3`、索引は`site_bag_index_v3`、推論snapshotは`prediction_inputs_v9`です。28チャネルの旧モデル・部位Parquet v1・prepared入力は混用しません。旧raw read Parquet v3は原値の資料として保持できますが、今回の既定はBAMから新しい保存先`results/training_v8`へ再抽出します。

入口`training/test/00_rebuild_site_parquet.py`はfeatures→CV→HEK単独manifestを作り、NN学習は開始しません。参照FASTA窓を読む際に19bool repeat maskも取得し、sequence_tokens/read_featuresと同じ部位Parquetに保存します。追加NPY・mask sidecar・fit中のFASTA再読込・read再encodeはありません。推論時の学習クラス・教師フラグはすべてnullです。[共通形式](SITE_BAG_PARQUET.md)と[実行順](RUN_ORDER.md)も参照してください。

保存復元、独立した2枝、実read数1/10/299/300本のpadding前後の予測と勾配、paddingへの勾配0、read順序不変性を人工データで検証します。validation/testには元depth別・3クラス別の指標を保存します。これらの弱い教師による内部評価を独立した生物学的検証として扱いません。

## 軽量化と入力互換性（2026-11-06）

read_hidden_dim=16、read_attention_heads=2、read_transformer_ff=32をModelConfigと保存モデルへ記録します。参照Transformerは従来の16次元/2heads/FF32、read集約は64次元、融合は128次元、出力3classのままです。新しい重みは再学習します。4入力の型・18チャネル・repeat定義・raw Parquet v3・索引v3・prepared v9は変わりません。現在の抽出を継続し、そのParquetを軽量モデルの学習に使います。以前の約0.0454秒/batchは33,251パラメータモデルの人工入力計測です。軽量モデルの実データ速度・精度の測定値ではありません。
