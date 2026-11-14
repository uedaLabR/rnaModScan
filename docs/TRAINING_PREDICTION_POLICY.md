# 学習・返却方針（2026-11-06）

m6A/inosineを共通A部位・3クラスNNへ統合しています。現行出力はtraining_v10、モデルはadenineです。

| 対象 | 学習/判定 | 返却 |
| --- | --- | --- |
| m6A DRACH内外、inosine | 同じ3クラスNN（m6A/inosine/FP） | 元候補についてDB・short-read・NN argmax一致・DRACH・METTL16モチーフ（後二者はm6Aのみ）の独立フラグをORで返却 |
| m5C/psi | NN学習・推論をしない | 既知部位は元の10 read・AF 0.10条件、NSUN6/NSUN2（m5C）・TRUB1/PUS7（ψ）モチーフ保持だけは20 read・AF 0.40を追加して返却 |

共通候補の既定はbedMethylのdepth>=10、対象修飾read>=3、AF>=0.1です。m5C/ψの既知部位保持もこの元条件を使います。モチーフ保持（コード3–6）だけにdepth>=20・AF>=0.40を追加し、`filtering.auxiliary_motif_min_depth`/`auxiliary_motif_min_af`の省略時も20/0.40を使います。DBから未観測部位や別修飾の候補行を追加しません。両修飾の元候補がそれぞれPASSする場合は両方返します。
flag_db、flag_short_readはWTだけに適用し、IVTは既知参照でrescueしません。元候補の6種類のモチーフ保持規則はIVTにも適用します。
METTL16規則はRNA方向の`UAC[m6A]GAGAA`（DNA `TACAGAGAA`、対象Aは4番目）に一致する元m6A候補だけを保持します。拡張`UAC[m6A]GARAA`や未観測部位・別修飾を追加しません。
`filtering.m6a_mettl16_keep`はpublic/Docker設定でtrue、省略された既存YAMLではfalseです。m6A/inosineと既知部位保持の元depth/read/AF閾値、NN学習・予測値は変更しません。
m5Cの`m5c_nsun6_keep`/`m5c_nsun2_keep`、ψの`pseu_trub1_keep`/`pseu_pus7_keep`はpublic/Docker設定でtrue、省略時はfalseです。元候補の鎖方向配列だけを検査し、二次構造や酵素依存性を検証しません。既知部位保持は引き続きWTだけです。RNAパターンと候補位置は[出力フラグ一覧](OUTPUT_FLAGS.md#motif-codes)を参照してください。
AFの分母はconfidence failureを含み、af_passは監査専用です。

WT正例は細胞対応short-read/RMBase根拠による弱教師です。
同座標/鎖/修飾のIVT N_mod>=3かつAF_IVT>=0.6*AF_WTなら負例を優先します。
欠測/非overlapを陰性にしません。両修飾正例は除外し、WTで片方陰性・他方不明もjointFPにしません。
全正例と対応IVTを保持し、その他FPは細胞ごとの正例合計と同数をseed42で抽出します。

現在の選択数はm6A194,592・inosine7,036・FP347,107、計548,735です。
FPは対応IVT145,479と追加無作為201,628です。これは独立実証された真の部位数ではありません。
旧全候補・論文集計は保持し、検証済みtraining_v5/selectionを新しいBAM抽出の元に使います。

1部位最大300readをreservoirで保存し、原depth/AF/countは全read由来の値を保持します。
repeatはsoftmasked genome FASTAの小文字acgtから19boolを同時取得し、同じ部位Parquetに保存します。
細かいclassは使わず、3-merの順序付き3binaryをゼロ初期化の補助残差として配列枝へ渡します。
割合・別mask表・NPY・旧全readNN経路は使いません。

CVは位置窓groupによる内部評価で、参照で作った弱教師への成績です。
m6Anet比較、repeat有無ablation、ヒトから他生物種への性能は未実施です。
repeat入力後のAlu整合性だけを独立したNN妥当性の証拠としません。

実行は[training](../training/README.md)、[prediction](../prediction/README.md)、
ラベルの詳細は[統合A部位](ADENINE_THREE_CLASS.md)を参照してください。

元bedMethylのPASS行だけを`predictions/bedmethyl/filtered.bed`と修飾別BEDへ保存します。元列に`flag_db, flag_short_read, flag_nn, motif_keep, filter_pass`の順で5列を追加します。`motif_keep`は0=モチーフ保持なし、1=DRACH、2=代表METTL16、3=NSUN6、4=NSUN2関連G-rich、5=TRUB1、6=PUS7です。各BEDのコメントヘッダーに列とコードの説明を記録します。元AF/countは変更しません。TSVではDB/short-read/NNと6種類のモチーフの独立9フラグを保持し、全候補の除外監査も残します。m5C/ψで追加のモチーフ閾値を満たさない行は`auxiliary_motif_eligible=0`となり、モチーフフラグは0ですが、既知部位フラグとPASSは1になりえます。
