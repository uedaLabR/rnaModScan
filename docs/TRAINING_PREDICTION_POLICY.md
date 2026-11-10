# 学習・返却方針（2026-11-06）

m6A/inosineを共通A部位・3クラスNNへ統合しています。現行出力はtraining_v10、モデルはadenineです。

| 対象 | 学習/判定 | 返却 |
| --- | --- | --- |
| m6A DRACH内外、inosine | 同じ3クラスNN（m6A/inosine/FP） | 元候補についてDB・short-read・NN argmax一致・DRACH（m6Aのみ）の独立フラグをORで返却 |
| m5C/psi | NN学習・推論をしない | depth/AF条件を満たすWTの既知候補だけ |

共通候補はbedMethylのdepth>=10、対象修飾read>=3、AF>=0.1です。DBから未観測部位や別修飾の候補行を追加しません。両修飾の元候補がそれぞれPASSする場合は両方返します。
flag_db、flag_short_readはWTだけに適用し、IVTは既知参照でrescueしません。元DRACH m6Aの保持はIVTにも適用します。
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

元bedMethylのPASS行だけを`predictions/bedmethyl/filtered.bed`と修飾別BEDへ保存します。元列に`flag_db, flag_short_read, flag_nn, flag_drach, filter_pass`の順で5列を追加します。元AF/countは変更しません。TSVは全候補の除外監査も保持します。
