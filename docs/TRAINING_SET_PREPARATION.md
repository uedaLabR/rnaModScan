# 学習部位一覧 v2（2026-10-29）

現行は元ラベルを保存したまま[統合A部位・3クラス](ADENINE_THREE_CLASS.md)へ変換。
両方正例は除外。片方だけ負例で他方不明のWTをFP教師にしない。

2026-11-05追記：以下の全候補ラベルを保存したまま、NN学習はm6A/Inosineへ限定。
正例・対応IVTを全件保持し、その他の負例を各cell/targetの正例と同数選ぶ。
選択後に特徴抽出・位置group分割を行う。[最新方針・実測件数](TRAINING_PREDICTION_POLICY.md)参照。

最新のユーザー指定に基づき、旧v1のDRACH単独正例とIVT AF>WT AFの規則を変更した仕様です。
入口は `python training/test/01_sites.py --config training/config.yaml`、実装は`training/site_lists.py`、出力は`output_dir/sites/`です。
実行方法と設定は[実行手順](RUN_ORDER.md)、学習用の配置は[training/README](../training/README.md)を参照してください。

## 対象とラベル

HEK293T/HeLaのWT・IVT、判定対象4修飾を使います。NN入力には8修飾すべてを使用します。
各WT候補は、次の優先順で判定します。

1. IVTサンプル自体が未完成/未設定なら、ラベル空欄、`label_status=pending`。
2. 同座標・鎖・対象修飾のIVTで **N_mod≥3 かつ IVT AF≥0.6×WT AF** なら負例。
   外部参照との一致があってもこの負例を優先します。等号を含みます。
   3本は総リード数ではなく、対象修飾の修飾リード数です。
   IVT総depthの候補閾値はこの上書き条件に追加しません。
3. 細胞対応の既存実験手法、またはRMBase 3.0との一致があれば正例。
   短鎖実験手法のIDとDBのIDを別列に保存します。
4. それ以外は未ラベル。非overlapを負例にしません。DRACH単独では正例にしません。

IVT比較は候補抽出前のbedMethylを使います。候補閾値未満でも対応WT部位の行を保持します。
IVTが取得済みでも、その部位が未記録/ゼロdepthなら不明として扱い、AF=0を補完しません。
その欠測だけでは外部根拠のあるWT正例を取り消しません。
一方、IVTサンプル自体がまだ未取得の段階は、後日の負例上書きに備えてWTを判定保留にします。
`positive_evidence` で保留中も正例候補を確認できます。

IVTでは通常の候補条件を満たす部位を負例とします。WT正例の対応IVTは総depth≥10なら、
修飾リード/AFが低くても負例に含める従来の規則を保持します。
IVTはcDNAから作るため、inosine由来の編集が鋳型のGとして残り得ます。
これらは運用上の弱い教師ラベルであり、部位ごとの独立実証やリードごとの真の修飾状態を保証しません。

## 出力と評価

`all_sites.tsv` は1サンプル×座標×鎖×対象修飾につき1行。
元WT/IVTのAF・修飾数・総depth、参照ソース、ラベル理由、判定状態を保持します。
部位IDは `sample|chrom:start:strand:modification`。同一入力で安定し、後日ラベル更新時もBAM特徴と結合できます。
IVTの比較列はWT行にのみ入り、IVT行の空欄はnot_applicableです。

±9塩基窓（計19塩基）が重なる部位を鎖・対象修飾・細胞・WT/IVTをまたいで同じgroupにします。
重複の連鎖も同じgroupです。BAM特徴の窓を広げる場合は先に一覧のgroupを再生成します。
部位一覧の段階ではfoldを割り当てません。全正例/負例のBAM抽出後、軽い位置・サンプル索引から
全対象のgroupをまとめて5-fold/seed42を割り当てます。
外側のtestは各回1/5、非testグループの10%をvalidationにし、目安はtrain72%/validation8%/test20%です。
同じ部位のHEK293T/HeLa、WT/IVTは別の例としてすべて残して同じsplitに入れます。
割当は位置・観測数・seedだけで決め、ラベル/ML/AFには依存しません。詳細はtraining/README.md。
遺伝子単位分割・独立外部試験を実施したという意味ではありません。

分割IDは`splits/{target}/fold_i/{train,validation,test}/sites.tsv`に分けます。
学習成果は`models/{target}/fold_i/train/`、validationの予測/指標は同foldの`validation/`、testは`test/`です。
testはvalidationによるbest epoch確定後に評価し、最終モデルの学習回数は各foldの`train/history.csv`から決めます。
最終モデルは`final_model/{target}/`へ保存します。

論文用集計はラベル理由、参照一致の排他的区分、ソース別一致、IVT比較の取得状態、depth/AF分布を保存します。
正例作成に使ったRMBase/実験手法との一致を独立評価として報告しません。
実データの学習・性能評価は未実行です。CV実行時は内部評価を保存し、最後に全ラベル部位で実解析用モデルを再学習します。
最終モデルの再学習データにはhold-outがないため、そこから評価性能を報告しません。詳細はtraining/README.md。

任意のquality工程による推奨閾値は、現在の部位一覧・AF・教師ラベルに自動適用しません。
リードの修飾confidence、部位AF、NN保持スコアの各閾値は別に管理します。

## 2026-11-02 PUS7追加正例

ユーザー指定によりCell GenomicsのTable S1（mmc2.xlsx）のPUS7依存Ψを正例根拠へ追加。
S1A 61部位、S1B 43部位、和集合84部位。元細胞はHepG2のまま、
HEK293T/HeLaへの適用をcatalogで明示する。short_read根拠欄で既存手法と同様に扱うが、
実際はNanopore direct RNA（S1BはDorado）であり、細胞対応short-readとは呼ばない。
同一部位を複数sourceで数えても学習例は増殖しない。IVT負例優先・欠測規則は維持。
共有catalogを使う予測レスキューにも適用される。独立検証セットとしては使用しない。
原本・監査・出典はdata/reference_sites_GRCh38/pus7/。既存結果は自動更新しない。
