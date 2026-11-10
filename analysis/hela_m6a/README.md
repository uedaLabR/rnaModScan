# HeLa m6A：GLORI・m6Anet・rnaModScanの比較

完了済みのHeLa WT解析結果を使い、ゲノムDRACH内は3集合、non-DRACHはGLORIと本法の2集合で比較します。
BAM抽出・NN学習・m6Anet推論は再実行しません。設定と結果は以前の版を上書きしません。

```bash
cd /home/ueda/PycharmProjects/rnaModScan
.venv310/bin/python analysis/hela_m6a/compare.py --config analysis/hela_m6a/config.yaml
.venv310/bin/python analysis/hela_m6a/write_report.py paper/data/hela_m6a_comparison_20261108
```

`config.yaml`に入力・新規出力先・図の保存先・m6Anet部位確率閾値を指定します。
本文用はユーザー指定の部位確率0.8以上とし、[m6Anet公式手順](https://m6anet.readthedocs.io/en/latest/quickstart.html)で推奨される0.9以上をサプリに併記します。
0.8は今回の比較用設定で、m6Anetの公式既定値とは呼びません。0.9の候補数を確認した後に指定されたことを記録します。
同じ出力先への再実行は拒否するので、再解析時は新しい出力先を指定してください。

## 座標と集合

- GENCODE v44 GFF3のexonから、m6Anetの転写産物位置をGRCh38のBED0座標へ変換します。
  m6Anet2.1.0では出力位置は0-basedの5-mer中央（eventalign位置+2）です。
  全行の5-merを元の転写産物FASTAに照合し、負鎖もRNAの5′→3′向きで変換します。
- 同じ染色体・座標・鎖のisoformは1部位にまとめます。確率は最大値、陽性は少なくとも1isoformが閾値以上の場合です。
  isoform間で共有されるread数は合算せず、最大depthとisoform数を監査用に残します。
- 全手法を同じゲノムFASTAの5塩基でDRACH/non-DRACH/unknownに分けます。
  元Dorado出力のnon-DRACHには中央が参照Aでない位置も含まれるので、`reference_A`を別列・集計で記録します。
  exon境界の転写産物DRACHとゲノムDRACHが異なる例も、座標変換表に残します。
- GLORIはHeLaの既存2replicate由来の登録済み和集合BEDをそのまま使用します。RMBaseと混ぜません。

## 図と分母

- `Fig_HeLa_m6A_Venn_NN_only`：本文用。既知部位・DRACHレスキュー前のm6A softmax argmaxだけの比較。AはDRACH内3集合、Bはnon-DRACH2集合。
- `FigS_HeLa_m6A_Venn_Dorado`：元Dorado/modkit候補の比較。
- `FigS_HeLa_m6A_Venn_NN_covered`：NN判定の副解析。Aは実際にm6Anetがスコアを出したDRACH座標に全3集合を限定。
  BはGLORIをDRS総depth≥10で被覆された部位に限定し、本法の同じdepth条件を通過した候補と比較します。
  AとBで分母の定義が違うことを図中に明示します。
- `FigS_HeLa_m6A_Venn_NN_p0p90`：m6Anetのみ部位確率0.9以上にした感度解析。本法のNN判定は変更しません。

各図はSVG・PDF・PNGで保存します。円面積は部位数に比例しない模式ベン図で、各排他的領域の数値は厳密です。
集合・交差数・割合・coverage分母・閾値感度解析・座標変換QCはTSVと累積Excelに保存します。
以前の52シートの値を読み戻して照合し、今回のシートを追加します。
`membership_*.tsv`にはベン図に使う各部位の座標・所属・フラグ・スコア・AF/depthを残します。
検証済みの座標変換がある場合は、`--mapped-sites 完了済みの比較ディレクトリ`で入力同一性を確認して再利用できます。
確率と閾値判定・集計・図は改めて作り、元のNNスコア・BAM・モデルは変更しません。

## 解釈上の区別

本法の最終出力はGLORI/RMBaseでレスキューし、DRACH候補はNN結果にかかわらず保持します。
最終レスキューは文書での説明にとどめ、ベン図には使用しません。
`Results_HeLa_m6A_ja.txt`と`Results_and_figure_legends_en.txt`に結果・図注・最終レスキューの説明を保存します。本文Word22は直接上書きしません。
HeLaを含む過去のCVはモデル開発で参照済みなので、HEK293T-only重みの転用結果を未使用の独立外部検証とは呼びません。
GLORIの陰性部位・全short-read coverageはないため、参照非一致を偽陽性、参照全件の未回収を偽陰性として数えません。
`HeLa_GLORI_recovery.tsv`はDRS被覆GLORI陽性の回収率を示し、precisionやspecificityではありません。
m6Anetの最低20read・MAPQ20・信号取得条件は、本法の総depth10・MAPQ0と異なります。
既存のPOD5欠測とisoform集約、exon境界の監査情報も保存し、共通領域の副解析を全カタログ比較と区別します。

小規模検証：

```bash
.venv310/bin/python -m unittest tests.test_transcript_coordinates tests.test_hela_m6a_comparison
```
