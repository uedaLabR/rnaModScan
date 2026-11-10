# HeLa inosine：遺伝子領域と領域別Alu割合

完了済みのSlic-seq・Dorado/modkit・レスキュー前NNの集合を、同じGENCODE v44注釈で比較します。
モデル入力、学習ラベル、予測値は変更しません。BAMも再走査しません。

実行方法：

```bash
.venv310/bin/python analysis/hela_regions/compare.py --config analysis/hela_regions/config.yaml
.venv310/bin/python -m unittest tests.test_hela_regions
```

`config.yaml`に部位一覧、GENCODE GFF3/GTF、出力先を指定します。既存出力を上書きしないため、再実行時は新しい出力先を指定してください。

## 領域の定義

- 座標は入力アセンブリGRCh38のBED0です。GENCODEの1始まり閉区間を0始まり半開区間へ変換し、同じ染色体・鎖の転写産物を照合します。
- 個別リードの転写産物を推定せず、重なる全isoformの領域を記録します。複数の異なる領域に該当する部位は`Ambiguous_regions`です。CDSなどを優先して曖昧さを隠しません。
- 5′/3′UTRは明示されたタグを尊重し、generic UTRおよび末端exonは同じcoding transcriptのCDS＋stop codonの両端との位置関係で区別します。マイナス鎖では5′/3′を反転します。CDSを持たない転写産物のexonは`noncoding_exon`で、UTRに含めません。
- 転写産物内でexonに含まれない位置をintronとします。少なくとも一つのisoformのCDS/UTRと別isoformのintronが重なる場合も曖昧部位です。
- 同じ鎖の転写産物がない部位は`outside_same_strand_transcript`、この注釈に染色体自体がない部位は`unknown_annotation`です。後者は今回alternate/random/unplaced contigです。暗黙の座標変換や染色体名変換は行いません。

`exclusive`集計では各部位を一回ずつ数え、分母を各集合の全unique部位とします。5′UTR/CDS/intron/3′UTRだけに絞った分母へ変更しません。曖昧・noncoding・その他・注釈なしも含めて100%です。
`inclusive`集計では各isoformで認められた全領域を数えるため、合計が100%を超えます。全isoformの`transcript_id=region`対応もTSVに保存します。

## 比較集合とAlu

全Slic-seq 26,372部位、以前の共通coverage集計でDRS総depth≥10だったSlic-seq 4,638部位、元Dorado/modkit 50,062部位、NN判定後8,290部位を主な比較に用います。Slic-seq/NN共有・固有集合も監査用に含めます。既知部位レスキュー後の集合は図に用いません。

Aluは既存のRepeatMasker中央座標overlapの`any_Alu`をそのまま使用します。複数familyと重なる部位もAluを一つでも含めばtrueです。領域ごとにAlu数/領域の全部位数を計算し、加えて全Alu部位がどの領域に存在するかを記録します。

出力：

- `region_sites.tsv`：75,007部位の集合フラグ、領域、全isoformの領域対応、Alu・repeat family。
- `region_summary.tsv`：全集合のexclusive/inclusive件数、分母、割合、領域内Alu割合、Alu内の領域構成。
- `region_plot.tsv`：図に描く全数値と分母。
- `region_patterns.tsv`：複数領域の組合せ。
- `excel_sheets.json`：上記4表の列名・型・行数。Excelへの取り込み用です。
- `run.json`：入力パス・SHA256、注釈ポリシー、集合件数、図。
- `FigS_HeLa_inosine_region_composition.{svg,pdf,png}`および`FigS_HeLa_inosine_Alu_by_region.{svg,pdf,png}`。

全Slic-seqにはintronが多く、DRSで被覆された部分集合では3′UTRを含むexonic領域が増えます。このデータから「short readなので3′UTRに偏る」という因果関係は結論できません。領域内でもAlu割合の差が残るかを記述的に確認します。領域構成・Alu割合は独立した化学的正確性の評価ではなく、NNにはrepeat-mask入力があります。
