# HeLa inosine：Slic-seqとの一致とrepeat family

完了済みHeLa WTのinosine NN判定（softmax argmax、既知部位レスキュー前）と、同じ細胞株のSlic-seqを比較します。
参照はSlic-seqのみで、BACSとの和集合にはしません。元BAM・NN重み・学習ラベルは変更しません。

```bash
cd /home/ueda/PycharmProjects/rnaModScan
.venv310/bin/python -m unittest tests.test_hela_inosine
MPLCONFIGDIR=/tmp/rnamodscan-mpl .venv310/bin/python analysis/hela_inosine/compare.py --config analysis/hela_inosine/config.yaml
```

再実行時はYAMLで新しい`output`と`figures`を指定してください。既存の出力先を上書きしません。

## 比較と被覆範囲

染色体名・BED0座標・strandが完全一致する部位だけを共通とします。
本文用は全Slic-seqカタログとNN判定のベン図です。
副解析は、同じmodkit入力を以前走査した`reference_coverage.tsv`から総depth≥10のSlic-seq部位を取得します。
元bedMethylのファイル情報とSlic-seqのSHA・部位数が一致することを確認し、BAMの再走査はしません。
元Dorado候補で既に欠けていた参照部位と、NNで追加除外された部位を分けます。
参照非一致を偽陽性とは数えず、陽性参照の回収率を独立したprecision・感度と同一視しません。
最終レスキューは説明文と監査用件数だけを保存し、図は作りません。

## RepeatMasker family

UCSC hg38の17列rmsk表の`repFamily`列を使います（例Alu、L1、MIR）。`repName`や`repClass`と混同しません。
このrmskはプロジェクトで指定された既存ファイルで、元のUCSC公開日・download日は不明です。ファイルSHAを記録し、特定releaseを使用したとは断定しません。
部位中心がBED0半開区間`[genoStart,genoEnd)`に入る場合をoverlapとし、repeatの向きには制限しません。
同じfamilyが複数intervalに存在しても、その部位のfamilyは一度だけ数えます。

1部位を1回だけ数える構成比では、複数の異なるfamilyに重なる部位を`Multiple_families`にします。
全family名・class・repeat名は部位表に残します。併記するfamily別包含集計は同一部位が複数familyに入るため、合計が100%を超え得ます。
`Non-repeat`と`Unknown_contig`を明示し、分母は各群の全ユニーク部位数です。
FAIに存在するがrmskに全く現れないcontigは`Unknown_contig`とし、非repeatにしません。
FAIに存在しない部位はエラーです。

## 出力

- `inosine_membership.tsv`：全75,007部位の所属、座標、NN/元Dorado/Slic-seq、全family・class・repeat名・フラグ・score・AF/depth。
- `inosine_overlap.tsv`：本文・被覆副解析・元Dorado比較のベン図用排他的領域数。
- `inosine_repeat_summary.tsv`：repeat内・Alu内・非repeat・複数family・未知の件数と割合。
- `inosine_family_exclusive.tsv`：1部位1カテゴリの全family構成比。
- `inosine_family_inclusive.tsv`：全familyの重複を許した包含集計。
- `inosine_family_plot.tsv`：図の各カテゴリの元数値。上位8family以外は`Other_repeat`へまとめ、元表は全familyを保持。
- `inosine_transitions.tsv`：元候補・NN・被覆Slic-seqの回収と、文書用の最終レスキュー件数。
- `excel_sheets.json`：部位表を含む全7表の列型とシート名。rootの累積Excelへそのまま追加するためのマニフェスト。
- `Results_inosine_en.txt`：結果とMethodsの追記用説明。

図はSVG・PDF・PNGです。
`Fig_HeLa_inosine_Venn_NN_only`は本文用全カタログ比較、`FigS_HeLa_inosine_Venn_NN_covered`は被覆参照の副解析、
`FigS_HeLa_inosine_repeat_families`はサプリ用family構成比です。

このNNはrepeat-maskを入力します。Alu/repeat割合の上昇は記述的な性質で、独立した精度向上や背景を揃えたenrichmentの証拠とはしません。
