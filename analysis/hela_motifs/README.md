# HeLa non-DRACH m6A：周辺配列とWebLogo入力

レスキュー前の本法NNとHeLa GLORIのnon-DRACH集合を使い、候補中心の前後5塩基（計11塩基）を
GRCh38ゲノムから取得します。負鎖はreverse-complementしてRNAの5′→3′方向に揃えます。
転写産物やexonを連結した配列ではなく、同じゲノム座標の局所配列です。

```bash
cd /home/ueda/PycharmProjects/rnaModScan
.venv310/bin/python -m unittest tests.test_hela_motifs
MPLCONFIGDIR=/tmp/rnamodscan-mpl .venv310/bin/python analysis/hela_motifs/compare.py --config analysis/hela_motifs/config.yaml
```

入力・出力パスは`config.yaml`にあります。既存の出力先へは実行しません。
BAM、NN学習、既存の推論結果、以前の図やWordは変更しません。

## WebLogoへ渡すファイル

`paper/data/hela_non_drach_motifs_20261108/weblogo_inputs/`に、以下の5群ごとにRNAとDNAのFASTAを保存します。

| 群 | 内容 |
|---|---|
| `GLORI_all` | GLORIのnon-DRACH全候補 |
| `NN_all` | 本法NNがm6Aとしたnon-DRACH全候補 |
| `GLORI_only` | GLORIだけにある部位 |
| `NN_only` | 本法NNだけにある部位 |
| `shared` | GLORIと本法NNの共通部位 |

`*_RNA.fa`はA/C/G/U、`*_DNA.fa`はA/C/G/Tです。それぞれRNA/DNAの配列としてWebLogoへ入力できます。
配列は11列で、候補Aは6列目（one-based）です。表示を−5〜+5とすると位置0が候補です。
同じ群の中では同一座標・鎖を1回だけ含み、read数で配列を重み付けしません。

FASTAは11塩基すべてが既知のウィンドウだけを使用します。Nやcontig端のpaddingがあった場合は除外数を監査し、
元配列・座標は`site_sequences.tsv`へ残します。今回の56,979部位にはN/padding/非A中央塩基はありませんでした。
外部WebLogoへはアップロードしていません。

## 元データと集計

- `site_sequences.tsv`：全56,979部位の座標・所属・11塩基配列（DNA/RNA）・元Dorado総depth/NN score監査・中央5-mer。
- `group_summary.tsv`：各群の全件数、11塩基既知件数、欠測/非A/padding監査、DRACHから1塩基差の割合・平均距離。
- `position_frequencies.tsv`：各群×位置−5〜+5×塩基のcount/frequency。全元部位と完全11塩基部位の両方を保存。
- `central_5mers.tsv`：中央5-merの全種類・個数・割合。
- `drach_distance.tsv`：DRACHの許容配列からの最小置換数別の個数・割合。
- `drach_mismatches.tsv`：DRACHの各位置（−2、−1、0、+1、+2）で許容塩基から外れる割合。
- `frequency_similarity.tsv`：2群ごとの位置別塩基頻度のJensen–Shannon divergence（base2、bits）とL1距離。
- `weblogo_manifest.tsv`：各FASTAの件数、除外数、SHA256。
- `excel_sheets.json`：累積Excelへ取り込む8シートのTSVと数値列定義。空の数値セルは0へ変換せず空欄のままにする。

最小置換数は中央5-merの各塩基をDRACHの許容塩基集合D={A,G,T}、R={A,G}、A、C、H={A,C,T}と比較した不一致数です。
Nは不一致とはせず欠測にします。今回の入力はnon-DRACHに限定されているため、距離0がないこと自体は選別条件によります。
Jensen–Shannon divergenceは位置ごとの塩基頻度分布の記述的な差で、0なら同一です。

## 観測された傾向と解釈

DRACHから1塩基差の割合はGLORI全体66.31%、本法NN全体78.04%、共通群98.39%でした。
ただし、NNだけの群ではCAACA、CCACA、CTACAなどが多く、共通群ではGGACG、GGATT、GTACTが多いため、
「DRACHへの最小置換数が小さい」ことと「GLORIと同じモチーフ分布」であることは区別します。
NN全体では+1のCが91.18%、GLORI全体では46.24%で、配列構成には差があります。

この解析は生の塩基構成とモチーフ距離の比較です。GC/depth/遺伝子/マッパビリティを合わせた背景や、
de novo motif enrichment、独立した化学的検証、統計的有意差を示すものではありません。
GLORI全体とNN全体は共通部位を含み、独立したサンプル群でもありません。
WebLogoの中心AやDRACHへの近さだけで、NNだけの候補が真のm6Aであるとは結論しません。

図は`paper/figures/hela_non_drach_motifs_20261108/`へ、SVG/PDF/PNGで保存します。
`FigS_HeLa_non_DRACH_base_frequencies`は塩基頻度、`FigS_HeLa_non_DRACH_distance`は最小置換数分布です。
