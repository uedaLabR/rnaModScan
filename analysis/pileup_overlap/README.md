# bedMethylの参照一致とDRACH集計

rnaModScanの集計工程です。学習・ラベル作成はしません。現在の設定はHEK293T WT/IVT、HeLa WT/IVTの4本。WTはpass readsのみ。
全量集計はサーバーで実行してください。このPCでは全量を実行していません。

## 起動

Python 3.10以上。集計は標準ライブラリのみ、ベン図にはmatplotlibを使います。初回のみ次を実行してください。

    python -m pip install -r analysis/pileup_overlap/requirements-plotting.txt

プロジェクト直下で実行します。

    python -B analysis/pileup_overlap/run_overlap.py

`run_overlap.py` 冒頭のSAMPLES、REFERENCE_FASTA、OUTPUT_DIRを編集してください。CLI引数はありません。
初期値のBED、log、FASTAは今回のpileup.logに記録されたサーバーパスです。
FASTAは実在する `/share/reference/hg38.fa` を設定済みです。
圧縮済みBEDの場合はbedmethylをその `.bed.gz` に変更します。展開・原本コピーは行いません。
FASTAは非圧縮と対応する `.fai` が必要です。BEDと同じGRCh38、同じ染色体名を使ってください。

## サーバーへ配置するもの

`data` 全体や `docs` を転送していない場合は、[最小ZIPの転送手順](../reference_catalog/SERVER_UPLOAD.md)を使ってください。
整形済みshort-read/RMBaseの参照とカタログだけを追加し、サーバーにあるBEDとゲノムを使います。

プロジェクト内の相対配置を維持します。

- `analysis/pileup_overlap/run_overlap.py`
- 共通コード `rnamodscan/`
- `data/references/catalog.json`
- `data/reference_sites_GRCh38/by_method/*.bed` と `data/reference_sites_GRCh38/sources.json`
- `references/rmbase3/*.GRCh38.bed` と `references/rmbase3/import.json`

既に整形済みの参照BEDを使用します。元のExcel、RMBase tar.gz、巨大なprovenance.tsvの再転送・再構築は、この集計には不要です。
カタログの実行用パスは相対パスです。sources.json/import.json内の過去の取得元パスは来歴として保持し、実行時入力には使いません。

## 設定

- depth >= 10、修飾リード数 >= 3、AF >= 0.1。
- AF = N_mod / (N_valid_cov + N_fail + N_diff + N_nocall)。欠失は含めません。
- 対象はm6A、m5C、inosine、pseUの4修飾。残りの修飾コードはobserved_codesに記録し、集計対象から除きます。
- HEK293TにはHEK293Tのshort-read、HeLaにはHeLaのshort-readを対応させます。
- 今回は固定ML=125ではありません。log上の自動閾値はA=0.5917969、C=0.6035156、G=0.7324219、T=0.6191406。
  各ファイルのlogから実際の条件を記録します。集約BEDからMLを再適用しません。
- MODEL名は入力BAMパスからの参考情報です。Doradoとmodkitの実際の版はNOT_RECORDEDを記入し直してください。
- 4サンプル実行時は残りのSAMPLESのパスとlogを設定し、enabled=Trueにします。
- OUTPUT_DIRは毎回新しい名前にします。既存出力は上書きしません。
- WRITE_CALLED_SITES=Falseが既定です。Trueでは部位表をgzip保存し、ディスク使用量が増えます。

## 結果

`results/pileup_overlap_v1/` に次を出します。

- call_counts.tsv：修飾別・閾値別call部位数と修飾リード数合計。m6Aはall/DRACH/non_DRACH/unknown。
- overlap_counts.tsv：short-read手法別・RMBase別・それぞれの和集合の一致数、一致数/call数。
- reference_overlap_summary.tsv：両方に一致、short-readのみ、DBのみ、どちらにも一致しない、の重複しない4区分。
- run.json：入力、設定、参照ハッシュ、logの閾値・コマンド、観測コード、処理行数、警告。
- called_sites.tsv.gz：必要なときのみ保存する部位別詳細。

m6Aのallは内訳の合計です。allと内訳を加算しないでください。
DRACHはRNAの鎖方向の5塩基で判定します。配列端・N等はunknownです。
short-read/DBと同染色体・同座標・同鎖・同修飾の完全一致を数えます。
RMBaseは全ヒトのDBであり、細胞特異的な真陽性集合とは扱いません。
WTの参照非一致は偽陽性と判定しません。IVTのcallは参照に一致しても運用上false callと数えます。
参照不足はunavailable/空欄とし、4区分も不完全な参照では空欄にします。

100万行ごとに進捗を表示します。入力は一度ずつ順次読み、ゲノムは小さな窓をキャッシュして参照します。
INCOMPLETE.txtまたはFAILED.txtが残る出力を完成結果として使わないでください。

## 確認テスト

    python -B -m unittest discover -s analysis/pileup_overlap/tests -v

テストは小さな人工BED/FASTAを使います。実データの集計結果ではありません。

## ベン図

run_overlap.pyはWRITE_VENN_PLOTS=Trueが既定です。集計完了後、出力先のvenn/にPNG・SVG・PDFを保存します。
サーバー更新時はrun_overlap.py、plot_overlap.py、requirements-plotting.txtと
rnamodscan/analysis/overlap_plots.pyを転送し、run_overlap.pyのサーバーパス設定を引き継いでください。
必要なコードだけのZIPはexports/venn_scripts_20261027.zipです（参照データは前回のZIPを使います）。

- 母集団（外枠）：そのサンプル・修飾・閾値でDorado/modkitがcallした部位。
- 左円：同じ細胞のshort-read手法の和集合との一致。右円：RMBase 3.0との一致。
- 左のみ・両方・右のみ・どちらにも一致しない、の4領域に部位数と全callに対する割合を表示。
- 円面積は件数に比例しない模式図です。図中にもその旨を記載します。
- 全ゲノムのshort-read/RMBase部位数を使う3集合ベン図ではありません。未検出の参照部位の数や感度は示しません。
- m6Aはall/DRACH/non_DRACH/unknown、m5C・inosine・pseUはall。
- HEK293T WT、HEK293T IVT、HeLa WT、HeLa IVTはサンプル別。現在は4サンプルすべて有効です。OUTPUT_DIRは新しい名前にしてください。
- 参照が不足する修飾は図をスキップし、plots.jsonに理由を記録。ゼロ件と取り違えません。
- 参照がそろっていてcallがゼロの条件も保存し、割合はn/aとします。
- 参照非一致は偽陽性の確定を意味しません。IVTは既定方針に従い全callを運用上false callと数えます。

図はreference_overlap_summary.tsvとrun.jsonだけから作成します。
既に集計済みなら、plot_overlap.pyのSUMMARY_DIRとOUTPUT_DIRを設定して次を実行すれば、
pileup・FASTA・参照BEDを読み直さずに作図できます。

    python -B analysis/pileup_overlap/plot_overlap.py

既存のvenn/があると停止します。作り直す場合はOUTPUT_DIRをvenn_v2など新しい名前にしてください。
図のplots.jsonがstatus=completeで、図の出力先にINCOMPLETE.txt/FAILED.txtがないことを確認してください。
作図に失敗しても、既に正常終了した集計表は保持されます。
matplotlibを用意せず表だけ作る場合はWRITE_VENN_PLOTS=Falseにします。

2026-11-02: 完了済み4サンプルをrun_overlap.pyに設定。結果は1つの出力先にsample_id列でまとめる。
HEK293T IVTの上流pileupに `~1 failed processing` の記録あり。出力は生成済みだが影響範囲は未確定。

2026-11-02 PUS7追加: catalogのPUS7_S1A/S1BはHepG2由来Nanoporeの追加正例。
手法別とshort_read_anyに含むが、これらは短鎖実験ではない。出典/細胞対応はcatalogに明記。
既存完了結果は新しいOUTPUT_DIRで再集計が必要。
