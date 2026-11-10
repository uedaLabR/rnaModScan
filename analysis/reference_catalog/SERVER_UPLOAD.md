# サーバーへの最小参照データ転送

対象はHEK293TとHeLa、WT/IVTの4サンプルで共通に使う参照です。
サーバーにはプログラム本体、modkit pileup BED、pileup.log、GRCh38 FASTAと.faiがある前提です。
dataフォルダー全体とdocs（またはdoc）はアップロード不要です。

## 追加アップロード

このPCの次のZIPをサーバーに転送してください。

    C:\project\rnaMosScan\analysis\reference_catalog\exports\overlap_references_GRCh38_20261027.zip

ZIPには以下の必要部分だけを、プロジェクトからの相対パスで収録しています。

- data/references/catalog.json
- data/reference_sites_GRCh38/by_method/ のカタログ登録済みshort-read BED 11本
- data/reference_sites_GRCh38/sources.json
- references/rmbase3/ の4修飾のGRCh38 BEDとimport.json
- analysis/reference_catalog/ 配下の確認スクリプト、この手順、SHA256一覧

元のExcel、data/external、data/pileup、sourceのRMBase tar.gz、FASTA、大きなprovenance.tsvは含みません。
4サンプルとも同じ参照セットを使えます。short-readは対応する細胞株を、RMBaseは全ヒトDBを参照します。

## 展開と確認

サーバーのrnaMosScanプロジェクト直下へ移動し、ZIPの実際の転送先を指定します。

    cd /実際のパス/rnaMosScan
    unzip -n /転送先/overlap_references_GRCh38_20261027.zip -d .
    python -B analysis/reference_catalog/check_server_references.py

Python 3.10以上、標準ライブラリのみ。既にアップロード済みのrnamodscan/パッケージが必要です。
unzipの-nは既存ファイルを上書きしません。既存の参照が異なる場合、確認スクリプトが不一致で停止します。
その場合は表示されたファイルを確認し、このZIPの対応ファイルを別名で保管した旧ファイルと入れ替えてから再確認してください。
確認スクリプトは15参照のハッシュ・BED形式・部位数を検査します。pileupの全量読み込みは行いません。

## 集計実行

analysis/pileup_overlap/run_overlap.py の次の設定をサーバーに合わせます。

- REFERENCE_FASTA：今回のアラインメントと同じGRCh38 FASTA。非圧縮FASTAと対応する.faiが必要。
  初期値はログの表記どおり /sahre/reference/hg38.fa です。実際の綴りを確認してください。
- SAMPLES内のbedmethylとpileup_log：HEK293T WTの実際のファイル。BEDは非圧縮または.bed.gz/.bgz。
- OUTPUT_DIR：まだ存在しない出力先。既定はanalysis/pileup_overlap/results/HEK293T_WT_v1。
- REFERENCE_CATALOG：既定のROOT / 'data/references/catalog.json'のまま使えます。

HEK293T WTだけenabled=Trueの状態で、プロジェクト直下から実行します。

    python -B analysis/pileup_overlap/run_overlap.py

SSH切断後も続ける場合の例（ログ名は実行ごとに変更）：

    nohup python -u -B analysis/pileup_overlap/run_overlap.py > HEK293T_WT_overlap_v1.console.log 2>&1 &
    tail -f HEK293T_WT_overlap_v1.console.log

完了時は出力先のrun.jsonがstatus=completeとなり、INCOMPLETE.txtが消えます。
FAILED.txtやINCOMPLETE.txtが残っている場合は完成結果として使わないでください。
集計表はcall_counts.tsv、overlap_counts.tsv、reference_overlap_summary.tsvです。
DRACH/non_DRACH判定を行い、既定では巨大な部位別一覧は保存しません。

## ZIPの再作成（ローカルPCのみ）

参照を更新した際はpackage_server_references.pyのOUTPUTを新しいファイル名に変更します。
現在の整形済み参照から作り、既存ZIPには上書きしません。

    python -B analysis/reference_catalog/package_server_references.py

この再作成処理だけは、ローカルのdata/references/catalog.jsonと登録済み参照一式が必要です。
サーバー上では再作成や参照の再構築を行う必要はありません。
