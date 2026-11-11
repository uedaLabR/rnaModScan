# short-readとRMBaseの参照準備

通常のヒト予測に必要な整形済み参照は、`data/references/hg38/` に同梱しています。
17個の圧縮BEDと3個の元出典JSONを含み、元BEDのバイト列を損失なくgzip圧縮しています。
`genome: hg38`、`references.catalog: null` で同梱版カタログを選択します。
手動展開・元アーカイブの取得・参照再構築は不要です。ゲノムFASTA/FAIは別途必要です。
`data/references/hg38/README.md` と `SHA256SUMS` を参照してください。

旧 `data/references/catalog.json` と元の整形済みファイルは、過去の解析用に保持しています。
このフォルダーの再構築スクリプトは、元データから新しい参照を作り直す場合だけ使用します。

集計・確認スクリプトをこのフォルダーへまとめています。データ本体は複製していません。

- `prepare_reference_catalog.py`：既存short-read BEDとRMBase 3.0を登録。RMBase未整形時はsourceのhg38 tar.gzを直接読み、references/rmbase3へ整形。既存整形BEDはハッシュ照合して再利用。
- `rebuild_short_read_beds.py`：原著表からshort-read BEDと手法別件数・和集合等を再構築する既存スクリプト。openpyxlが必要。出力はanalysis/reference_catalog/results/short_read_rebuildで、既存ディレクトリには上書きしません。今回は実行しません。
- `legacy/update_praise_reference.py`：旧PRAISE更新スクリプトを履歴として保存。旧DドライブやDownloadsへの絶対パス、原本コピー・既存出力の更新処理を含みます。現在の解析には使用せず、既存の確定BEDを使ってください。

通常のpileupとの比較では参照の再構築は不要です。`analysis/pileup_overlap/run_overlap.py` を使います。
新しくカタログを準備する場合のみ、設定を確認して次を実行します。

    python -B analysis/reference_catalog/prepare_reference_catalog.py

新規カタログの既定保存先はdata/references/catalog.jsonです。既存ファイルには上書きしません。
再構築結果を正式参照に採用する場合は別途内容と出典を確認してください。整形済みデータの場所は変更していません。
