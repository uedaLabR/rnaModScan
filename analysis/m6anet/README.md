# HeLa WTのm6Anet実行準備

POD5、Dorado BAM、hg38 FASTA、GENCODE v44 GFF3の実在を確認済み。
スクリプトをユーザーが実行して環境作成・変換・m6Anet計算を開始します。
現在のtraining_v10抽出とは別工程で、専用環境と保存先を使います。

```bash
cd /home/ueda/PycharmProjects/rnaModScan
bash analysis/m6anet/00_setup.sh
bash analysis/m6anet/01_run_hela.sh --check
nohup bash analysis/m6anet/01_run_hela.sh all > m6anet_HeLa_WT.log 2>&1 < /dev/null &
```

`00_setup.sh`は既存の`/home/ueda/anaconda3/bin/conda`から専用tools環境を作り、
m6Anet2.1.0用のPython3.8＋CPU版PyTorch1.6.0（`.m6anet_py38`）と、
blue-crab用のPython3.10環境（`.venv_bluecrab`）を分けて用意します。
公式m6anet2.1.0のRequires-Pythonは`>=3.7,<3.9`、torchは`==1.6.0`です。
以前のPython3.10＋torch1.13.1の設定はこの指定と合わないため修正しました。
途中まで作った`.venv_m6anet`は使用せず、そのまま保持します。
既存のNN用`.venv310`・`.venv_cuda112`へ追加インストールしません。
setupにはインターネット接続が必要です。環境は再実行可能ですが、生成途中のデータは自動上書きしません。

`--check`で最初の5ツールが見つかり、`m6anet`と`blue-crab`だけ不足する場合：

```bash
cd /home/ueda/PycharmProjects/rnaModScan
bash analysis/m6anet/00_setup.sh --python-only 2>&1 | tee m6anet_setup_python.log
bash analysis/m6anet/01_run_hela.sh --check
```

このオプションは既存の5ツールを再インストールせず、Python環境の準備から続けます。
`--check`が成功してから`01_run_hela.sh all`へ進みます。セットアップは構文・fakeツールによる制御を検証し、
実ダウンロード・インストールはこの修正後にはまだ行っていません。

設定は`config.sh`。実在確認した入力は次のとおりです。

- POD5: `/mnt/ssdnas/riboML/HeLa/pod5_pass`（1473ファイル）
- Dorado BAM: `.../HeLa/PAU09364_pass_c7fa45d4/bam_pass/PAU09364_pass_c7fa45d4.sorted.bam`
- genome: `/share/reference/hg38.fa`
- annotation: `/mnt/ssdnas07/pipeline/rna_modscore_v2/source/gencode.v44.annotation.gff3`

全工程はreference→fastq→signal→align→index→eventalign→dataprep→infer。
個別実行例は`bash analysis/m6anet/01_run_hela.sh eventalign`です。成功した工程には`*.complete`を残します。
同じ出力先のconfig変更と重複起動を拒否します。失敗した途中データは保存し、新出力先または原因確認後の整理が必要です。

## POD5の信号破損エラーを除外して続行する

`signal`工程は`convert_pod5_files.py`でPOD5を1ファイルずつ独立したblue-crabプロセスへ渡します。
同時に動かす数は`SIGNAL_PROCESSES`（現在2）。`Compressed signal data is corrupt`、または入力信号のzstd展開で明示された`Data corruption detected`で失敗したファイルは、
そのファイル全体を除外して次へ進みます。容量不足など別のエラーは停止し、全ファイルが失敗した場合も成功にはしません。
途中まで読めたreadを含め、失敗したPOD5の全readがm6Anetから除外される方針です。

- `signal_conversion.tsv`：処理結果、入力POD5名、read数、ログ・出力パス
- `signal_conversion.json`：成功・除外・失敗ファイル数と除外read数（取得不能なread数は別集計）
- `signal_file_logs/`：POD5ごとの変換ログ
- `signal_records/`：処理中も保存されるファイル別結果、入力サイズ・mtime
- `blow5_parts_verified/`：変換成功とquickcheckを確認した出力だけ。結合はこのディレクトリのみ使用
- `signal_work/`：失敗途中のファイル。結合には使わない

再実行では確認済み出力を再利用します。入力の変更は拒否し、新出力先を要求します。
以前の一括変換でできた`blow5_parts/`は保存しておき、新方式では使用しません。
参照配列とFASTQの完了マーカーはそのまま利用できます。旧プロセスが残っている間はworkflow lockで再開を拒否します。
旧実行を終了してから、同じ設定で次を実行するとsignalから後続工程へ進みます。

```bash
nohup bash analysis/m6anet/01_run_hela.sh all > m6anet_HeLa_WT_resume.log 2>&1 < /dev/null &
```

2026-11-07に旧プロセスを終了し、この方式で再開済みです。入力信号のzstd破損にも対応し、17:11に再開しました。
起動記録は`restart_*.json`、起動ログは`resume_*.log`です。`signal_conversion.json/tsv`は直近終了時の集計なので、
再開中は前回のfailed集計が残る場合があります。処理中の最新進捗は`signal_records/*.json`と`logs/signal.log`で確認します。
比較時は信号欠測による解析対象外read/siteを記録し、m6Anetのfalse negativeと混同しないよう、実際の評価可能領域・depthを用います。

GENCODE GFF3からgffreadで転写産物FASTAを作り、同じDorado primary readをFASTQへ出して転写産物へ再アラインします。
現在のゲノムBAMをそのまま転写産物FASTAと組み合わせません。
POD5はblue-crabでBLOW5へ変換し、f5cでeventalignします。
RNA004 **5-mer**モデルを明示し、f5cの新しい既定9-merをそのまま使いません。
推論も`--pretrained_model HEK293T_RNA004 --device cpu --seed 42`を明示します。
CPU版なので、現在の古いCUDA環境の更新は不要です。

出力は`results/m6anet_HeLa_WT_v1/`です。

- `logs/<stage>.log`: 各工程のログ
- `eventalign.summary.tsv`: read_indexとread名等の対応
- `dataprep/`: m6Anet入力
- `inference/data.site_proba.csv`: 転写産物位置ごとのスコア
- `inference/data.indiv_proba.csv`: readごとのスコア
- `reference/sha256.txt`, `config_snapshot.sh`, `logs/*packages.txt`: 入力・設定・環境記録

**結果は転写産物座標です。** DoradoのGRCh38 BEDとの比較には、同じGFF3に基づくゲノム座標への対応と
isoform間の重複扱いの確定が別途必要です。今回のスクリプトはm6Anet実行までで、精度比較はまだ行いません。
公式2.1.0モデル推論は最低20readの設定を持ち、f5c MAPQ既定はこの設定では20です。
本ツールのdepth10/MAPQ0と単純に同一視せず、後の比較では共通に評価可能なDRACHを定義してください。
公式RNA004モデルは開発版chemistry由来でsite-level評価が限定的、read-level精度も未検証とされています。

原データのBAM/POD5は変更しません。signal変換・eventalignは時間とディスクを使うため、
1473個すべての変換を含む所要時間は実測後に確認します。

公式資料：
[m6Anet RNA004手順](https://github.com/GoekeLab/m6anet/blob/master/README.md)、
[f5c](https://hasindu2008.github.io/f5c/docs/commands)、
[blue-crab](https://github.com/Psy-Fer/blue-crab/blob/main/docs/cli.md)、
[公式推論の最低read数](https://github.com/GoekeLab/m6anet/blob/v-2.1.0/m6anet/utils/constants.py)。
環境の制約は[m6anet 2.1.0 PyPI](https://pypi.org/project/m6anet/2.1.0/)と
[公式setup.py](https://github.com/GoekeLab/m6anet/blob/master/setup.py)、
CPU版torchの導入は[PyTorch旧版の公式手順](https://docs.pytorch.org/get-started/previous-versions/)を参照しています。
