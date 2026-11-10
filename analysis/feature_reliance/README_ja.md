# 学習済みNNの入力群への依存度

保存済みのCVモデルと、そのfoldのtestデータを使い、入力を変えたときのF1低下を調べます。重み・ラベル・Parquet・既存の予測結果は変更しません。BAM抽出や再学習も行いません。

最初の対象はfold0のtest全109,747件です。m6A正例38,944件、Inosine正例1,439件、false_positiveクラス69,364件を含みます。これは既存の運用上の弱ラベルによる内部評価です。

| 入力群 | 変更する内容 | 維持する内容 |
|---|---|---|
| 参照配列 | 19塩基からなる17個の重複3-merを部位間でまとめて交換 | read側の塩基・MATCH情報、repeat、他の入力 |
| repeat | 19塩基のマスクを部位間でまとめて交換 | 配列とread入力 |
| 周辺修飾スコア | 中心以外の18位置で8スコアの組を置換 | 中心スコア、塩基の適用条件、read数・mask |
| quality | 取得済みの正の逆quality値を置換 | 欠測・未被覆・paddingの状態 |

参照情報の交換は保存read数の区分内で行います。readスコアは、観測塩基・相対位置・保存read数区分が同じ観測値から置き換えます。個々のML値を別々に混ぜず、8種類の組を保持します。一方、元のread内・部位内の相関は崩れるため、結果はこの置換条件におけるモデルの依存度として解釈します。

補助条件は「配列＋repeat」「中心MLのみ」「全ML」です。3つの乱数seedで反復します。主指標は、通常入力からのF1低下幅（percentage points）で、macro-F1、DRACH m6A、DRACH外m6A、Inosineを出します。precision・recall・accuracy・log loss・判定変更率も保存します。m6Aの各motif別F1では、そのmotifに属する全クラスを評価に含めます。

SDは同じモデルでの3回の置換のばらつきです。fold間SDや信頼区間ではありません。入力の相関や重複情報があるため、4群の値を合計100%に正規化しません。参照配列枝を変えてもread側には配列関連情報が残り、qualityの欠測パターンも保持されます。「その入力を除いて再学習したモデル」との比較は別の実験です。

## 実行と確認

各実行は新しい出力先を要求します。同じ出力先の完了済み・実行中ジョブを再起動しないでください。起動には接続切断後も続く`--background`を使えます。

```bash
bash prediction/test/12_analyze_feature_reliance.sh \
  --fold 0 --device gpu --gpu 0 \
  --batch-sites 128 --inference-batch-sites 1024 --pool-batch-sites 256 \
  --output results/feature_reliance_fold0_20261109 --check
```

`--check`は出典と設定の確認のみです。実行時は`--check`を`--background`に変更します。CPUでは`--device cpu`を指定できます。保存済み入力や重みを変更する処理はありません。

進捗は出力先の`progress.json`、完了状態と指標は`run.json`、実行ログは出力先と同じ階層の`.launch.log`にあります。`scores.npy`は全条件のスコア、`sites.tsv`は行と座標・ラベルの対応です。コードと出典のSHAも保存します。

完了後にExcel・図を作成します。pilotや不完全な結果は受け付けません。

```bash
.venv310/bin/python analysis/feature_reliance/summarize.py \
  --run results/feature_reliance_fold0_20261109 \
  --output paper/data/feature_reliance_fold0_20261109 \
  --figures paper/figures/feature_reliance_fold0_20261109
```

バッチサイズによるGPU丸め差を区別するため、各推論呼び出しに元入力のbaselineを含めます。同じ部位のbaseline判定が変わる場合や確率差が許容値を超える場合は、成功として扱いません。
