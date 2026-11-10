# HEK293T IVT抽出の性能調査（2026-11-06）

稼働中のtraining_v9を止めずに調査した。productionのPython・設定・出力schemaは変更していない。
計測・試作は独立processで行い、NN学習用データへ混ぜない。

## 何に時間を使うか

稼働workerの5秒観測でCPUは約95–100%（1core分）、major faultとiodelayは0。
サーバー全体の短時間観測はCPU idle約93–94%、swap-in/outは0だった。
この時間帯では、CPU上の処理を改善する余地が大きい。NFS/page cacheの影響は別途あり、
この短い観測だけで常時I/O待ちがないとは断定しない。

HEK293T IVTのchr1先頭・chr1 30M以降・chr12先頭から48候補ずつ、最大8192alignment/範囲を別途取得。
cProfileによる局所計測（未完了coverage、人工的な書出し境界、計測自体のoverheadあり）：

| 処理 | 計測時間に占める割合 |
| --- | ---: |
| 周辺窓のraw特徴作成 | 38–42% |
| MM/ML関連（下の復号を含む） | 17–21% |
| うちHTSlib/pysamの復号・返却と空group確認 | 9–11% |
| NN用18ch数値化 | 14–16% |
| Arrow変換・Parquet圧縮書出し | 11–15% |
| CIGAR構築＋短い窓への座標対応 | 2–3% |

含まれる処理を重ねて足さない。これらは全BAM実行時間の比率ではない。
MM/MLの復号本体はpysam.modified_bases_forward経由のHTSlib。
必要位置だけPython辞書へ保持しても、pysamから返される修飾位置一覧はread全体分になる。
今回の3範囲では、その約9–10%のエントリを必要窓の辞書へ残していた。

raw特徴作成は単なるタグ取得ではなく、各位置で8チャネルの辞書検索、null処理、観測/適用監査mask、
塩基・quality・mismatch・挿入・splice情報の整形を行う。
300read×19位置×18chで、NN数値だけでも1site最大102,600要素になる。
raw監査情報も同じParquetに保存している（局所ファイルのraw_reads列は圧縮列bytesの約4割）。

## 300read samplingの確認

採用判定はMM/ML復号・窓特徴作成より前。あるreadが全ての対応候補で不採用なら、それらは実行しない。
別の候補で採用されたreadは、その必要窓だけを作る。quality取得も採用後。
ただし配列文字列の取得は採否判定前にあるため、そこには小さな無駄が残る。

現行はstreaming reservoirで、一度採用して特徴を作ったreadが後で置き換えられる。
保存する最終read数が300でも、特徴作成回数が300とは限らない。
実装のseedを使う単純な1site模擬でdepth10000なら1343回一時採用、8657回即時不採用、最後に300本。
これは実BAMの実測本数ではなくsampling手順の確認例。
実際のchr1 30M局所範囲でも4438read-site窓を作成し、最終保持は3724read-siteだった。

稼働中のHEK IVT先頭8完成Parquetの2048siteをmetadataだけ読み、153siteがdepth>300、
最大eligible_depth18815に対してretained_depth300を確認。全2048siteでmin(depth,300)と一致した。

## 前回の高速化の効き方が違う理由

最初の約5分の旧ログを、同じ完了site数まで線形補間して比較すると、訪問alignment数は
HEK WTで約56%減、HEK IVTで約13%減、HeLa IVTで約9%減だった。
この局所比較ではHEK IVTに対する候補group化の読み飛ばし効果が小さい。
native処理が速くても、その後のPython窓処理が残る。
旧4BAM並列と新3BAM並列・cache/負荷の違いもあり、純粋な実装改善率とは扱わない。

## 次に改善する候補

1. **最終300本を確定してから特徴を作る。** CIGAR後はreservoirに共有read contextを保持し、
   site完了時に残ったreadだけMM/ML・sequence・windowを作る方式。
   同一alignmentが複数siteで選ばれる場合もCIGAR/MMを1回にするため、共有contextと必要query位置集合を管理する。
   完了bagと共有参照を解放する。whole-BAM cacheは作らない。
   生BAM recordを遅延保持するため、長いread/spliceでのmemory上限と同等性確認が必要。今回未実装。
2. **query位置ごとに8修飾を一つの配列で保持する。** 現在の(q,modification)検索8回を、q検索1回へまとめ、
   塩基ごとの適用maskも再利用。別process内の試作で144site・7521保持read-siteのraw/tensor/metadataが完全一致。
   計測器なし・同じcached BAM入力・ローカル一時書出し・各方式3反復の中央値で1.17–1.30倍。
   全量/NFS込みの短縮率ではない。production未適用。
3. **染色体ごとのprocess並列化。** 現在はBAMあたり1processでPython処理がほぼ1core。
   染色体間はalignmentが独立なので、同一alignmentの1回変換を保ちやすい。
   2–4worker/BAMからの比較が候補。RAM/NFS負荷、出力結合、seed順序を確認してから採用する。速度未測定。
4. 数値化の配列演算、Arrowへの列単位変換、rowgroup幅調整も候補だが、まず上記の不要変換削減を優先する。

本調査では変更を稼働jobに適用していない。現在の抽出を完了させる。

## 保存物

- `results/model_checks/hek_ivt_profile_20261106/profile.json`と各`.prof`/`.txt`：現行の局所計測。
- 同directoryの`packed_cpu_benchmark.json`：試作A/Bの反復計測。
- 同directoryの`verification.json`：全列同等性、source SHA不変、同じPIDでの進捗継続。
- 同directoryのPython3本：調査・試作のsnapshot。実行scriptは出力先が重複しないよう設定して使う。
- `results/model_checks/hek_ivt_profile_packed_20261106/`：独立process内試作の比較出力。

いずれも局所/部分coverageの診断用で、NN学習データや生物学的性能評価には使わない。

## 続報：最終確定後の変換を実装

ユーザーの追加依頼でv9を停止し、上記候補1をDeferredReadとして実装した。
各部位の最終reservoirが完成するまでMM/ML・配列・quality・raw/NN変換を遅延する。
共有contextでCIGAR/HTSlib MM復号を各alignment最大1回に保つ。候補2のpacked試作は今回未適用。
新旧比較の3局所範囲＋depth18815の実部位で、最終raw/NN/metadataは完全一致。
深い部位はMM復号1524→300、窓作成300、CIGAR19898→19898、最終保存300で一致した。
cached入力の局所計測で深い部位は約2倍だったが、低depth範囲では同等～遅くなる例もある。全run速度は別途確認する。
証跡はresults/model_checks/deferred_sampling_20261106。過去runは保存し、新training_v10で4 BAMをすべて再抽出する。
