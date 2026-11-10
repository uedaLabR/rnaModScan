# 学習・推論共通の部位Parquet（2026-11-06）

1行は1sample×chromosome×position×strandのadenine候補です。学習と推論で同じ形式を使います。

| 列 | 形式・意味 |
| --- | --- |
| row_id/sample_id/cell/reference_cell/condition | 部位IDと明示した細胞・WT/IVT対応 |
| chrom/start/end/position/strand | GRCh38 BED0、position=start、end=start+1 |
| reference_sequence | RNA方向の19塩基、大文字 |
| sequence_tokens | 重複3-merの17 int32 |
| sequence_repeat_mask | 同じ19位置のbool。FASTA小文字acgt=1、N/padding=0。負鎖は逆順 |
| read_features | R×19×18 float32。R<=300、実readだけ |
| raw_reads | 同じR本の原ML/quality/base/CIGAR等。欠測と実測0を区別 |
| eligible_depth/retained_depth | 全適格read数と保存本数min(depth,300) |
| support_short_read_m6A/inosine | 同じreference_cellの実験参照支持、ソースID別列 |
| support_rmbase_m6A/inosine | RMBase3.0支持、ソースID別列 |
| training_class/training_label_m6A/inosine/false_positive | 既存ラベルとone-hot教師。推論では全てnull |
| AF/count/gene/region等 | bedMethyl全read由来の値と報告用メタデータ |

参照支持はtrue=一致、false=適用可能参照への非一致、null=参照なしです。false/nullから負例を作りません。
原AFの分母はN_valid_cov+N_fail+N_diff+N_nocall、af_passは監査専用です。

schemaは`site_bags_parquet_v3`、モデルfeature_setは`sequence_reads_repeat18_v1`です。
`references.fasta`一つにsoftmasked genomeを指定し、窓のFASTA取得1回で配列とrepeatを同時に作ります。
read座標変換はalignmentごと1回、抽出中のreservoir samplingはseed42/最大300本です。
Parquetは`features/samples/<sample>/sites/chrom=<chr>/strand=plus|minus/part-*.parquet`へ保存します。
RAMのNN batchだけ300枠にpadし、実read数からread_maskを作ります。ゼロ埋めreadは保存しません。

索引`features/index/adenine/sites.tsv`はschema `site_bag_index_v3`です。
site_parquet/row_group/row_offsetで保存行を指し、CVごとに特徴を複製しません。
重複19nt窓のgroupを保ってCVを作り、HEK-only subsetも同じ元特徴を参照します。

予測CPU準備はrawをhardlinkまたはcopyして、chr/strand別のprepared `prediction_inputs_v9`へ保存します。
inferは準備済み行の4入力だけを読み、元BAM/FASTA/DBを開きません。
旧v1/v2部位Parquet・NPY・別maskファイルは現行fitへ混用しません。
旧結果データ・学習ログ・原稿は保存し、現行の再抽出出力は`results/training_v8`に分けます。

## 2026-11-06 最終reservoir確定後の変換

保存schemaと18chは同じです。抽出中のbagにはDeferredReadへの共有参照を置きます。
CIGAR解析と窓の座標対応だけを先に作り、部位の最終最大300本が確定してからMM/ML、配列、quality、raw/NN特徴を作ります。
同一alignmentが複数部位の最終集合へ残る場合は、共有context内のMM/ML結果を再利用します。
最後の未完了部位から参照がなくなったcontextは解放され、whole-BAM cacheは作りません。
`run.json`の`sampling_feature_order=final_reservoir_then_mm_features_v1`で実行方式を記録します。
`modification_decodes`は最終採用alignmentの和集合の復号数、`encoded_read_site_windows`は作成した最終窓数、
`reservoir_admissions`は置換前を含む一時採用数です。MMチャネル観測監査も最終採用alignmentだけの集計になります。
今回のtraining_v10は4 BAMとも新規抽出し、過去データの再利用を行いません。
