"""Write manuscript-ready text and audit the exact memberships behind the Venns."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.hela_m6a.compare import read_tsv
from utils.references.catalog import sha256


def write_report(output):
    run = json.loads((output / 'run.json').read_text())
    if run['status'] != 'complete' or (output / 'INCOMPLETE.txt').exists():
        raise ValueError('Comparison is incomplete')
    partitions = list(read_tsv(output / 'HeLa_venn_regions.tsv'))
    verified = []
    for path in sorted(output.glob('membership_*.tsv')):
        name = path.stem[len('membership_'):]
        expected = {row['pattern']:int(row['count']) for row in partitions if row['name']==name}
        methods = ['GLORI','m6Anet','rnaModScan'] if name.startswith('DRACH__') else ['GLORI','rnaModScan']
        actual, sites = Counter(), set()
        for row in read_tsv(path):
            key = row['chrom'], int(row['start']), row['strand']
            if key in sites:
                raise ValueError('Duplicate membership coordinate')
            sites.add(key)
            actual[''.join(row[method] for method in methods)] += 1
        if {key:actual[key] for key in expected} != expected or sum(expected.values()) != len(sites):
            raise ValueError('Venn partitions do not match membership table')
        verified.append(dict(view=name, unique_sites=len(sites), sha256=sha256(path)))
    totals = {(row['name'],row['method']):int(row['sites']) for row in read_tsv(output/'HeLa_set_sizes.tsv')}
    transitions = {row['region']:{key:int(value) for key,value in row.items() if key!='region'}
                   for row in read_tsv(output/'HeLa_filter_transitions.tsv')}
    counts = {(row['name'],row['pattern']):int(row['count']) for row in partitions}
    d, n = 'DRACH__NN__full_catalog', 'non_DRACH__NN__full_catalog'
    g_d,g_n = totals[d,'GLORI'],totals[n,'GLORI']
    net,ours_d,ours_n = totals[d,'m6Anet'],totals[d,'rnaModScan'],totals[n,'rnaModScan']
    triple, non_overlap = counts[d,'111'],counts[n,'11']
    glori_nn_d = counts[d,'101'] + triple
    glori_net = counts[d,'110'] + triple
    cutoff = run['notes']['m6anet_threshold']
    sizes = list(read_tsv(output/'HeLa_set_sizes.tsv'))
    sensitivity_values = sorted({float(row['m6anet_threshold']) for row in sizes if float(row['m6anet_threshold']) != cutoff})
    sensitivity_counts = {value:next(int(row['sites']) for row in sizes if row['region']=='DRACH'
                          and row['stage']=='NN' and row['universe']=='full_catalog' and row['method']=='m6Anet'
                          and float(row['m6anet_threshold'])==value) for value in sensitivity_values}
    sensitivity = ', '.join(f'{value:.2f}' for value in sensitivity_values)
    threshold_results = '; '.join(f'>= {value:.2f}: {sensitivity_counts[value]:,} sites' for value in sensitivity_values)
    before_d,before_n = transitions['DRACH']['dorado'],transitions['non_DRACH']['dorado']
    final_d,final_n = transitions['DRACH']['final'],transitions['non_DRACH']['final']
    english = f'''HeLa WT m6A comparisons: results and figure legends (completed)

Results
The HEK293T-only rnaModScan classifier retained {ours_d:,} of {before_d:,} DRACH candidates ({100*ours_d/before_d:.2f}%) and {ours_n:,} of {before_n:,} non-DRACH candidates ({100*ours_n/before_n:.2f}%) by m6A softmax argmax, before database, assay-reference or DRACH retention. The respective GLORI catalogues contained {g_d:,} and {g_n:,} sites. Raw NN predictions overlapped {glori_nn_d:,} DRACH and {non_overlap:,} non-DRACH GLORI sites. m6Anet at site probability >= {cutoff:.2f} yielded {net:,} genomic DRACH sites, including {glori_net:,} GLORI matches; {triple:,} DRACH sites were shared by all three sets.

The primary m6Anet threshold of {cutoff:.2f} was selected for this descriptive comparison after inspecting the number of calls at 0.9. Supplementary threshold sensitivity yielded {threshold_results}. These choices are not independently optimized or calibrated to equal false-positive rates across methods.

Final reporting (text only; not the sets shown in the Venn diagrams)
The public workflow additionally retains original candidates supported by database or cell-matched assay references and all original DRACH m6A candidates. This yields {final_d:,} DRACH and {final_n:,} non-DRACH m6A sites. Because GLORI contributes to this rescue, final-reference agreement is not an independent assessment of NN accuracy. No final-rescue Venn diagram is included.

Main figure legend
HeLa WT m6A overlap before reference/motif rescue. (A) Genomic DRACH sites: GLORI, m6Anet (site probability >= {cutoff:.2f}) and rnaModScan m6A softmax argmax. (B) Genomic non-DRACH sites: GLORI and rnaModScan m6A softmax argmax; m6Anet is not treated as a non-DRACH comparator. Matches require identical GRCh38 chromosome, BED0 center and strand. Transcript isoforms are collapsed to one genome locus/strand using maximum m6Anet probability, without adding read counts across isoforms. The full positive GLORI catalogue is used in this main comparison. Circle areas are schematic; labels are exact disjoint counts. Reference non-overlap does not establish a false positive or false negative.

Supplementary figures
The m6Anet thresholds {sensitivity} repeat the comparison while leaving rnaModScan decisions unchanged. The covered/scored panel restricts all DRACH sets to positions actually scored by m6Anet, whereas the non-DRACH panel restricts GLORI to original-pileup total depth >=10; these denominators are distinct. The Dorado panel shows threshold-qualified candidates before NN filtering, with the primary m6Anet threshold.

Coordinate/coverage audit
All 184,624 m6Anet transcript-site 5-mers were checked against the original transcript FASTA and mapped using exact versioned GENCODE v44 exon coordinates, including minus-strand transcripts. They collapse to 175,079 genomic loci. All mapped centers agree with genomic-strand A. Of these, 225 have non-DRACH genomic 5-mers because transcript and genomic contexts can differ at exon junctions; one passes 0.8 and none passes 0.9. These are retained in the audit and excluded from the primary genomic-DRACH m6Anet set. The original Dorado non-DRACH candidates include 1,292 non-A genomic centers; all are excluded by the NN. GLORI coverage comes from the verified scan of the same bedMethyl input, not from a new BAM scan. Full short-read negative-site coverage is unavailable. Previously pooled development included HeLa; these descriptive transfer overlaps are not an untouched external chemical-accuracy evaluation.
'''
    japanese = f'''HeLa WT m6A比較：今回の確定方針と結果

本文図はレスキュー前のNN判定のみ。m6Anetは部位確率{cutoff:.2f}以上、{sensitivity}はサプリ。
DRACH内：GLORI {g_d:,}、m6Anet {net:,}、NN {ours_d:,}。3者共通 {triple:,}。
non-DRACH：GLORI {g_n:,}、NN {ours_n:,}、共通 {non_overlap:,}。
元Dorado候補はDRACH {before_d:,}、non-DRACH {before_n:,}。NNによる保持率はそれぞれ{100*ours_d/before_d:.2f}%、{100*ours_n/before_n:.2f}%。

最終レスキューは文書でのみ説明：DB/同一細胞GLORI等の根拠またはDRACH保持を追加すると、DRACH {final_d:,}、non-DRACH {final_n:,}となる。最終出力のベン図は作成しない。

全カタログ比較と被覆範囲の副解析を混ぜない。GLORI非一致を偽陽性とは数えず、参照陰性のcoverageがないためprecision/specificityを主張しない。m6Anet0.8は今回指定された比較用の閾値であり、公式推奨0.9や本法のsoftmax argmaxと同じ判定規則ではない。
本文Word22は今回直接変更していない。本文への追記用英文と図注はResults_and_figure_legends_en.txtに保存。全図の数値は60シートの累積Excelに追加し、旧52シートの値は同一性を確認した。全座標・所属・フラグはmembership_*.tsv。
'''
    for filename, content in [('Results_and_figure_legends_en.txt',english),('Results_HeLa_m6A_ja.txt',japanese)]:
        path=output/filename
        if path.exists():
            raise FileExistsError(path)
        path.write_text(content)
    verification=dict(status='complete',membership_tables=verified,
        original_workbook_sheets_preserved=run['workbook']['original_sheets'],
        comparison_workbook_sheets=run['workbook']['sheets'],
        figure_scope='NN only; final rescue described in text',
        main_m6anet_threshold=cutoff,supplement_m6anet_thresholds=sensitivity_values)
    (output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    return verification


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    print(json.dumps(write_report(parser.parse_args().output),indent=2))
