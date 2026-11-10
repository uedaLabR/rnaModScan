"""HeLa NN inosine versus Slic-seq; exact overlap and unique-site repeat families."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.io.artifacts import file_info
from utils.io.bed import load_bed_sites
from utils.references.catalog import sha256

PATHS = ('predictions','prediction_run','slic','coverage','coverage_run','coverage_sources',
         'rmsk','genome_fai','output','figures')


def read_tsv(path):
    with Path(path).open(newline='') as handle:
        yield from csv.DictReader(handle, delimiter='\t')


def write_tsv(path, rows, fields):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def read_predictions(path, sample):
    records, nn, final = {}, set(), set()
    for row in read_tsv(path):
        site = row['chrom'], int(row['start']), row['strand']
        if row['sample_id'] != sample or row['modification'] != 'inosine' or site in records:
            raise ValueError('Unexpected/duplicate prediction row')
        if int(row['flag_nn']) != int(row['nn_class']=='inosine'):
            raise ValueError('NN flag is not the inosine softmax argmax')
        records[site] = dict(flag_nn=int(row['flag_nn']), final_keep=int(row['filter_pass']),
            flag_db=int(row['flag_db']),flag_short_read=int(row['flag_short_read']),
            score_inosine=float(row['score_inosine']),total_reads=int(row['total_reads']),af=float(row['af']))
        if records[site]['flag_nn']:
            nn.add(site)
        if records[site]['final_keep']:
            final.add(site)
    return records, nn, final


def checked_slic_coverage(config, slic):
    prediction = json.loads(config['prediction_run'].read_text())
    coverage = json.loads(config['coverage_run'].read_text())
    sources = json.loads(config['coverage_sources'].read_text())
    if prediction['status'] != 'complete' or coverage['status'] != 'complete':
        raise ValueError('Incomplete upstream run')
    if prediction['reports']['candidates']['source'] != coverage['input']:
        raise ValueError('Coverage came from another bedMethyl')
    source = next(ref for ref in sources['references'] if ref['id']=='HeLa.inosine.Slic-seq')
    if sha256(config['slic']) != source['sha256'] or len(slic) != source['unique_sites']:
        raise ValueError('Slic-seq differs from the coverage reference')
    minimum = coverage['thresholds']['min_depth']
    depths = {}
    for row in read_tsv(config['coverage']):
        if row['modification'] != 'inosine' or 'Slic-seq' not in row['reference_sources'].split(';'):
            continue
        site = row['chrom'], int(row['start']), row['strand']
        if site not in slic or site in depths or int(row['total_reads']) < minimum:
            raise ValueError('Invalid/duplicate covered Slic-seq site')
        depths[site] = int(row['total_reads'])
    return depths, minimum


def annotate_repeat_families(sites, rmsk, genome_fai):
    """Stream rmsk once; retain hits at requested positions, not the whole table.

    Repeat orientation does not filter a genomic interval. A site may belong
    to several families, each retained once. All coordinates are BED0 half-open.
    """
    with Path(genome_fai).open() as handle:
        lengths = {fields[0]:int(fields[1]) for fields in (line.split() for line in handle)}
    positions = defaultdict(set)
    for chrom, position, strand in sites:
        if chrom not in lengths or not 0 <= position < lengths[chrom] or strand not in ('+','-'):
            raise ValueError('Site is outside the input genome: ' + str((chrom,position,strand)))
        positions[chrom].add(position)
    positions = {chrom:sorted(values) for chrom,values in positions.items()}
    represented, families, classes, names = set(), defaultdict(set), defaultdict(set), defaultdict(set)
    digest = hashlib.sha256()
    interval_rows = relevant_rows = 0
    with Path(rmsk).open('rb') as handle:
        for raw in handle:
            digest.update(raw)
            if not raw.strip() or raw.startswith(b'#'):
                continue
            fields = raw.decode().rstrip('\r\n').split('\t')
            if len(fields) != 17:
                raise ValueError('Expected UCSC rmsk with 17 columns including bin')
            chrom,lo,hi = fields[5],int(fields[6]),int(fields[7])
            if not 0 <= lo < hi:
                raise ValueError('Invalid repeat interval')
            represented.add(chrom)
            interval_rows += 1
            if chrom not in positions:
                continue
            coords = positions[chrom]
            first,last = bisect_left(coords,lo),bisect_left(coords,hi)
            if first == last:
                continue
            relevant_rows += 1
            family = fields[12] if fields[12] not in ('','?') else 'Unclassified_repeat'
            for position in coords[first:last]:
                key = chrom,position
                families[key].add(family)
                classes[key].add(fields[11])
                names[key].add(fields[10])
    result = {}
    for site in sites:
        chrom,position,strand = site
        values = sorted(families.get((chrom,position),set()))
        category = ('Unknown_contig' if chrom not in represented else
                    'Non-repeat' if not values else values[0] if len(values)==1 else 'Multiple_families')
        result[site] = dict(repeat_category=category,repeat_families=';'.join(values),family_count=len(values),
            repeat_classes=';'.join(sorted(classes.get((chrom,position),set()))),
            repeat_names=';'.join(sorted(names.get((chrom,position),set()))),
            any_Alu=int('Alu' in values),any_repeat=int(bool(values)),
            repeat_annotation_known=int(chrom in represented))
    return result, dict(rmsk_rows=interval_rows,relevant_rmsk_rows=relevant_rows,
        represented_contigs=len(represented),rmsk_sha256=digest.hexdigest())


def venn_counts(reference, called):
    return dict(reference_only=len(reference-called),shared=len(reference&called),nn_only=len(called-reference),
                reference_total=len(reference),nn_total=len(called),union=len(reference|called))


def repeat_tables(groups, annotations):
    exclusive,inclusive,summary = [],[],[]
    for group,sites in groups.items():
        counts = Counter(annotations[site]['repeat_category'] for site in sites)
        total = len(sites)
        known = sum(annotations[site]['repeat_annotation_known'] for site in sites)
        alu = sum(annotations[site]['any_Alu'] for site in sites)
        repeats = sum(annotations[site]['any_repeat'] for site in sites)
        for category,count in sorted(counts.items()):
            exclusive.append(dict(group=group,repeat_category=category,sites=count,total_sites=total,
                                  percent_all_sites=100*count/total if total else None))
        family_counts = Counter()
        for site in sites:
            values = annotations[site]['repeat_families']
            family_counts.update(values.split(';') if values else [])
        for family,count in sorted(family_counts.items()):
            inclusive.append(dict(group=group,repeat_family=family,sites=count,total_sites=total,
                                  percent_all_sites=100*count/total if total else None))
        summary.append(dict(group=group,total_sites=total,known_annotation_sites=known,
            unknown_annotation_sites=total-known,any_repeat_sites=repeats,any_Alu_sites=alu,
            non_repeat_sites=counts['Non-repeat'],multiple_family_sites=counts['Multiple_families'],
            repeat_percent_all=100*repeats/total if total else None,
            Alu_percent_all=100*alu/total if total else None,Alu_percent_known=100*alu/known if known else None))
        if sum(counts.values()) != total:
            raise ValueError('Repeat categories do not partition unique sites')
    return exclusive,inclusive,summary


def plot_family_rows(exclusive, summary, top_n=8):
    groups = ['Slic-seq_all','Slic-seq_DRS_covered','Dorado_all','NN_all']
    special = ['Other_repeat','Multiple_families','Non-repeat','Unknown_contig']
    combined = Counter()
    for row in exclusive:
        if row['group'] in ('Slic-seq_all','NN_all') and row['repeat_category'] not in special:
            combined[row['repeat_category']] += row['sites']
    categories = [name for name,count in combined.most_common(top_n)] + special
    result = []
    totals = {row['group']:row['total_sites'] for row in summary}
    for group in groups:
        counts = Counter()
        for row in exclusive:
            if row['group']==group:
                category = row['repeat_category'] if row['repeat_category'] in categories else 'Other_repeat'
                counts[category] += row['sites']
        for category in categories:
            result.append(dict(group=group,plot_category=category,sites=counts[category],total_sites=totals[group],
                               percent_all_sites=100*counts[category]/totals[group] if totals[group] else None))
    return result


def run(config):
    output,figures = config['output'],config['figures']
    if output.exists() or figures.exists():
        raise FileExistsError('Choose new output and figure directories')
    output.mkdir(parents=True)
    marker=output/'INCOMPLETE.txt'
    marker.write_text('HeLa inosine comparison is incomplete.\n')
    inputs={name:file_info(config[name]) for name in PATHS if name not in ('output','figures')}
    records,nn,final=read_predictions(config['predictions'],config['sample'])
    dorado=set(records)
    slic=load_bed_sites(config['slic'])
    depths,minimum=checked_slic_coverage(config,slic)
    covered=set(depths)
    print(f'Annotating repeats for {len(dorado|slic):,} unique sites',flush=True)
    annotations,audit=annotate_repeat_families(dorado|slic,config['rmsk'],config['genome_fai'])
    groups={'Slic-seq_all':slic,'Slic-seq_DRS_covered':covered,'Dorado_all':dorado,'NN_all':nn,
            'NN_shared_Slic':nn&slic,'NN_only':nn-slic,'Slic-seq_only':slic-nn}
    exclusive,inclusive,repeat_summary=repeat_tables(groups,annotations)
    plot_rows=plot_family_rows(exclusive,repeat_summary)
    overlaps=[dict(scope='full_catalogue',**venn_counts(slic,nn)),
              dict(scope='DRS_depth_eligible',**venn_counts(covered,nn)),
              dict(scope='Dorado_before_NN',**venn_counts(slic,dorado))]
    transitions=[dict(sample=config['sample'],dorado_candidates=len(dorado),nn_retained=len(nn),
                      nn_removed=len(dorado-nn),slic_catalogue=len(slic),slic_drs_depth_eligible=len(covered),
                      slic_overlap_dorado=len(slic&dorado),slic_overlap_nn=len(slic&nn),
                      covered_slic_recovery_nn_pct=100*len(covered&nn)/len(covered),
                      final_kept_text_only=len(final),final_added_beyond_nn_text_only=len(final-nn))]
    membership=[]
    for chrom,start,strand in sorted(dorado|slic):
        site=chrom,start,strand
        row=dict(chrom=chrom,start=start,end=start+1,strand=strand,Slic_seq=int(site in slic),
            Slic_seq_DRS_covered=int(site in covered),Dorado=int(site in dorado),NN=int(site in nn),
            **annotations[site],**records.get(site,{}))
        row['slic_drs_depth']=depths.get(site)
        membership.append(row)
    fields=['chrom','start','end','strand','Slic_seq','Slic_seq_DRS_covered','Dorado','NN',
            'repeat_category','repeat_families','family_count','repeat_classes','repeat_names','any_Alu',
            'any_repeat','repeat_annotation_known','flag_nn','final_keep','flag_db','flag_short_read',
            'score_inosine','total_reads','af','slic_drs_depth']
    tables={'inosine_membership':(membership,fields),'inosine_overlap':(overlaps,list(overlaps[0])),
        'inosine_repeat_summary':(repeat_summary,list(repeat_summary[0])),
        'inosine_family_exclusive':(exclusive,list(exclusive[0])),
        'inosine_family_inclusive':(inclusive,list(inclusive[0])),
        'inosine_family_plot':(plot_rows,list(plot_rows[0])),
        'inosine_transitions':(transitions,list(transitions[0]))}
    sheets=[]
    for name,(rows,headers) in tables.items():
        filename=name+'.tsv'
        write_tsv(output/filename,rows,headers)
        numeric = [key for key in headers if key not in ('group','repeat_category','repeat_family','plot_category','scope','sample',
                   'chrom','strand','repeat_families','repeat_classes','repeat_names')]
        floats = [key for key in numeric if key in ('af','score_inosine') or key.endswith('_pct')
                  or key.startswith('percent_') or '_percent_' in key]
        sheets.append(dict(name=name,path=filename,headers=headers,rows=len(rows),
                           integer_columns=[key for key in numeric if key not in floats],float_columns=floats))
    from analysis.hela_inosine.plots import draw_figures
    figure_report=draw_figures(overlaps,plot_rows,repeat_summary,figures)
    for name,before in inputs.items():
        if file_info(config[name]) != before:
            raise ValueError('Input changed during analysis: '+name)
    sources={name:dict(info,sha256=audit['rmsk_sha256'] if name=='rmsk' else sha256(config[name]))
             for name,info in inputs.items()}
    run=dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),sample=config['sample'],
        assembly=config['assembly'],reference='Slic-seq only; BACS excluded',sources=sources,
        repeat_definition='UCSC repFamily; any half-open genomic interval overlap, irrespective of repeat strand',
        repeat_denominator='unique sites including Non-repeat and Unknown_contig; all groups sum to100%',
        multi_family_policy='all families retained in per-site audit; Multiple_families is one mutually exclusive category; inclusive-family fractions may sum above100%',
        unknown_policy='genome-valid contigs absent from rmsk are Unknown_contig, never Non-repeat',
        minimum_drs_coverage=minimum,NN_definition='inosine softmax argmax before reference rescue',
        final_rescue_policy='text/count audit only, no final-rescue Venn or repeat-composition plot',
        interpretation='descriptive reference overlap and family composition, not independent precision/sensitivity or repeat-matched enrichment',
        audit=audit,overlaps=overlaps,transitions=transitions,repeat_summary=repeat_summary,
        figures=figure_report,sheets=sheets)
    (output/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    (output/'excel_sheets.json').write_text(json.dumps(sheets,indent=2)+'\n')
    notes=make_notes(run)
    (output/'Results_inosine_en.txt').write_text(notes)
    marker.unlink()
    print(json.dumps(dict(overlaps=overlaps,repeat_summary=repeat_summary,transitions=transitions),indent=2),flush=True)
    return run


def make_notes(run):
    count=run['transitions'][0]
    summaries={row['group']:row for row in run['repeat_summary']}
    n=summaries['NN_all'];s=summaries['Slic-seq_all'];c=summaries['Slic-seq_DRS_covered'];d=summaries['Dorado_all']
    return f'''HeLa inosine: descriptive comparison before reference rescue

The inosine NN retained {count['nn_retained']:,} of {count['dorado_candidates']:,} original Dorado/modkit candidates. It overlapped {count['slic_overlap_nn']:,} of {count['slic_catalogue']:,} Slic-seq catalogue sites. Slic-seq alone and NN alone are exact genomic-coordinate/strand sets; BACS is not included. Of the Slic-seq catalogue, {count['slic_drs_depth_eligible']:,} sites had DRS total depth at least10. The NN recovered {count['covered_slic_recovery_nn_pct']:.2f}% of these covered positive-reference sites. Before NN filtering, {count['slic_overlap_dorado']:,} Slic-seq sites were already present among the original candidates. Thus misses before candidate generation are separated from additional NN losses. Reference non-overlap does not establish a false positive.

RepeatMasker repFamily assignments were strand-independent, half-open genomic overlaps. Any-Alu fractions among all sites were {s['Alu_percent_all']:.2f}% for the full Slic-seq catalogue, {c['Alu_percent_all']:.2f}% for DRS-covered Slic-seq sites, {d['Alu_percent_all']:.2f}% for original Dorado candidates and {n['Alu_percent_all']:.2f}% for NN-retained sites. Every site contributes once to the mutually exclusive composition. Sites overlapping several families form Multiple_families; all family identities are retained separately. Inclusive family counts can overlap and need not sum to100%. Genome-valid contigs absent from rmsk remain Unknown_contig, and are included explicitly in the all-site denominator. Repeat composition is descriptive and is not independently validated specificity or matched-background enrichment; the NN itself uses repeat-mask input.

Final reference-assisted reporting would return {count['final_kept_text_only']:,} inosine sites, adding {count['final_added_beyond_nn_text_only']:,} beyond the NN. This is text-only; all Venn and composition figures use raw NN decisions. GLORI is not an inosine comparator. The source is Slic-seq (not silkSeq) and is not merged with BACS. All table/figure counts and per-site family assignments are provided as TSVs with an Excel import manifest.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text())
    for name in PATHS:config[name]=(args.config.resolve().parent/config[name]).resolve()
    if config['sample']!='HeLa_WT' or config['assembly']!='GRCh38':
        raise ValueError('This analysis explicitly uses HeLa WT / GRCh38')
    run(config)


if __name__=='__main__':
    main()
