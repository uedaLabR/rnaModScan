"""Stranded isoform-aware regions and region-stratified Alu proportions."""
import argparse
from collections import Counter
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.features.position import annotation_attributes, _merge, _contains
from utils.io.artifacts import file_info
from utils.references.catalog import sha256

REGIONS = ('five_prime_UTR','CDS','intron','three_prime_UTR','noncoding_exon','other_exon',
           'outside_same_strand_transcript','unknown_annotation')
EXCLUSIVE = REGIONS[:4] + ('Ambiguous_regions',) + REGIONS[4:]
FEATURES = {'transcript','exon','CDS','stop_codon','UTR','five_prime_utr','three_prime_utr'}


@dataclass
class Transcript:
    chrom: str
    strand: str
    gene_id: str
    transcript_id: str
    exons: list = field(default_factory=list)
    coding: list = field(default_factory=list)
    utr5: list = field(default_factory=list)
    utr3: list = field(default_factory=list)
    utr: list = field(default_factory=list)

    def finish(self):
        self.exons = _merge(self.exons)
        self.coding = _merge(self.coding)
        self.utr5,self.utr3,self.utr = _merge(self.utr5),_merge(self.utr3),_merge(self.utr)
        self.start,self.end = self.exons[0][0],self.exons[-1][1]
        self.coding_start = self.coding[0][0] if self.coding else None
        self.coding_end = self.coding[-1][1] if self.coding else None

    def region(self, position):
        if not self.start <= position < self.end:
            raise ValueError('Position outside transcript')
        if not _contains(self.exons,position):
            return 'intron'
        if _contains(self.coding,position):
            return 'CDS'
        # A noncoding transcript's exons are not UTRs.
        if not self.coding:
            return 'noncoding_exon'
        if _contains(self.utr5,position):
            return 'five_prime_UTR'
        if _contains(self.utr3,position):
            return 'three_prime_UTR'
        # GENCODE generic UTR records and partially typed terminal exons use CDS bounds.
        if position < self.coding_start:
            return 'five_prime_UTR' if self.strand=='+' else 'three_prime_UTR'
        if position >= self.coding_end:
            return 'three_prime_UTR' if self.strand=='+' else 'five_prime_UTR'
        return 'other_exon'


class RegionAnnotation:
    """All same-strand isoforms contribute; no hidden representative or priority."""
    def __init__(self, annotation):
        transcripts = {}
        digest = hashlib.sha256()
        self.bins,self.chromosomes = {},set()
        self.bin_size = 100000
        with Path(annotation).open('rb') as handle:
            for raw in handle:
                digest.update(raw)
                if not raw.strip() or raw.startswith(b'#'):
                    continue
                columns = raw.decode().rstrip('\r\n').split('\t')
                if len(columns)!=9:
                    raise ValueError('Expected GENCODE nine-column annotation')
                chrom,_,feature,start,end,_,strand,_,text = columns
                feature = feature.lower() if feature in ('five_prime_UTR','three_prime_UTR') else feature
                if feature not in FEATURES:
                    continue
                attrs = annotation_attributes(text)
                name,gene = attrs.get('transcript_id'),attrs.get('gene_id')
                if name is None or gene is None or strand not in ('+','-'):
                    raise ValueError('Transcript identifiers/strand missing')
                lo,hi=int(start)-1,int(end)
                if not 0<=lo<hi:
                    raise ValueError('Invalid annotation coordinates')
                tx=transcripts.setdefault(name,Transcript(chrom,strand,gene,name))
                if (chrom,strand,gene)!=(tx.chrom,tx.strand,tx.gene_id):
                    raise ValueError('Conflicting transcript identity')
                interval=(lo,hi)
                if feature=='exon':tx.exons.append(interval)
                elif feature in ('CDS','stop_codon'):tx.coding.append(interval)
                elif feature=='five_prime_utr':tx.utr5.append(interval)
                elif feature=='three_prime_utr':tx.utr3.append(interval)
                elif feature=='UTR':tx.utr.append(interval)
        self.transcript_count=0
        for tx in transcripts.values():
            if not tx.exons:
                continue
            tx.finish()
            self.transcript_count+=1
            self.chromosomes.add(tx.chrom)
            for index in range(tx.start//self.bin_size,(tx.end-1)//self.bin_size+1):
                self.bins.setdefault((tx.chrom,tx.strand,index),[]).append(tx)
        self.sha256=digest.hexdigest()
        if not self.transcript_count:
            raise ValueError('No annotated transcripts')

    def annotate(self, chrom, position, strand):
        if strand not in ('+','-') or position<0:
            raise ValueError('Explicit strand and BED0 center required')
        txs=[tx for tx in self.bins.get((chrom,strand,position//self.bin_size),())
             if tx.start<=position<tx.end]
        categories={tx.region(position) for tx in txs}
        if not txs:
            categories.add('outside_same_strand_transcript' if chrom in self.chromosomes else 'unknown_annotation')
        primary=next(iter(categories)) if len(categories)==1 else 'Ambiguous_regions'
        return dict(region_exclusive=primary,region_categories=';'.join(sorted(categories)),
            region_category_count=len(categories),transcript_count=len(txs),
            gene_ids=';'.join(sorted({tx.gene_id for tx in txs})),
            transcript_ids=';'.join(sorted(tx.transcript_id for tx in txs)),
            transcript_region_assignments=';'.join(sorted(
                tx.transcript_id+'='+tx.region(position) for tx in txs)),
            **{'has_'+region:int(region in categories) for region in REGIONS})


def read_tsv(path):
    with Path(path).open(newline='') as handle:
        yield from csv.DictReader(handle,delimiter='\t')


def write_tsv(path, rows, fields):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fields,delimiter='\t',lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def memberships(row):
    groups=[]
    if row['Slic_seq']:groups.append('Slic-seq_all')
    if row['Slic_seq_DRS_covered']:groups.append('Slic-seq_DRS_covered')
    if row['Dorado']:groups.append('Dorado_all')
    if row['NN']:groups.append('NN_all')
    if row['NN'] and row['Slic_seq']:groups.append('NN_shared_Slic')
    elif row['NN']:groups.append('NN_only')
    if row['Slic_seq'] and not row['NN']:groups.append('Slic-seq_only')
    return groups


def summarize(sites):
    totals,alu_totals,counts,alu_counts=Counter(),Counter(),Counter(),Counter()
    for row in sites:
        for group in memberships(row):
            totals[group]+=1;alu_totals[group]+=row['any_Alu']
            regions={'exclusive':[row['region_exclusive']],
                     'inclusive':row['region_categories'].split(';')}
            for scheme,categories in regions.items():
                for region in categories:
                    key=(scheme,group,region)
                    counts[key]+=1;alu_counts[key]+=row['any_Alu']
    result=[]
    for scheme in ('exclusive','inclusive'):
        categories=EXCLUSIVE if scheme=='exclusive' else REGIONS
        for group,total in totals.items():
            for region in categories:
                key=(scheme,group,region);count,alu=counts[key],alu_counts[key]
                result.append(dict(scheme=scheme,group=group,region=region,sites=count,
                    group_total_sites=total,site_percent_all=100*count/total,
                    any_Alu_sites=alu,Alu_percent_in_region=100*alu/count if count else None,
                    group_total_Alu_sites=alu_totals[group],
                    region_percent_among_Alu=100*alu/alu_totals[group] if alu_totals[group] else None))
    for group,total in totals.items():
        if sum(counts['exclusive',group,region] for region in EXCLUSIVE)!=total:
            raise ValueError('Exclusive categories do not partition all sites')
    return result


def plot_rows(summary):
    groups=['Slic-seq_all','Slic-seq_DRS_covered','Dorado_all','NN_all']
    shown=REGIONS[:4]+('Ambiguous_regions','noncoding_exon','Other_regions')
    rows=[]
    for group in groups:
        source=[row for row in summary if row['scheme']=='exclusive' and row['group']==group]
        for region in shown:
            selected=[row for row in source if (row['region']==region if region!='Other_regions'
                      else row['region'] in ('other_exon','outside_same_strand_transcript','unknown_annotation'))]
            count=sum(row['sites'] for row in selected);alu=sum(row['any_Alu_sites'] for row in selected)
            total=source[0]['group_total_sites']
            rows.append(dict(group=group,region=region,sites=count,group_total_sites=total,
                percent_all_sites=100*count/total,any_Alu_sites=alu,
                Alu_percent_in_region=100*alu/count if count else None))
    return rows


def descriptor(name,path,rows,integer_columns,float_columns):
    return dict(name=name,path=path,headers=list(rows[0]),rows=len(rows),
                integer_columns=integer_columns,float_columns=float_columns)


def run(config):
    output,figures=config['output'],config['figures']
    if output.exists() or figures.exists():
        raise FileExistsError('Choose fresh region output/figure directories')
    output.mkdir(parents=True)
    (output/'INCOMPLETE.txt').write_text('Region annotation incomplete.\n')
    print('Loading same-strand GENCODE transcript annotations',flush=True)
    annotation=RegionAnnotation(config['annotation'])
    records=[];seen=set()
    for source in read_tsv(config['membership']):
        key=source['chrom'],int(source['start']),source['strand']
        if key in seen:raise ValueError('Duplicate region source coordinate')
        seen.add(key)
        row=dict(chrom=key[0],start=key[1],end=key[1]+1,strand=key[2],
            **{name:int(source[name]) for name in ('Slic_seq','Slic_seq_DRS_covered','Dorado','NN','any_Alu','any_repeat')},
            repeat_category=source['repeat_category'],repeat_families=source['repeat_families'])
        row.update(annotation.annotate(*key));records.append(row)
    summary=summarize(records);plot=plot_rows(summary)
    patterns=Counter((row['region_exclusive'],row['region_categories']) for row in records)
    pattern_rows=[dict(region_exclusive=key[0],region_categories=key[1],sites=count)
                  for key,count in sorted(patterns.items())]
    tables=[('Region_site_annotations','region_sites.tsv',records,
        ['start','end','Slic_seq','Slic_seq_DRS_covered','Dorado','NN','any_Alu','any_repeat',
         'region_category_count','transcript_count',*['has_'+region for region in REGIONS]],[]),
        ('Region_summary','region_summary.tsv',summary,
         ['sites','group_total_sites','any_Alu_sites','group_total_Alu_sites'],
         ['site_percent_all','Alu_percent_in_region','region_percent_among_Alu']),
        ('Region_figure_values','region_plot.tsv',plot,['sites','group_total_sites','any_Alu_sites'],
         ['percent_all_sites','Alu_percent_in_region']),
        ('Region_ambiguity_patterns','region_patterns.tsv',pattern_rows,['sites'],[])]
    sheets=[]
    for name,path,rows,integers,floating in tables:
        write_tsv(output/path,rows,list(rows[0]));sheets.append(descriptor(name,path,rows,integers,floating))
    from analysis.hela_regions.plots import draw
    figure_files=draw(plot,figures)
    policies=dict(assembly='GRCh38',coordinate='BED0 exact chromosome/start/strand; GENCODE1-based closed converted to BED0',
        isoforms='all same-strand overlapping isoforms; distinct categories become Ambiguous_regions; no representative transcript or priority',
        UTR='explicit5/3UTR honored; coding-transcript terminal exons inferred relative to CDS+stop-codon bounds; negative strand reversed; noncoding exons are notUTR',
        denominators='exclusive categories partition ALL sites including ambiguous/other/unknown; inclusive category flags can sum above100%',
        Alu='existing any-Alu central genomic membership, including any multi-family Alu overlap',
        interpretation='descriptive source composition and within-region Alu association; does not establish assay bias or explain all disagreement causally',
        model='reporting only; no NN inputs, labels or prediction values changed')
    report=dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
        inputs=dict(membership=dict(file_info(config['membership']),sha256=sha256(config['membership'])),
                    annotation=dict(file_info(config['annotation']),sha256=annotation.sha256)),
        transcript_count=annotation.transcript_count,unique_sites=len(records),policies=policies,
        groups={row['group']:row['group_total_sites'] for row in summary},excel_sheets=sheets,figures=figure_files)
    (output/'excel_sheets.json').write_text(json.dumps(sheets,indent=2)+'\n')
    (output/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'INCOMPLETE.txt').unlink()
    print(json.dumps(dict(unique_sites=len(records),transcripts=annotation.transcript_count,groups=report['groups']),indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    args=parser.parse_args();config=yaml.safe_load(args.config.read_text())
    for name in ('membership','annotation','output','figures'):
        config[name]=(args.config.resolve().parent/config[name]).resolve()
    if config['assembly']!='GRCh38':raise ValueError('This report uses GRCh38')
    run(config)


if __name__=='__main__':main()
