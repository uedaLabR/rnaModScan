"""Strand-oriented non-DRACH sequence contexts and auditable WebLogo inputs."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.io.artifacts import file_info
from utils.references.catalog import sha256

GROUPS = ('GLORI_all', 'NN_all', 'GLORI_only', 'NN_only', 'shared')
BASES = 'ACGT'
ALLOWED = ('AGT', 'AG', 'A', 'C', 'ACT')
COMPLEMENT = str.maketrans('ACGTN', 'TGCAN')


def read_tsv(path):
    with Path(path).open(newline='') as handle:
        yield from csv.DictReader(handle, delimiter='\t')


def write_tsv(path, rows, fields):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def sequence_context(fasta, chrom, center, strand, flank=5):
    """Fetch just the requested genome window and orient it into RNA5-prime order."""
    if strand not in ('+', '-'):
        raise ValueError('Explicit strand required')
    length = fasta.get_reference_length(chrom)
    if not 0 <= center < length:
        raise ValueError('Position outside reference contig')
    lo, hi = center - flank, center + flank + 1
    left, right = max(0, -lo), max(0, hi - length)
    sequence = 'N' * left + fasta.fetch(chrom, max(0,lo), min(length,hi)).upper() + 'N' * right
    sequence = ''.join(base if base in 'ACGTN' else 'N' for base in sequence)
    if strand == '-':
        sequence = sequence.translate(COMPLEMENT)[::-1]
    if len(sequence) != 2 * flank + 1:
        raise ValueError('Reference window length mismatch')
    return sequence, left + right


def distance_to_drach(sequence):
    """Minimum Hamming distance to any allowed DRACH5-mer; N is missing, not a mismatch."""
    if len(sequence) != 5 or any(base not in BASES for base in sequence):
        return None
    return sum(base not in allowed for base,allowed in zip(sequence,ALLOWED))


def member_groups(glori, nn):
    groups = []
    if glori:
        groups.append('GLORI_all')
    if nn:
        groups.append('NN_all')
    groups.append('shared' if glori and nn else 'GLORI_only' if glori else 'NN_only')
    return groups


def jensen_shannon(left, right):
    """Base2 Jensen-Shannon divergence of two normalized base distributions."""
    middle = [(a+b)/2 for a,b in zip(left,right)]
    def divergence(values):
        return sum(p*math.log2(p/m) for p,m in zip(values,middle) if p)
    return (divergence(left)+divergence(right))/2


def site_records(config):
    import pysam
    seen = set()
    with pysam.FastaFile(str(config['fasta'])) as fasta:
        for source in read_tsv(config['membership']):
            chrom, center, strand = source['chrom'], int(source['start']), source['strand']
            key = chrom, center, strand
            if key in seen:
                raise ValueError('Duplicate source position')
            seen.add(key)
            glori, nn = int(source['GLORI']), int(source['rnaModScan'])
            if glori not in (0,1) or nn not in (0,1) or not (glori or nn):
                raise ValueError('Invalid set-membership flags')
            sequence, padding = sequence_context(fasta, chrom, center, strand, config['flank'])
            middle = config['flank']
            kmer = sequence[middle-2:middle+3]
            if source['motif'] != kmer or distance_to_drach(kmer) == 0:
                raise ValueError('Source/genome motif mismatch or DRACH site in non-DRACH input')
            groups = member_groups(glori,nn)
            complete = all(base in BASES for base in sequence)
            yield dict(chrom=chrom,start=center,end=center+1,strand=strand,
                subset=groups[-1],GLORI=glori,NN=nn,reference_A=int(sequence[middle]=='A'),
                dna_11nt=sequence,rna_11nt=sequence.replace('T','U'),central_5mer=kmer,
                DRACH_distance=distance_to_drach(kmer),padding_bases=padding,
                ambiguous_bases=sum(base not in BASES for base in sequence),
                weblogo_complete_11nt=int(complete),dorado_depth=source['dorado_depth'],
                nn_score=source['rnamodscan_m6a_score'])


def summarize(records, flank):
    groups = {name:[] for name in GROUPS}
    for row in records:
        for name in member_groups(row['GLORI'],row['NN']):
            groups[name].append(row)
    summary, positions, kmers, distances, mismatches = [], [], [], [], []
    for name,rows in groups.items():
        valid = [row for row in rows if row['weblogo_complete_11nt']]
        known = [row['DRACH_distance'] for row in rows if row['DRACH_distance'] is not None]
        summary.append(dict(group=name,sites=len(rows),complete_11nt_sites=len(valid),
            ambiguous_11nt_sites=len(rows)-len(valid),padding_sites=sum(row['padding_bases']>0 for row in rows),
            reference_A_sites=sum(row['reference_A'] for row in rows),
            central_5mer_known_sites=len(known),one_change_from_DRACH=sum(value==1 for value in known),
            one_change_from_DRACH_pct=100*sum(value==1 for value in known)/len(known) if known else None,
            mean_DRACH_distance=sum(known)/len(known) if known else None))
        for scope,selected in [('all_sites',rows),('complete_11nt',valid)]:
            for index in range(2*flank+1):
                counts = Counter(row['dna_11nt'][index] for row in selected)
                known_count = sum(counts[base] for base in BASES)
                for base in 'ACGTN':
                    positions.append(dict(group=name,scope=scope,position=index-flank,base_DNA=base,
                        base_RNA=base.replace('T','U'),count=counts[base],total_sites=len(selected),
                        known_bases=known_count,
                        fraction_of_sites=counts[base]/len(selected) if selected else None,
                        fraction_among_ACGT=counts[base]/known_count if base in BASES and known_count else None))
        for kmer,count in sorted(Counter(row['central_5mer'] for row in rows).items(),key=lambda item:(-item[1],item[0])):
            kmers.append(dict(group=name,central_5mer_DNA=kmer,central_5mer_RNA=kmer.replace('T','U'),count=count,
                fraction=count/len(rows),DRACH_distance=distance_to_drach(kmer)))
        distance_counts = Counter(known)
        for distance in range(6):
            distances.append(dict(group=name,DRACH_distance=distance,count=distance_counts[distance],
                known_5mer_sites=len(known),fraction=distance_counts[distance]/len(known) if known else None))
        for index,allowed in enumerate(ALLOWED):
            count = sum(row['central_5mer'][index] not in allowed for row in rows
                        if row['DRACH_distance'] is not None)
            mismatches.append(dict(group=name,position=index-2,DRACH_symbol='DRACH'[index],
                allowed_DNA=allowed,allowed_RNA=allowed.replace('T','U'),mismatch_sites=count,
                known_5mer_sites=len(known),mismatch_fraction=count/len(known) if known else None))
    return groups,summary,positions,kmers,distances,mismatches


def frequency_similarity(positions):
    lookup = {(row['group'],row['position'],row['base_DNA']):row['fraction_among_ACGT']
              for row in positions if row['scope']=='complete_11nt' and row['base_DNA'] in BASES}
    rows=[]
    for a,b in combinations(GROUPS,2):
        values=[]
        for position in range(-5,6):
            left = [lookup[a,position,base] for base in BASES]
            right = [lookup[b,position,base] for base in BASES]
            if any(value is None for value in left+right):
                continue
            value=jensen_shannon(left,right)
            values.append((position,value))
            rows.append(dict(group_a=a,group_b=b,scope='complete_11nt',position=str(position),
                JS_divergence_bits=value,L1_distance=sum(abs(x-y) for x,y in zip(left,right))))
        for name,selected in [('mean_all_11_positions',values),
                              ('mean_flanks_excluding_center',[v for v in values if v[0]!=0]),
                              ('mean_central_5_excluding_center',[v for v in values if -2<=v[0]<=2 and v[0]!=0])]:
            rows.append(dict(group_a=a,group_b=b,scope='complete_11nt',position=name,
                JS_divergence_bits=sum(v[1] for v in selected)/len(selected) if selected else None,L1_distance=None))
    return rows


def write_weblogo(groups, output):
    directory=output/'weblogo_inputs'
    directory.mkdir()
    manifest=[]
    for name,rows in groups.items():
        valid=[row for row in rows if row['weblogo_complete_11nt']]
        for alphabet,column in [('RNA','rna_11nt'),('DNA','dna_11nt')]:
            path=directory/(name+'_'+alphabet+'.fa')
            with path.open('w') as handle:
                for row in valid:
                    handle.write(f">{row['chrom']}:{row['start']}:{row['strand']}|BED0|{row['subset']}\n{row[column]}\n")
            manifest.append(dict(group=name,alphabet=alphabet,sites=len(rows),
                included_complete_11nt=len(valid),excluded_ambiguous_11nt=len(rows)-len(valid),
                path=str(path.relative_to(output)),sha256=sha256(path)))
    return manifest


def draw_figures(output, summary, positions, distances):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    output.mkdir(parents=True,exist_ok=False)
    labels={'GLORI_all':'GLORI: all non-DRACH','NN_all':'rnaModScan NN: all non-DRACH',
            'GLORI_only':'GLORI only','NN_only':'NN only','shared':'Shared'}
    lookup={(row['group'],row['position'],row['base_DNA']):row['fraction_among_ACGT']
            for row in positions if row['scope']=='complete_11nt' and row['base_DNA'] in BASES}
    totals={row['group']:row['complete_11nt_sites'] for row in summary}
    paths=[]
    with matplotlib.rc_context({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42}):
        fig,axes=plt.subplots(2,1,figsize=(9,7.8))
        for ax,group in zip(axes,('GLORI_all','NN_all')):
            matrix=np.array([[lookup[group,position,base] for position in range(-5,6)] for base in BASES])
            image=ax.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='Blues')
            ax.set(xticks=range(11),xticklabels=range(-5,6),yticks=range(4),yticklabels=list('ACGU'),
                   xlabel='Position relative to candidate A (RNA 5′ to 3′)')
            ax.set_title(f'{labels[group]} (complete 11-nt windows: {totals[group]:,})',fontsize=12)
            for y in range(4):
                for x in range(11):
                    ax.text(x,y,f'{100*matrix[y,x]:.0f}',ha='center',va='center',fontsize=9,
                            color='white' if matrix[y,x]>.55 else '#172B3A')
        fig.colorbar(image,ax=list(axes),label='Base frequency',fraction=.025,pad=.025)
        fig.subplots_adjust(left=.08,right=.87,top=.9,bottom=.12,hspace=.58)
        fig.suptitle('HeLa non-DRACH m6A: flanking sequence composition',fontsize=14,weight='bold')
        fig.text(.08,.035,'Numbers are percentages. Complete A/C/G/U windows only; raw sequences and exclusions are retained.\nDescriptive sequence similarity is not evidence of modification identity.',fontsize=8.5)
        for extension in ('svg','pdf','png'):
            path=output/('FigS_HeLa_non_DRACH_base_frequencies.'+extension)
            fig.savefig(path,dpi=240,facecolor='white');paths.append(path.name)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,5.8))
        x=np.arange(1,5);width=.15
        for index,group in enumerate(GROUPS):
            values=[next(row['fraction'] for row in distances if row['group']==group and row['DRACH_distance']==distance)
                    for distance in range(1,5)]
            ax.bar(x+(index-2)*width,[100*value for value in values],width,label=labels[group])
        ax.set(xticks=x,xlabel='Minimum substitutions in central 5-mer to a DRACH motif',
               ylabel='Sites (%)',ylim=(0,100),title='HeLa non-DRACH m6A: distance from DRACH')
        ax.legend(frameon=False,fontsize=9,ncol=2)
        ax.spines[['top','right']].set_visible(False)
        fig.subplots_adjust(bottom=.22,left=.09,right=.98,top=.9)
        fig.text(.09,.055,'Genomic strand-oriented 5-mers; central A is retained. Groups share sites and are not independent.\nA distance of 1 indicates a one-base departure from the DRACH consensus; it does not validate m6A.',fontsize=8.5)
        for extension in ('svg','pdf','png'):
            path=output/('FigS_HeLa_non_DRACH_distance.'+extension)
            fig.savefig(path,dpi=240,facecolor='white');paths.append(path.name)
        plt.close(fig)
    return paths


def excel_manifest(output,tables):
    integer={'start','end','GLORI','NN','reference_A','DRACH_distance','padding_bases','ambiguous_bases',
             'weblogo_complete_11nt','dorado_depth','sites','complete_11nt_sites','ambiguous_11nt_sites',
             'padding_sites','reference_A_sites','central_5mer_known_sites','one_change_from_DRACH',
             'position','count','total_sites','known_bases','known_5mer_sites','mismatch_sites',
             'included_complete_11nt','excluded_ambiguous_11nt'}
    floating={'nn_score','one_change_from_DRACH_pct','mean_DRACH_distance','fraction_of_sites',
              'fraction_among_ACGT','fraction','mismatch_fraction','JS_divergence_bits','L1_distance'}
    descriptors=[]
    for name,(filename,rows) in tables.items():
        fields=list(rows[0])
        if name=='Motif_similarity':
            numeric_integer=sorted((set(fields)&integer)-{'position'})
        else:
            numeric_integer=sorted(set(fields)&integer)
        descriptors.append(dict(name=name,path=filename,headers=fields,
            integer_columns=numeric_integer,float_columns=sorted(set(fields)&floating),rows=len(rows)))
    (output/'excel_sheets.json').write_text(json.dumps(descriptors,indent=2)+'\n')
    return descriptors


def run(config):
    output=Path(config['output'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'INCOMPLETE.txt').write_text('Motif analysis is incomplete.\n')
    print('Extracting strand-oriented 11-nt contexts from non-DRACH membership sites',flush=True)
    records=list(site_records(config))
    groups,summary,positions,kmers,distances,mismatches=summarize(records,config['flank'])
    similarity=frequency_similarity(positions)
    manifests=write_weblogo(groups,output)
    tables=dict(Motif_site_sequences=('site_sequences.tsv',records),Motif_group_summary=('group_summary.tsv',summary),
                Motif_position_frequencies=('position_frequencies.tsv',positions),Motif_central_5mers=('central_5mers.tsv',kmers),
                Motif_DRACH_distance=('drach_distance.tsv',distances),Motif_DRACH_mismatches=('drach_mismatches.tsv',mismatches),
                Motif_similarity=('frequency_similarity.tsv',similarity),Motif_WebLogo_files=('weblogo_manifest.tsv',manifests))
    for filename,rows in tables.values():
        write_tsv(output/filename,rows,list(rows[0]))
    descriptors=excel_manifest(output,tables)
    figures=draw_figures(config['figures'],summary,positions,distances)
    report=dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),assembly=config['assembly'],
        window='candidate ±5nt, 11nt, RNA5-prime orientation; central5mer for DRACH distance',
        source='HeLa raw NN m6A argmax before DB/short-read/DRACH rescue',
        inputs=dict(membership=dict(file_info(config['membership']),sha256=sha256(config['membership'])),
                    fasta=file_info(config['fasta']),fasta_index_sha256=sha256(str(config['fasta'])+'.fai')),
        group_summary=summary,figures=figures,excel_sheets=descriptors,
        limitations=['Descriptive sequence composition; no statistical significance or m6A validity inferred',
                     'All-GLORI and all-NN sets overlap; only-comparison groups are locus-disjoint',
                     'No gene/GC/depth/mappability matched null distribution constructed',
                     'Known-positive and NN-passed inputs were selected as non-DRACH, so none can have distance0 by definition',
                     'FASTA is genomic context, not exon-stitched transcript context',
                     'Ambiguous/padded 11nt windows remain in raw audit; WebLogo FASTA uses complete11nt windows only'])
    (output/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'INCOMPLETE.txt').unlink()
    by_name={row['group']:row for row in summary}
    notes=['HeLa non-DRACH m6A motif analysis (completed).',
           'WebLogo inputs: weblogo_inputs/*_RNA.fa (A/C/G/U) or *_DNA.fa (A/C/G/T).',
           'Set sequence type to RNA/DNA respectively; width11; candidate A is column6 (one-based).',
           'Use center position0 and flank5 each side; no sequence is duplicated within a group.',
           'Complete11nt windows only are in FASTA; every original site and exclusion is retained in site_sequences.tsv.',
           'WebLogo may draw the provided sequences, but no de novo motif enrichment or chemical validation is claimed.']
    for name in GROUPS:
        row=by_name[name]
        notes.append(f"{name}: sites{row['sites']:,}; complete11nt{row['complete_11nt_sites']:,}; distance1 {row['one_change_from_DRACH_pct']:.2f}%; mean distance{row['mean_DRACH_distance']:.4f}.")
    (output/'README.txt').write_text('\n'.join(notes)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text())
    for name in ('membership','fasta','output','figures'):
        config[name]=(args.config.resolve().parent/config[name]).resolve()
    if config['assembly']!='GRCh38' or config['flank']!=5:
        raise ValueError('This analysis uses GRCh38 candidate±5nt')
    run(config)


if __name__=='__main__':
    main()
