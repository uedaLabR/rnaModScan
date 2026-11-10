"""Region composition and within-region Alu values with explicit denominators."""
from pathlib import Path


LABELS={'five_prime_UTR':"5′UTR",'CDS':'CDS','intron':'Intron','three_prime_UTR':"3′UTR",
        'Ambiguous_regions':'Multiple regions','noncoding_exon':'Noncoding exon','Other_regions':'Other/outside/unknown'}
GROUPS={'Slic-seq_all':'Slic-seq (all)','Slic-seq_DRS_covered':'Slic-seq (DRS-covered)',
        'Dorado_all':'Dorado/modkit','NN_all':'rnaModScan NN only'}
COLORS=['#E9B44C','#4C96B7','#B7B7B7','#65A384','#A283AC','#E09E8A','#E4E8EB']


def save(fig,output,stem):
    files=[]
    for extension in ('svg','pdf','png'):
        path=output/(stem+'.'+extension)
        fig.savefig(path,dpi=240,facecolor='white')
        files.append(path.name)
    return dict(stem=stem,files=files)


def draw(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    lookup={(row['group'],row['region']):row for row in rows}
    records=[]
    with matplotlib.rc_context({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','pdf.fonttype':42}):
        fig,ax=plt.subplots(figsize=(11.5,7.2))
        groups=list(GROUPS);regions=list(LABELS)
        starts=np.zeros(len(groups))
        for region,color in zip(regions,COLORS):
            values=[lookup[group,region]['percent_all_sites'] for group in groups]
            ax.barh(range(len(groups)),values,left=starts,color=color,label=LABELS[region],height=.62,
                    edgecolor='white',linewidth=.5)
            starts+=values
        labels=[GROUPS[group]+f"\nn = {lookup[group,regions[0]]['group_total_sites']:,}" for group in groups]
        ax.set_yticks(range(len(groups)),labels);ax.invert_yaxis()
        ax.set(xlim=(0,100),xlabel='Fraction of all unique sites (%)')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='upper center',bbox_to_anchor=(.48,-.16),ncol=4,frameon=False,fontsize=10)
        fig.suptitle('HeLa inosine: genomic region composition',fontsize=17,weight='bold')
        fig.subplots_adjust(left=.24,right=.97,top=.87,bottom=.31)
        fig.text(.055,.065,'Same-strand GENCODE v44 isoforms; conflicting region labels form Multiple regions.\n'
                 'Every site contributes once; noncoding and other/outside/unknown sites remain in the denominator.',fontsize=9,color='#425666')
        records.append(save(fig,output,'FigS_HeLa_inosine_region_composition'));plt.close(fig)

        fig,ax=plt.subplots(figsize=(11.5,7.2))
        displayed=[region for region in LABELS if region!='Other_regions']
        y=np.arange(len(displayed));height=.35
        for group,offset,color in [('Slic-seq_all',-height/2,'#478DB3'),('NN_all',height/2,'#5DA584')]:
            values=[lookup[group,region]['Alu_percent_in_region'] for region in displayed]
            ax.barh(y+offset,[value if value is not None else 0 for value in values],height=height,
                    label=GROUPS[group],color=color)
            for index,region in enumerate(displayed):
                row=lookup[group,region]
                value=row['Alu_percent_in_region']
                text=(f"{value:.1f}% (n={row['sites']:,})" if value is not None else 'n=0')
                ax.text(101,index+offset,text,va='center',fontsize=9,color=color)
        ax.set_yticks(y,[LABELS[region] for region in displayed]);ax.invert_yaxis()
        ax.set(xlim=(0,100),xlabel='Any-Alu sites within each region (%)')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='upper center',bbox_to_anchor=(.47,-.12),ncol=2,frameon=False)
        fig.suptitle('HeLa inosine: Alu association within genomic regions',fontsize=16,weight='bold')
        fig.subplots_adjust(left=.19,right=.77,top=.87,bottom=.24)
        fig.text(.055,.065,'n is the number of unique sites in that region; multiple-region sites stay separate.\n'
                 'Any-Alu includes multi-family Alu overlaps. Composition differences do not establish assay bias.',fontsize=9,color='#425666')
        records.append(save(fig,output,'FigS_HeLa_inosine_Alu_by_region'));plt.close(fig)
    return records
