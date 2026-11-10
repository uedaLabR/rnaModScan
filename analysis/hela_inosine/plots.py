"""Exact-count Venn diagrams and unique-site repeat-family composition."""
from pathlib import Path


def draw_venn(record, path, covered):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig,ax=plt.subplots(figsize=(7.8,5.5))
    fig.subplots_adjust(bottom=.25,top=.80,left=.08,right=.92)
    for center,color in [((-.65,0),'#478DB3'),((.65,0),'#5DA584')]:
        ax.add_patch(Circle(center,1.1,facecolor=color,alpha=.28,edgecolor='none'))
        ax.add_patch(Circle(center,1.1,facecolor='none',edgecolor=color,linewidth=1.7))
    ax.text(-1.1,1.4,f"Slic-seq\nn = {record['reference_total']:,}",ha='center',va='center',
            color='#478DB3',weight='bold',fontsize=12)
    ax.text(1.1,1.4,f"rnaModScan NN only\nn = {record['nn_total']:,}",ha='center',va='center',
            color='#5DA584',weight='bold',fontsize=12)
    for x,key in [(-1.1,'reference_only'),(0,'shared'),(1.1,'nn_only')]:
        ax.text(x,0,f"{record[key]:,}",ha='center',va='center',weight='bold',fontsize=15,color='#172B3A')
    ax.set(xlim=(-2.1,2.1),ylim=(-1.35,1.75),aspect='equal')
    ax.axis('off')
    title='DRS-covered Slic-seq (depth >= 10)' if covered else 'Full Slic-seq catalogue'
    fig.suptitle('HeLa WT: inosine site overlap\n'+title,fontsize=15,weight='bold')
    notes=['Exact GRCh38 coordinate and strand; raw inosine softmax argmax before reference rescue.',
           'Slic-seq only; BACS excluded. Circle areas are schematic, not proportional.',
           'Reference non-overlap does not establish false positives or false negatives.']
    fig.text(.07,.18,'\n'.join(notes),fontsize=8.5,va='top',color='#425666',linespacing=1.5)
    files=save_figure(fig,path)
    plt.close(fig)
    return files


def save_figure(fig, stem):
    files=[]
    for extension in ('svg','pdf','png'):
        path=Path(str(stem)+'.'+extension)
        fig.savefig(path,dpi=240,facecolor='white')
        files.append(path.name)
    return files


def draw_composition(plot_rows, output):
    import matplotlib.pyplot as plt
    import numpy as np
    categories=list(dict.fromkeys(row['plot_category'] for row in plot_rows))
    lookup={(row['group'],row['plot_category']):row for row in plot_rows}
    palettes=list(plt.get_cmap('tab20').colors)
    color_map={category:palettes[index] for index,category in enumerate(categories)}
    color_map.update(Alu='#478DB3',Non_repeat='#D9DEE2',Unknown_contig='#6A6A6A')
    if 'Non-repeat' in color_map:color_map['Non-repeat']='#D9DEE2'
    color_map['Multiple_families']='#C291BC'
    color_map['Other_repeat']='#C8A374'
    fig,axes=plt.subplots(1,2,figsize=(12.7,6.3))
    fig.subplots_adjust(left=.13,right=.98,top=.81,bottom=.31,wspace=.52)
    for ax,reference,title,letter in zip(axes,['Slic-seq_all','Slic-seq_DRS_covered'],
        ['Full Slic-seq catalogue','DRS-covered Slic-seq'],['A','B']):
        groups=[reference,'Dorado_all','NN_all']
        labels=[]
        left=np.zeros(len(groups))
        for category in categories:
            values=[lookup[group,category]['percent_all_sites'] for group in groups]
            ax.barh(range(len(groups)),values,left=left,color=color_map[category],edgecolor='white',
                    linewidth=.5,height=.62,label=category)
            left+=values
        for group in groups:
            total=lookup[group,categories[0]]['total_sites']
            label={'Slic-seq_all':'Slic-seq','Slic-seq_DRS_covered':'Slic-seq (covered)',
                   'Dorado_all':'Dorado/modkit','NN_all':'NN only'}[group]
            labels.append(f'{label}\nn = {total:,}')
        ax.set_yticks(range(len(groups)),labels,fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0,100)
        ax.set_xlabel('Fraction of unique sites (%)',fontsize=10)
        ax.set_title(title,fontsize=12,pad=15)
        ax.text(-.23,1.12,letter,transform=ax.transAxes,weight='bold',fontsize=17)
        ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.12),ncol=4,fontsize=9,frameon=False)
    fig.suptitle('HeLa inosine: RepeatMasker family composition',fontsize=15,weight='bold',y=.98)
    notes=['One count per site: multiple-family overlaps form a separate category; all families remain in source tables.',
           'Non-repeat and unknown-contig sites are explicit. Top families shown; remaining families grouped as Other_repeat.',
           'Genomic intervals ignore repeat strand. NN includes repeat-mask input; composition is descriptive.']
    fig.text(.06,.095,'\n'.join(notes),va='top',fontsize=8,color='#425666',linespacing=1.4)
    files=save_figure(fig,output/'FigS_HeLa_inosine_repeat_families')
    plt.close(fig)
    return files


def draw_figures(overlaps,plot_rows,summary,output):
    import matplotlib
    matplotlib.use('Agg')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    with matplotlib.rc_context({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42}):
        for row in overlaps[:2]:
            covered=row['scope']=='DRS_depth_eligible'
            name='FigS_HeLa_inosine_Venn_NN_covered' if covered else 'Fig_HeLa_inosine_Venn_NN_only'
            files=draw_venn(row,output/name,covered)
            records.append(dict(name=name,files=files,scope=row['scope']))
        records.append(dict(name='FigS_HeLa_inosine_repeat_families',files=draw_composition(plot_rows,output),
                            scope='mutually exclusive unique-site repeat families'))
    return records
