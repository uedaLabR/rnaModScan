"""Publication SVG/PDF/PNG Venn panels with exact, auditable region counts."""
from pathlib import Path


COLORS = dict(GLORI='#478DB3', m6Anet='#E39C45', rnaModScan='#5DA584')


def draw_panel(ax, view, letter):
    from matplotlib.patches import Circle
    from analysis.hela_m6a.compare import venn_regions
    methods = view['methods']
    counts = {row['pattern']:row['count'] for row in venn_regions(methods)}
    label = {'NN':'rnaModScan NN only', 'Dorado':'Dorado/modkit'}[view['stage']]
    names = list(methods)
    if len(names) == 3:
        centers, radius = [(-.52,.35),(.52,.35),(0,-.55)], 1.05
        numbers = {'100':(-1.13,.55), '010':(1.13,.55), '001':(0,-1.20),
                   '110':(0,1.02), '101':(-.63,-.40), '011':(.63,-.40), '111':(0,.12)}
        labels = [(-1.05,1.62),(1.05,1.62),(0,-1.83)]
    else:
        centers, radius = [(-.60,0),(.60,0)], 1.08
        numbers = {'10':(-1.02,0),'01':(1.02,0),'11':(0,0)}
        labels = [(-1.05,1.42),(1.05,1.42)]
    for name,center,where in zip(names,centers,labels):
        color = COLORS[name]
        ax.add_patch(Circle(center,radius,facecolor=color,alpha=.27,edgecolor='none'))
        ax.add_patch(Circle(center,radius,facecolor='none',edgecolor=color,linewidth=1.5))
        title = label if name == 'rnaModScan' else name
        ax.text(*where,f'{title}\nn = {len(methods[name]):,}',ha='center',va='center',fontsize=11,
                weight='bold',color=color,linespacing=1.3)
    for pattern,where in numbers.items():
        ax.text(*where,f'{counts[pattern]:,}',ha='center',va='center',fontsize=12,color='#172B3A',weight='bold')
    group = 'DRACH' if view['region']=='DRACH' else 'Non-DRACH'
    scope = {'full_catalog':'Full GLORI catalogue', 'm6anet_scored':'m6Anet-scored positions only',
             'DRS_depth_eligible':'GLORI: DRS depth >= 10'}[view['universe']]
    ax.set_title(group+'\n'+scope,fontsize=13,pad=17,linespacing=1.5)
    ax.text(-.02,1.13,letter,transform=ax.transAxes,weight='bold',fontsize=18)
    ax.set(xlim=(-2,2),ylim=(-2.1,1.95),aspect='equal')
    ax.axis('off')


def draw_figures(views, output, *, threshold=.8, sensitivity_thresholds=(.9,)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    mapping = {(v['region'],v['stage'],v['universe'],v['m6anet_threshold']):v for v in views}
    specs = [
        ('Fig_HeLa_m6A_Venn_NN_only','NN','full_catalog','full_catalog',threshold),
        ('FigS_HeLa_m6A_Venn_Dorado','Dorado','full_catalog','full_catalog',threshold),
        ('FigS_HeLa_m6A_Venn_NN_covered','NN','m6anet_scored','DRS_depth_eligible',threshold)]
    for cutoff in sensitivity_thresholds:
        suffix = f'{cutoff:.2f}'.replace('.', 'p')
        specs.append(('FigS_HeLa_m6A_Venn_NN_p'+suffix,'NN','full_catalog','full_catalog',cutoff))
    records=[]
    with matplotlib.rc_context({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42}):
        for stem,stage,left_scope,right_scope,cutoff in specs:
            fig,axes=plt.subplots(1,2,figsize=(12,6.8))
            fig.subplots_adjust(left=.04,right=.98,bottom=.24,top=.79,wspace=.20)
            panels=[mapping['DRACH',stage,left_scope,cutoff],mapping['non_DRACH',stage,right_scope,cutoff]]
            for ax,view,letter in zip(axes,panels,('A','B')):
                draw_panel(ax,view,letter)
            fig.suptitle('HeLa WT: m6A site overlap',fontsize=16,weight='bold',y=.98)
            notes=['Exact GRCh38 coordinate and strand; schematic circles (areas are not proportional).',
                   f'm6Anet site probability >= {cutoff:.2f}; genomic isoforms collapsed by maximum probability.']
            if stage=='NN':
                notes.append('NN only: m6A softmax argmax before reference/DRACH rescue.')
            else:
                notes.append('Dorado/modkit candidates: depth >= 10, modified reads >= 3, AF >= 0.10.')
            notes.append('Reference non-overlap does not establish a false positive; non-DRACH m6Anet is outside this comparison.')
            fig.text(.055,.16,'\n'.join(notes),fontsize=8.3,va='top',linespacing=1.5,color='#425666')
            files=[]
            for extension in ('svg','pdf','png'):
                path=output/(stem+'.'+extension)
                fig.savefig(path,dpi=240,facecolor='white')
                files.append(path.name)
            plt.close(fig)
            records.append(dict(name=stem,views=[v['name'] for v in panels],files=files,area_proportional=False))
    return records
