from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
import csv, gzip, hashlib, json
import openpyxl

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'data/external'
OUT = ROOT / 'analysis/reference_catalog/results/short_read_rebuild'
OUT.mkdir(parents=True, exist_ok=False)
(OUT/'by_method').mkdir(exist_ok=True)
groups = defaultdict(lambda: defaultdict(set))
sources, rawcounts, replstats, liftstats = [], [], [], []

def add(mod, cell, method, path, rows, sheet='', doi='', bed=False):
    records = []
    for chrom, pos, strand in rows:
        pos = int(pos)
        assert chrom.startswith('chr') and strand in ('+', '-')
        start = pos if bed else pos-1
        assert start >= 0
        records.append((chrom,start,strand))
    sites = set(records)
    groups[mod,cell][method].update(sites)
    rawcounts.append([mod,cell,method,path.name,sheet,len(records),len(sites)])
    sources.append(dict(modification=mod,cell=cell,method=method,path=str(path),sheet=sheet,doi=doi,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),coordinate_conversion='already BED0' if bed else 'start=position-1; end=position'))
    return sites

for cell in ('HeLa','HEK293T'):
    reps=[]
    for f in sorted((SRC/'glori/raw').glob('*.csv.gz')):
        if ('Hela' in f.name) != (cell=='HeLa'): continue
        with gzip.open(f,'rt') as h: rows=list(csv.DictReader(h,delimiter='\t'))
        reps.append(add('m6A',cell,'GLORI',f,((r['Chr'],r['Sites'],r['Strand']) for r in rows),doi='10.1038/s41587-022-01487-9'))
    count=Counter(x for r in reps for x in r)
    replstats.append(['m6A',cell,'GLORI',len(reps),len(count),sum(v>=2 for v in count.values()),sum(v==len(reps) for v in count.values())])
    reps=[]
    wpath=SRC/'slic_seq/raw/GSE169710_Data_S2_A-to-I_sites_identified_by_slic-seq.xlsx'
    w=openpyxl.load_workbook(wpath,read_only=True,data_only=True)
    for sheet in w.sheetnames:
        if not sheet.lower().startswith(('hela' if cell=='HeLa' else 'hek293t')+'-rep'): continue
        vals=list(w[sheet].values)
        header=vals[0]; rows=[dict(zip(header,r)) for r in vals[1:] if r[0] is not None]
        reps.append(add('inosine',cell,'Slic-seq',wpath,((r['Chromosome'],r['position'],r['strand']) for r in rows),sheet=sheet,doi='10.1093/nar/gkad604'))
    assert reps, w.sheetnames
    count=Counter(x for r in reps for x in r)
    replstats.append(['inosine',cell,'Slic-seq',len(reps),len(count),sum(v>=2 for v in count.values()),sum(v==len(reps) for v in count.values())])
    f=SRC/f'bid_seq/raw/GSE179798_{cell}_mRNA_WT_BID-seq.xlsx'
    w=openpyxl.load_workbook(f,read_only=True,data_only=True)
    rows=list(w.active.values)[4:]
    add('psi',cell,'BID-seq',f,((r[0],r[1],r[5]) for r in rows if r[0]),sheet='Sheet1',doi='10.1038/s41587-022-01505-w')

f=SRC/'bacs/raw/GSE241849_Supplementary_Table_3_10.xlsx'
w=openpyxl.load_workbook(f,read_only=True,data_only=True)
for mod, sheet in [('psi','Supplementary Table 8'),('inosine','Supplementary Table 10')]:
    rows=list(w[sheet].values)[3:]
    add(mod,'HeLa','BACS',f,((r[0],r[1],r[2]) for r in rows if r[0]),sheet=sheet,doi='10.1038/s41592-024-02439-8')

# PRAISE Dataset2: only reference-validated single positions; ranges excluded by user decision.
f=SRC/'praise/prepared/HEK293T_PRAISE_single_sites_GRCh38.bed'
rows=[line.split('\t') for line in f.read_text().splitlines()]
add('psi','HEK293T','PRAISE',f,((r[0],r[1],r[5]) for r in rows),doi='10.1038/s41589-023-01304-7',bed=True)

for f in sorted((SRC/'m5c_tac_seq/liftover_candidates').glob('hglft*.bed')):
    rows=[line.split() for line in f.read_text().splitlines() if line and not line.startswith('#')]
    cells={r[3].split('_')[0] for r in rows}
    assert len(cells)==1
    cell=cells.pop(); assert cell in ('HeLa','HEK293T')
    original=SRC/f'm5c_tac_seq/liftover_candidates/{cell}_m5C_TAC_cutoff1_plus_cutoff2_hg19.bed'
    names={r.split()[3] for r in original.read_text().splitlines()}
    mapped={r[3] for r in rows}
    assert mapped==names and len(mapped)==len(rows)
    assert all(int(r[2])-int(r[1])==1 for r in rows)
    liftstats.append([cell,len(names),len(mapped),len(names-mapped)])
    add('m5C',cell,'m5C-TAC-seq',f,((r[0],r[1],r[5]) for r in rows),doi='10.1016/j.molcel.2024.06.021',bed=True)

def bedfile(path,mod,cell,sites,support):
    with path.open('w',encoding='utf-8',newline='') as f:
        for chrom,start,strand in sorted(sites):
            methods=','.join(sorted(support[(chrom,start,strand)]))
            f.write(f'{chrom}\t{start}\t{start+1}\t{mod}|{cell}|{methods}\t0\t{strand}\n')
    lines=path.read_text().splitlines()
    assert len(lines)==len(sites)
    assert len({(r.split('\t')[0],r.split('\t')[1],r.split('\t')[5]) for r in lines})==len(sites)

summary=[]; methods=[]; overlaps=[]
for mod in ('m6A','m5C','inosine','psi'):
    for cell in ('HeLa','HEK293T'):
        sets=groups[mod,cell]; assert sets
        support=defaultdict(set)
        for method,sites in sets.items():
            for site in sites: support[site].add(method)
            methods.append([mod,cell,method,len(sites)])
            bedfile(OUT/'by_method'/f'{cell}_{mod}_{method}_GRCh38.bed',mod,cell,sites,{s:{method} for s in sites})
        union=set(support)
        bedfile(OUT/f'{cell}_{mod}_GRCh38.bed',mod,cell,union,support)
        total=sum(map(len,sets.values()))
        summary.append([mod,cell,';'.join(sets),total,len(union),total-len(union),sum(len(m)>=2 for m in support.values())])
        for a,b in combinations(sets,2): overlaps.append([mod,cell,a,b,len(sets[a]&sets[b]),len(sets[a]-sets[b]),len(sets[b]-sets[a]),len(sets[a]|sets[b])])

def tsv(name,header,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f,delimiter='\t'); writer.writerow(header); writer.writerows(rows)
tsv('union_audit.tsv',['modification','cell','methods','sum_method_unique','union_unique','duplicate_entries_removed','sites_supported_by_multiple_methods'],summary)
tsv('summary.tsv',['modification','sample','unique_sites'],[(r[0],r[1],r[4]) for r in summary])
tsv('method_counts.tsv',['modification','method','sample','unique_sites'],[(r[0],r[2],r[1],r[3]) for r in methods])
tsv('method_overlaps.tsv',['modification','cell','method_A','method_B','intersection','A_only','B_only','union'],overlaps)
tsv('source_counts.tsv',['modification','cell','method','file','sheet','data_rows','unique_sites'],rawcounts)
tsv('replicate_support.tsv',['modification','cell','method','replicates','union','support_ge2','all_replicates'],replstats)
tsv('liftover_audit.tsv',['cell','input_sites','mapped_ids','missing_ids'],liftstats)
(OUT/'sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'README.txt').write_text('''GRCh38 short-read candidate reference BEDs (2026-10-22)
Root: 8 BED6 files = 4 modifications x 2 cell lines. psi means pseudouridine.
Cell line is HEK293T, not HEK293. No cells are merged.
Deduplication key: chromosome + 0-based start + strand, within modification/cell.
Single-nucleotide BED6: chr, start, start+1, modification|cell|methods, 0, strand.
by_method/: method-specific unique BEDs. Root files are unions across methods.
GLORI and Slic-seq: union of all available replicate candidate lists; replicate_support.tsv separately records >=2/all-replicate counts. This is not an intersection-only training set.
m5C: user-provided hg19->hg38 liftOver outputs, cutoff-1 plus cutoff-2-additional. All original site IDs retained. BED coordinates used unchanged. Reference assembly is based on user's conversion context; not independently established from BED metadata.
Psi: BID-seq WT (>10% fraction) for both cells, plus BACS Table 8 HeLa polyA+.
Inosine: Slic-seq Data S2 plus BACS Table 10 HeLa polyA+.
PRAISE: HEK293T Dataset2 single-coordinate candidates with hg38 T/A reference validation included; strand inferred from reference. Ranges excluded from union by user instruction and saved separately in data/external/praise/prepared. See manifest.json and Dataset2_row_audit.tsv for excluded/missing records and original XLSX provenance. BACS HEK293T table not available in current sources; missing does not mean zero.
No new AF, DRACH, coverage, biotype, or reproducibility filters applied. Source thresholds differ. Slic-seq includes intronic/intergenic sites. These are candidate evidence, not automatically confirmed training labels; absent sites are unlabeled, not negative.
Original GLORI/Slic/BACS/BID positions converted as 1-based to BED0. hg38 UCSC reference spot checks support this: Slic chr1:492159(-),727162(-); GLORI chr10:47499(-),3835789(+); BACS psi chr1:100874411(-),10403085(+); BID chr1:43450939(+),chrX:78129969(-); BACS inosine chr1:1318636(-). Expected RNA base matches at position-1. This is a spot check, not exhaustive reference validation.
summary.tsv: modification/sample unique union counts; method_counts.tsv: modification/method/sample unique counts; union_audit.tsv: detailed union accounting; method_overlaps.tsv: pairwise overlap; source_counts.tsv: per-source rows; replicate_support.tsv: reproducibility counts; liftover_audit.tsv: ID accounting.
Source paths, sheets, DOI, SHA256 and coordinate conversions are recorded in sources.json; original download URLs are in data/external/<method> manifests/inventory and docs/SHORT_READ_COLLECTION_PROTOCOL.txt.
''',encoding='utf-8')
print(json.dumps(dict(summary=summary,methods=methods,overlaps=overlaps,liftover=liftstats),indent=2))
