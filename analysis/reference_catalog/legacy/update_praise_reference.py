from pathlib import Path
from collections import defaultdict, Counter
import openpyxl, json, re, csv, shutil, hashlib
root=Path(r'D:\project\rnaModScan')
raw=root/'data/external/praise/raw'
raw.mkdir(exist_ok=True)
src=Path(r'C:\Users\hirok\Downloads\41589_2023_1304_MOESM3_ESM.xlsx')
dest=raw/src.name
shutil.copy2(src,dest)
out=root/'data/reference_sites_GRCh38'
prep=root/'data/external/praise/prepared'
prep.mkdir(exist_ok=True)
checks=Path(r'D:\project\modScore\praise_reference_check.json')
shutil.copy2(checks,prep/checks.name)
refs={(r['chrom'],int(r['pos'])):r for r in json.loads(checks.read_text(encoding='utf-8-sig'))}
w=openpyxl.load_workbook(dest,read_only=True,data_only=True)
rows=list(w['Supplementary Dataset 2'].values)[3:]
sites=defaultdict(list);ranges=[];audit=[]
for i,r in enumerate(rows,4):
    coord=str(r[-1]);m=re.fullmatch(r'(chr.+)_(\d+)(?:-(\d+))?',coord)
    status='';strand='';dna=''
    if not m: status='missing_coordinate'
    else:
        chrom=m[1];a=int(m[2]);b=int(m[3] or m[2])
        if a!=b:
            strand='+' if a<b else '-'
            if min(a,b)<1: status='range_invalid_zero_coordinate'
            else:
                status='range_separate_only'
                ranges.append((chrom,min(a,b)-1,max(a,b),f'PRAISE_row{i}|{r[0]}',0,strand))
        elif a<1: status='single_invalid_zero_coordinate'
        else:
            dna=refs.get((chrom,a),{}).get('dna','').upper()
            if not dna: status='reference_unavailable'
            elif dna[0] not in ('T','A'): status='reference_not_T_or_A'
            else:
                strand='+' if dna[0]=='T' else '-'
                sites[(chrom,a-1,strand)].append(i)
                status='single_included'
    audit.append([i,r[0],str(r[1]),r[13],coord,status,strand,dna])
single=prep/'HEK293T_PRAISE_single_sites_GRCh38.bed'
single.write_text(''.join(f'{c}\t{s}\t{s+1}\tPRAISE_rows'+','.join(map(str,ids))+f'\t0\t{strand}\n' for (c,s,strand),ids in sorted(sites.items())),encoding='utf-8')
(prep/'HEK293T_PRAISE_ambiguous_ranges_GRCh38.bed').write_text(''.join('\t'.join(map(str,r))+'\n' for r in sorted(ranges)),encoding='utf-8')
with (prep/'Dataset2_row_audit.tsv').open('w',encoding='utf-8-sig',newline='') as f:
    wr=csv.writer(f,delimiter='\t');wr.writerow(['excel_row','transcript','transcript_site','gene','original_chr_site','status','strand','reference_pos_and_next_base']);wr.writerows(audit)
counts=Counter(r[5] for r in audit)
manifest=dict(source=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),doi='10.1038/s41589-023-01304-7',source_url='https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41589-023-01304-7/MediaObjects/41589_2023_1304_MOESM3_ESM.xlsx',retrieval='user supplied original workbook; URL recorded as publisher supplement identity, not fetched in this step',sheet='Supplementary Dataset 2',coordinate_column='O / chr_site',data_rows=len(rows),statuses=dict(counts),unique_single_sites=len(sites),duplicate_single_rows=counts['single_included']-len(sites),reference='UCSC hg38 sequence API; source chr_site interpreted as 1-based; 842/848 positive nonzero unique coordinates match expected T/A if counts unchanged',strand='INFERRED from hg38 base: T -> +, A -> -; not provided in workbook; range orientation inferred from ordered endpoints',ranges='Kept separately, not expanded or included as single-site truth. Two intervals span thousands of bases and may cross introns; bounds do not imply all bases modified.')
(prep/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
builder=out/'build_reference_beds.py'
code=builder.read_text(encoding='utf-8')
anchor="for f in sorted((SRC/'m5c_tac_seq/liftover_candidates').glob('hglft*.bed')):"
block="""# PRAISE Dataset2: only reference-validated single positions; ranges excluded by user decision.
f=SRC/'praise/prepared/HEK293T_PRAISE_single_sites_GRCh38.bed'
rows=[line.split('\\t') for line in f.read_text().splitlines()]
add('psi','HEK293T','PRAISE',f,((r[0],r[1],r[5]) for r in rows),doi='10.1038/s41589-023-01304-7',bed=True)

"""
if "add('psi','HEK293T','PRAISE'" not in code: code=code.replace(anchor,block+anchor)
code=code.replace('PRAISE: candidate table not yet obtained; NOT included.', 'PRAISE: HEK293T Dataset2 single-coordinate candidates with hg38 T/A reference validation included; strand inferred from reference. Ranges excluded from union by user instruction and saved separately in data/external/praise/prepared. See manifest.json and Dataset2_row_audit.tsv for excluded/missing records and original XLSX provenance.')
builder.write_text(code,encoding='utf-8')
note=f'\n2026-10-22 PRAISE added from user-supplied MOESM3 Supplementary Dataset 2, coordinate column O (chr_site). DOI 10.1038/s41589-023-01304-7. Counts: {dict(counts)}. Unique single-coordinate sites: {len(sites)}. Strand inferred from hg38 T/A. User authorized separate range BED, single sites only in union. Source and per-row audit in data/external/praise/prepared.\n'
with (root/'docs/SHORT_READ_COLLECTION_PROTOCOL.txt').open('a',encoding='utf-8') as f:f.write(note)
print(note)
