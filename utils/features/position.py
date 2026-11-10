"""Strand-aware UTR/CDS/intron annotations shared by training and prediction.

All overlapping same-strand isoforms contribute flags. No representative
transcript is selected, and IDs are audit columns only, never model inputs.
"""
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from urllib.parse import unquote

from utils.io.artifacts import completed_run, file_info
from utils.io.bed import open_text
from utils.references.catalog import sha256

REGIONS = ('utr', 'cds', 'intron', 'other_exon', 'outside_transcript')
POSITION_CHANNEL_NAMES = ('annotation_known',) + tuple('region_' + r for r in REGIONS) + ('region_ambiguous',)
POSITION_FIELDS = ('chrom', 'start', 'end', 'strand') + POSITION_CHANNEL_NAMES + (
    'transcript_count', 'gene_count', 'gene_ids', 'gene_names', 'transcript_ids')
_GTF_FEATURES = {'transcript', 'exon', 'CDS', 'stop_codon', 'UTR', 'five_prime_utr', 'three_prime_utr'}


def checked_position_annotations(path):
    """Load completed GTF reporting metadata with its reference fingerprint."""
    if path is None:
        return {}, None
    path = Path(path)
    run = completed_run(path.parent)
    if (run.get('schema') != 'site_regions_v1'
            or run.get('position_channel_names') != list(POSITION_CHANNEL_NAMES)
            or run.get('sites_sha256') != sha256(path)):
        raise ValueError('Position annotation schema or checksum mismatch; regenerate annotations')
    return load_position_sites(path), dict(path=str(path.resolve()), sha256=sha256(path), gtf=run['gtf'])


def annotation_attributes(text):
    """Accept GTF attributes or GENCODE GFF3's explicit gene/transcript IDs."""
    if re.match(r'[^;\s=]+=', text):
        attributes = {}
        for item in text.split(';'):
            if not item:
                continue
            key, separator, value = item.partition('=')
            if not separator:
                raise ValueError('Invalid GFF3 attribute')
            attributes[key] = unquote(value)
        return attributes
    return dict(re.findall(r'(\S+)\s+"([^"]*)"', text))


def _merge(intervals):
    result = []
    for start, end in sorted(set(intervals)):
        if result and start <= result[-1][1]:
            result[-1] = result[-1][0], max(result[-1][1], end)
        else:
            result.append((start, end))
    return result


def _subtract(intervals, removed):
    result = []
    for start, end in intervals:
        cursor = start
        for lo, hi in removed:
            if hi <= cursor or lo >= end:
                continue
            if cursor < lo:
                result.append((cursor, lo))
            cursor = max(cursor, hi)
        if cursor < end:
            result.append((cursor, end))
    return result


def _contains(intervals, position):
    return any(start <= position < end for start, end in intervals)


@dataclass
class Transcript:
    chrom: str
    strand: str
    gene_id: str
    transcript_id: str
    gene_name: str = ''
    exons: list = field(default_factory=list)
    coding: list = field(default_factory=list)
    utr: list = field(default_factory=list)

    def finish(self):
        self.exons = _merge(self.exons)
        self.coding = _merge(self.coding)
        self.utr = _subtract(_merge(self.utr), self.coding)
        for name, intervals in (('CDS/stop_codon', self.coding), ('UTR', self.utr)):
            if _subtract(intervals, self.exons):
                raise ValueError(f'{name} outside exons: {self.transcript_id}')
        self.start, self.end = self.exons[0][0], self.exons[-1][1]

    def region(self, position):
        if _contains(self.coding, position):
            return 'cds'
        if _contains(self.utr, position):
            return 'utr'
        if _contains(self.exons, position):
            return 'other_exon'  # Includes noncoding or incompletely annotated exons.
        return 'intron'


class GtfAnnotation:
    """Read GENCODE GTF or GFF3 once; coarse bins avoid a genome-wide scan per site."""
    def __init__(self, path):
        transcripts = {}
        self.chromosomes, self.headers, self.bins = set(), [], {}
        self.bin_size = 100_000
        with open_text(path) as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                if line.startswith('#'):
                    self.headers.append(line.rstrip())
                    if re.search(r'\b(?:GRCh37|hg19|GRCm\d+|mm10|mm39)\b', line):
                        raise ValueError('Use a human GRCh38 GTF; the header specifies another assembly')
                    continue
                columns = line.rstrip('\r\n').split('\t')
                if len(columns) != 9:
                    raise ValueError(f'GTF line {number}: expected nine tab-separated fields')
                chrom, _, feature, start, end, _, strand, _, attributes = columns
                self.chromosomes.add(chrom)
                if feature in ('five_prime_UTR', 'three_prime_UTR'):
                    feature = feature.lower()  # GENCODE GFF3 uses uppercase UTR suffixes.
                if feature not in _GTF_FEATURES:
                    continue
                start, end = int(start) - 1, int(end)  # GTF 1-based closed -> BED0.
                attrs = annotation_attributes(attributes)
                gene_id, transcript_id = attrs.get('gene_id'), attrs.get('transcript_id')
                if start < 0 or end <= start or strand not in ('+', '-') or not gene_id or not transcript_id:
                    raise ValueError(f'GTF line {number}: invalid transcript feature')
                tx = transcripts.setdefault(transcript_id, Transcript(chrom, strand, gene_id, transcript_id))
                if (tx.chrom, tx.strand, tx.gene_id) != (chrom, strand, gene_id):
                    raise ValueError(f'Conflicting transcript identifiers: {transcript_id}')
                gene_name = attrs.get('gene_name', '')
                if gene_name:
                    if tx.gene_name and tx.gene_name != gene_name:
                        raise ValueError(f'Conflicting gene_name for transcript: {transcript_id}')
                    tx.gene_name = gene_name
                if feature == 'exon':
                    tx.exons.append((start, end))
                elif feature in ('CDS', 'stop_codon'):
                    tx.coding.append((start, end))
                elif feature != 'transcript':
                    tx.utr.append((start, end))
        if not self.chromosomes:
            raise ValueError('GTF contains no annotation rows')
        self.transcript_count = 0
        self.chromosomes.clear()  # Gene-only chromosomes cannot provide transcript categories.
        for tx in transcripts.values():
            if not tx.exons:
                if tx.coding or tx.utr:
                    raise ValueError(f'GTF transcript lacks exon records: {tx.transcript_id}')
                continue
            tx.finish()
            self.transcript_count += 1
            self.chromosomes.add(tx.chrom)
            for bucket in range(tx.start // self.bin_size, (tx.end - 1) // self.bin_size + 1):
                self.bins.setdefault((tx.chrom, tx.strand, bucket), []).append(tx)
        if not self.transcript_count:
            raise ValueError('GTF contains no transcripts with exon records')

    def annotate(self, chrom, start, strand):
        transcripts = [tx for tx in self.bins.get((chrom, strand, start // self.bin_size), ())
                       if tx.start <= start < tx.end]
        known = chrom in self.chromosomes
        regions = {tx.region(start) for tx in transcripts}
        if known and not transcripts:
            regions.add('outside_transcript')
        genes = sorted({tx.gene_id for tx in transcripts})
        return dict(chrom=chrom, start=start, end=start + 1, strand=strand,
                    annotation_known=int(known),
                    **{'region_' + region: int(region in regions) for region in REGIONS},
                    region_ambiguous=int(len(regions) > 1),
                    transcript_count=len(transcripts), gene_count=len(genes),
                    gene_ids=';'.join(genes),
                    gene_names=';'.join(sorted({tx.gene_name for tx in transcripts if tx.gene_name})),
                    transcript_ids=';'.join(sorted(tx.transcript_id for tx in transcripts)))


def _validate_position(row):
    start = int(row['start'])
    if start < 0 or int(row['end']) != start + 1 or row['strand'] not in ('+', '-') or not row['chrom']:
        raise ValueError('Position annotations require GRCh38 single-base BED coordinates and strand')
    values = [int(row[name]) for name in POSITION_CHANNEL_NAMES]
    if any(str(row[name]) not in ('0', '1') for name in POSITION_CHANNEL_NAMES):
        raise ValueError('Position annotation flags must be 0 or 1')
    known, *flags, ambiguous = values
    if ((not known and any(flags + [ambiguous])) or (known and not any(flags))
            or ambiguous != int(sum(flags) > 1) or (flags[-1] and sum(flags) != 1)):
        raise ValueError('Inconsistent known, region or ambiguous flags')
    return row['chrom'], start, row['strand']


def load_position_sites(path):
    """Exact-site TSV, keyed by (chrom, BED0 start, strand); missing means unknown."""
    if path is None:
        return {}
    result = {}
    with Path(path).open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if not set(('chrom', 'start', 'end', 'strand') + POSITION_CHANNEL_NAMES) <= set(reader.fieldnames or []):
            raise ValueError('Position annotation TSV is missing required fields')
        for row in reader:
            key = _validate_position(row)
            if key in result:
                raise ValueError(f'Duplicate position annotation: {key}')
            for name in POSITION_CHANNEL_NAMES:
                row[name] = int(row[name])
            result[key] = row
    return result


def position_inputs(row, annotations):
    """Seven category/known flags; no IDs, relative positions or splice distances."""
    annotation = annotations.get((row['chrom'], int(row['start']), row['strand']))
    return [float(annotation[name]) for name in POSITION_CHANNEL_NAMES] if annotation else [0.] * len(POSITION_CHANNEL_NAMES)


def annotate_positions(index_paths, gtf_path, output_dir):
    """Annotate unique sites from any number of training/prediction index TSVs.

    UTR and CDS use explicit GTF features. Stop codons count as CDS and are
    removed from overlapping UTR records. A noncoding exon is other_exon, never
    inferred to be intronic. Outside_transcript means no same-strand transcript
    span; it makes no assertion about opposite-strand genes or expression.
    """
    index_paths = [Path(path).resolve() for path in index_paths]
    gtf_path, output_dir = Path(gtf_path).resolve(), Path(output_dir)
    if not index_paths:
        raise ValueError('Supply at least one site index TSV')
    sources = [dict(**file_info(path), sha256=sha256(path)) for path in [gtf_path] + index_paths]
    sites = set()
    for path in index_paths:
        with path.open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                start = int(row['start'])
                if start < 0 or int(row['end']) != start + 1 or row['strand'] not in ('+', '-'):
                    raise ValueError('Site indexes must use single-base BED0 coordinates with strand')
                sites.add((row['chrom'], start, row['strand']))
    if output_dir.exists():
        raise FileExistsError(f'Use a new output directory: {output_dir}')
    output_dir.mkdir(parents=True)
    incomplete = output_dir / 'INCOMPLETE.txt'
    incomplete.write_text('Position annotations are being prepared.\n', encoding='utf-8')
    try:
        annotation = GtfAnnotation(gtf_path)
        partial = output_dir / 'sites.tsv.partial'
        counts = {name: 0 for name in POSITION_CHANNEL_NAMES}
        with partial.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=POSITION_FIELDS, delimiter='\t')
            writer.writeheader()
            for chrom, start, strand in sorted(sites):
                row = annotation.annotate(chrom, start, strand)
                writer.writerow(row)
                for name in counts:
                    counts[name] += row[name]
        for before in sources:
            if file_info(before['path']) != {key: before[key] for key in ('path', 'size_bytes', 'mtime_ns')}:
                raise ValueError('An input changed while producing annotations')
        partial.rename(output_dir / 'sites.tsv')
        run = dict(status='complete', schema='site_regions_v1', assembly='GRCh38',
                   coordinates='BED0', gtf=sources[0], indexes=sources[1:],
                   gtf_headers=annotation.headers, gtf_transcripts=annotation.transcript_count,
                   position_channel_names=list(POSITION_CHANNEL_NAMES), unique_sites=len(sites), counts=counts,
                   policy='All overlapping same-strand isoforms; flags may overlap. No transcript selection.',
                   sites_sha256=sha256(output_dir / 'sites.tsv'))
        (output_dir / 'run.json').write_text(json.dumps(run, indent=2) + '\n', encoding='utf-8')
        incomplete.unlink()
    except Exception as error:
        (output_dir / 'FAILED.txt').write_text(str(error) + '\n', encoding='utf-8')
        raise
    return output_dir / 'sites.tsv'
