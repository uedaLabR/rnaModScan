"""Single source of BED0, strand and AF semantics. No analysis/label policy here."""
import gzip
from pathlib import Path

TARGET_CODES = {'a': 'm6A', 'm': 'm5C', '17596': 'inosine', '17802': 'pseU'}
ALL_MODIFICATION_CODES = {
    'a': 'm6A', '17596': 'inosine', 'm': 'm5C', '17802': 'pseU',
    '69426': '2OmeA', '19228': '2OmeC', '19229': '2OmeG', '19227': '2OmeU',
}
CALL_COUNT_FIELDS = tuple('call_count_' + name for name in ALL_MODIFICATION_CODES.values())


def open_text(path):
    path = Path(path)
    if path.suffix in ('.gz', '.bgz'):
        return gzip.open(path, 'rt', encoding='utf-8-sig')
    return path.open('rt', encoding='utf-8-sig')


def parse_bedmethyl_row(line, number):
    c = line.split()
    if len(c) != 18:
        raise ValueError(f'line {number}: expected 18-column modkit bedMethyl')
    start, end = int(c[1]), int(c[2])
    cov = int(c[9])
    mod, canonical, other, deletion, fail, diff, nocall = map(int, c[11:18])
    if start < 0 or end != start + 1 or c[5] not in ('+', '-'):
        raise ValueError(f'line {number}: require single-base, strand-specific records')
    if min(cov, mod, canonical, other, deletion, fail, diff, nocall) < 0:
        raise ValueError(f'line {number}: negative count')
    if cov != mod + canonical + other:
        raise ValueError(f'line {number}: invalid valid_coverage; bedRMod is not supported')
    if ',' in c[3]:
        raise ValueError(f'line {number}: motif-annotated records unsupported; use unpartitioned pileup')
    total = cov + fail + diff + nocall
    return dict(chrom=c[0], start=start, end=end, strand=c[5],
                modification=TARGET_CODES.get(c[3]), mod_code=c[3], total_reads=total,
                modified_reads=mod, af=mod / total if total else 0.0,
                valid_coverage=cov, canonical_reads=canonical, other_mod_reads=other,
                deletions=deletion, failed_calls=fail, different_base_reads=diff, no_calls=nocall)


def bedmethyl_rows(path):
    """Stream plain/gzip BED, validating order and duplicates with bounded memory.

    All eight (and unknown) mod codes are retained in mod_code; modification is
    populated only for the four model targets. No thresholds are applied here.
    """
    chrom, previous, seen, finished = None, -1, set(), set()
    with open_text(path) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if line.startswith(('#', 'track', 'browser')):
                if 'bedrmod' in line.lower():
                    raise ValueError('bedRMod is unsupported')
                continue
            row = parse_bedmethyl_row(line, number)
            if row['chrom'] != chrom:
                if row['chrom'] in finished:
                    raise ValueError('Chromosome blocks must be contiguous')
                if chrom is not None:
                    finished.add(chrom)
                chrom, previous, seen = row['chrom'], -1, set()
            if row['start'] < previous:
                raise ValueError('bedMethyl must be sorted within chromosomes')
            if row['start'] != previous:
                seen.clear()
            key = row['strand'], row['mod_code']
            if key in seen:
                raise ValueError('Duplicate site/modification/strand')
            seen.add(key)
            previous = row['start']
            yield row


def load_bed_sites(path):
    """Exact single-base stranded BED6 keys; duplicate records collapse to a set."""
    sites = set()
    with open_text(path) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith(('#', 'track', 'browser')):
                continue
            c = line.split()
            if len(c) < 6:
                raise ValueError(f'{path}:{number}: expected BED6')
            start, end = int(c[1]), int(c[2])
            if start < 0 or end != start + 1 or c[5] not in ('+', '-'):
                raise ValueError(f'{path}:{number}: require single-base, stranded BED6')
            sites.add((c[0], start, c[5]))
    return sites


def passes_thresholds(row, min_depth, min_modified, min_af):
    return (row['total_reads'] >= min_depth and row['modified_reads'] >= min_modified
            and row['af'] >= min_af)
