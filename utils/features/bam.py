"""Eight-modification read bags; no truth/label columns in model feature arrays."""
import bisect
from copy import copy
import csv
import json
import hashlib
import random
import re
import time
import os
from collections import Counter, defaultdict
from pathlib import Path

from utils.references.catalog import sha256, MODIFICATIONS as TARGETS
from utils.io.artifacts import file_info
from utils.io.cigar import CigarIndex
from utils.features.parquet import (ShardedParquetWriter, PartitionedParquetWriter,
                                   feature_schema, site_schema, require_arrow)
from utils.features.grouping import candidate_groups, candidate_group_batches, read_batches

MODIFICATIONS = ('m6A', 'inosine', 'm5C', 'pseU', '2OmeA', '2OmeC', '2OmeG', '2OmeU')
MOD_CODES = {'a': 'm6A', '17596': 'inosine', 'm': 'm5C', '17802': 'pseU',
             '69426': '2OmeA', '19228': '2OmeC', '19229': '2OmeG', '19227': '2OmeU'}
CANONICAL = dict(zip(MODIFICATIONS, ('A', 'A', 'C', 'T', 'A', 'C', 'G', 'T')))
COMPLEMENT = str.maketrans('ACGTNacgtn', 'TGCANtgcan')


def build_extraction_report(site_list, sample_id, bam_path, reference_fasta, bam, *,
                            flank, min_mapq, bam_threads, parquet_batch_rows, parquet_file_rows,
                            parquet_files, counts, channels, partition_by_locus=False, read_ranges=None,
                            candidate_group_gap=None, read_batch_size=256, group_batch_size=1,
                            max_fetch_span=10000000):
    """Describe extracted data without requiring a parseable header dictionary.

    Some input BAMs have duplicate @HD lines but can be read and indexed by
    htslib. Preserve their original SAM header and the parsing warning; this
    audit conversion must not discard hours of completed feature extraction.
    """
    import pysam

    header_text = str(bam.header)
    try:
        header = bam.header.to_dict()
        header_warning = None
    except ValueError as error:
        header = None
        header_warning = str(error)
    return dict(schema='read_features_parquet_v3', status='complete', sample_id=sample_id,
                extraction_strategy='one_bam_pass_all_targets' if candidate_group_gap is None else 'candidate_groups_indexed_fetch',
                bam_passes=1 if candidate_group_gap is None else None,
                candidate_group_gap=candidate_group_gap,
                group_batch_size=group_batch_size if candidate_group_gap is not None else None,
                max_fetch_span=max_fetch_span if candidate_group_gap is not None else None,
                read_batch_size=read_batch_size if candidate_group_gap is not None else None,
                site_list=file_info(site_list), site_list_sha256=sha256(site_list),
                bam=file_info(bam_path), bam_header=header, bam_header_sam=header_text,
                bam_header_parse_warning=header_warning,
                reference=file_info(reference_fasta), reference_fai_sha256=sha256(str(reference_fasta) + '.fai'),
                pysam_version=pysam.__version__, pyarrow_version=require_arrow()[0].__version__,
                bam_threads=bam_threads, modification_channels=MODIFICATIONS, parquet_compression='zstd',
                parquet_batch_rows=parquet_batch_rows, parquet_file_rows=parquet_file_rows,
                parquet_files=parquet_files, partition_by_locus=partition_by_locus,
                read_ranges=read_ranges if partition_by_locus else None,
                read_offset_mode='logical emission order; read_ranges map partition files' if partition_by_locus else 'physical manifest order',
                window_flank=flank, max_reads=None, min_mapq=min_mapq,
                counts=dict(counts), observed_mm_groups=dict(channels) if channels is not None else None,
                missing_mm_groups=[m for m in MODIFICATIONS if not channels[m]] if channels is not None else None,
                policies=dict(orientation='target RNA 5-prime to 3-prime; genome window, no exon stitching',
                    sampling='all eligible reads; no subsampling',
                    eligible_depth='same-strand primary nonduplicate non-QC-failed centered aligned bases after MAPQ filter; before sampling',
                    ml='original byte 0..255; interval [byte/256,(byte+1)/256); null is not zero',
                    missing='unknown ML=-1/unreported remain null; observed/applicable masks and MM skip modes retained',
                    exclusions='central D/N counted separately; opposite strand/secondary/supplementary excluded',
                    splice_distance='nearest CIGAR N in the full same read, before/after in RNA 5-prime order; '
                                    'intervening query bases including insertions, excluding D/N; '
                                    'adjacent exon base=0, unobserved side=null; no annotation-derived junctions',
                    predictors='labels, overlaps, condition/cell identities and group IDs excluded from feature arrays',
                    scope='eight modification inputs; target-specific or joint adenine site bags; no NN training or POD5 processing'))


def iter_sites(path, sample_id):
    """Stream coordinate-sorted TSV; retain only duplicate keys at the current locus."""
    last = None
    seen = set()
    with Path(path).open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row['sample_id'] != sample_id:
                continue
            row = {k: row[k] for k in ('row_id', 'sample_id', 'chrom', 'start', 'end', 'strand', 'modification')}
            row['start'] = int(row['start'])
            row['end'] = int(row['end'])
            locus = row['chrom'], row['start']
            if last is not None and locus < last:
                raise ValueError('Site TSV must be sorted by chromosome and start')
            if locus != last:
                seen.clear()
            key = row['strand'], row['modification']
            if (row['start'] < 0 or row['end'] != row['start'] + 1 or row['strand'] not in ('+', '-')
                    or row['modification'] not in (*TARGETS, 'adenine') or key in seen):
                raise ValueError('Invalid or duplicate site')
            seen.add(key)
            last = locus
            yield row


def read_map(read, lo, hi):
    """Convenience for single-window callers; extraction reuses one CigarIndex."""
    return CigarIndex(read).window(lo, hi)


def read_splice_junctions(read):
    """CIGAR N boundaries in stored-query interbase coordinates, over the full read.

    Insertions and soft clips consume query bases; deletions and reference skips
    do not. Junctions outside the local feature window are retained.
    """
    return CigarIndex(read).junctions


def splice_distances(junctions, center_q, reverse=False):
    """Nearest observed junction before/after the center in RNA 5-prime order.

    Distance zero means the aligned center base touches the junction. Distances
    count intervening query bases, including insertions, excluding D and N.
    None means no junction was observed on that side in this read.
    """
    if center_q is None:
        return None, None
    split = bisect.bisect_right(junctions, center_q)
    before = None
    after = None
    if split:
        before = center_q - junctions[split - 1]
    if split < len(junctions):
        after = junctions[split] - center_q - 1
    if reverse:
        return after, before
    return before, after


def decode_mods(read, needed_positions=None):
    """Use original-read coordinates to remove ambiguity on reverse alignments."""
    if read.has_tag('MN') and read.get_tag('MN') != read.query_length:
        raise ValueError(f'{read.query_name}: MN/SEQ length mismatch')
    tag = None
    for name in ('MM', 'Mm'):
        if read.has_tag(name):
            tag = name
            break
    modes = {}
    if tag:
        for group in read.get_tag(tag).split(';'):
            if not group:
                continue
            match = re.fullmatch(r'([ACGTUN])([+-])([a-z]+|\d+)([.?]?)(?:,.*)?', group)
            if not match:
                raise ValueError(f'{read.query_name}: unsupported MM group {group}')
            base, strand, codes, mode = match.groups()
            if strand == '-':
                continue  # Only the sequenced RNA strand is a predictor.
            if codes.isdigit():
                codes = [codes]
            for code in codes:
                if code in MOD_CODES:
                    mod = MOD_CODES[code]
                    if base.replace('U', 'T') != CANONICAL[mod] or mod in modes:
                        raise ValueError(f'{read.query_name}: incompatible/duplicate MM channel {mod}')
                    modes[mod] = mode or '.'
    parsed = decode_reported_positions(read, tag)
    if tag and parsed is None:
        raise ValueError(f'{read.query_name}: MM/ML could not be decoded')
    result = {}
    query_length, reverse = read.query_length, read.is_reverse
    for (base, strand, code), positions in (parsed or {}).items():
        mod = MOD_CODES.get(str(code))
        if mod is None or strand != 0:
            continue
        for q, score in positions:
            if not 0 <= q < query_length or not -1 <= score <= 255:
                raise ValueError('Invalid decoded modification position/score')
            stored_q = q
            if reverse:
                stored_q = query_length - 1 - q
            if needed_positions is not None and stored_q not in needed_positions:
                continue
            result[stored_q, mod] = score
    return result, modes


def decode_reported_positions(read, tag):
    """Decode explicit MM positions without HTSlib's empty-numeric-group issue.

    Empty groups still declare channels and skip modes, but carry no ML entries.
    Retain those declarations above; remove them only from a temporary parser copy.
    """
    if tag is None:
        return {}
    groups = [group for group in read.get_tag(tag).split(';') if group]
    reported_groups = [group for group in groups if ',' in group]
    if len(reported_groups) == len(groups):
        return read.modified_bases_forward
    if not reported_groups:
        for name in ('ML', 'Ml'):
            if read.has_tag(name) and len(read.get_tag(name)):
                raise ValueError(f'{read.query_name}: MM/ML entry counts differ')
        return {}
    parser_read = copy(read)
    parser_read.set_tag(tag, ';'.join(reported_groups) + ';')
    return parser_read.modified_bases_forward


def reference_window(fasta, site, flank):
    return reference_context(fasta, site, flank)[0]


def reference_context(fasta, site, flank):
    """Read this site's sequence and repeat flags together, before uppercasing."""
    from utils.references.softmask import softmasked_window
    return softmasked_window(fasta, site['chrom'], site['start'], site['strand'], 2 * flank + 1)


def query_base_features(query, sequence, qualities, reverse, mods, cache):
    """Reuse base/ML arrays at shared query positions of overlapping windows."""
    if query in cache:
        return cache[query]
    base, quality = None, None
    if query is not None:
        base = sequence[query].upper()
        if reverse:
            base = base.translate(COMPLEMENT)
        if qualities is not None:
            quality = int(qualities[query])
    values = [mods.get((query, mod)) for mod in MODIFICATIONS]
    raw = [value if value is not None and value >= 0 else None for value in values]
    observed = [value is not None and value >= 0 for value in values]
    applicable = [base == CANONICAL[mod] for mod in MODIFICATIONS]
    result = base, quality, raw, observed, applicable
    cache[query] = result
    return result


def encode_read(read, site, flank, mapping, insertions, mods, modes, reference, junctions=None,
                read_context=None, base_cache=None, read_metadata=None):
    """Encode one short window, using coordinates and tags already decoded for this read."""
    reverse = site['strand'] == '-'
    if read_context is None:
        read_context = read.query_sequence, read.query_qualities
    sequence, query_qualities = read_context
    if base_cache is None:
        base_cache = {}
    if read_metadata is None:
        read_metadata = dict(read_id=read.query_name, alignment_start=read.reference_start,
                             cigar=read.cigarstring, mapq=read.mapping_quality, query_length=read.query_length)
    positions = list(range(site['start'] - flank, site['start'] + flank + 1))
    if reverse:
        positions.reverse()
    bases = []
    states = []
    qualities = []
    mismatches = []
    ins = []
    raw_ml = []
    applicable = []
    observed = []
    for i, p in enumerate(positions):
        q, state = mapping.get(p, (None, 'uncovered'))
        base, quality, ml, ml_observed, ml_applicable = query_base_features(
            q, sequence, query_qualities, reverse, mods, base_cache)
        bases.append(base)
        states.append(state)
        qualities.append(quality)
        mismatch = None
        if base is not None and base in 'ACGT' and reference[i] in 'ACGT':
            mismatch = int(base != reference[i])
        mismatches.append(mismatch)
        # Anchor insertions after the preceding RNA-oriented reference base.
        insertion_anchor = p - 1 if reverse else p
        ins.append(insertions.get(insertion_anchor, 0))
        raw_ml.append(ml)
        observed.append(ml_observed)
        applicable.append(ml_applicable)
    if junctions is None:
        junctions = read_splice_junctions(read)
    center_query, _ = mapping.get(site['start'], (None, None))
    before, after = splice_distances(junctions, center_query, reverse)
    return dict(read_metadata,
                splice_distance_before=before, splice_distance_after=after,
                mm_skip_modes=modes, base=bases, alignment_state=states, base_quality=qualities,
                mismatch=mismatches, insertion_length_after=ins, ml_byte=raw_ml,
                modification_applicable=applicable, modification_observed=observed)



def load_sample_sites(site_list, sample_id, flank):
    """Load this sample's candidate metadata and check the upstream window radius."""
    parents = (site_list.parent, site_list.parent.parent)
    for parent in parents:
        for marker in ('FAILED.txt', 'INCOMPLETE.txt'):
            if (parent / marker).exists():
                raise ValueError('Site-list run failed; do not extract incomplete outputs')
    for parent in parents:
        manifest = parent / 'run.json'
        if manifest.exists():
            run = json.loads(manifest.read_text())
            if run.get('window_flank', flank) < flank:
                raise ValueError('Feature window exceeds site-list grouping radius; regenerate lists')
            break
    by_chrom = defaultdict(list)
    for site in iter_sites(site_list, sample_id):
        by_chrom[site['chrom']].append(site)
    if not by_chrom:
        raise ValueError(f'No sites for {sample_id} in {site_list}')
    return by_chrom


def validate_reference_sites(by_chrom, bam, fasta):
    """Require the same contig lengths in BAM and FASTA, and valid candidate sites."""
    for chrom, sites in by_chrom.items():
        if chrom not in bam.references or chrom not in fasta.references:
            raise ValueError(f'BAM/FASTA contig missing: {chrom}')
        if bam.get_reference_length(chrom) != fasta.get_reference_length(chrom):
            raise ValueError(f'BAM/FASTA contig length mismatch: {chrom}')
        if sites[-1]['start'] >= fasta.get_reference_length(chrom):
            raise ValueError(f'Site outside {chrom}')


def is_eligible_read(read, min_mapq):
    """Keep primary reads passing flag/MAPQ filters; check each site's strand later."""
    excluded = read.is_unmapped or read.is_secondary or read.is_supplementary
    excluded = excluded or read.is_duplicate or read.is_qcfail
    return not excluded and read.mapping_quality >= min_mapq


def new_site_bag(site, sampling_seed):
    """Seed a site's reservoir using sample and position, never labels or scores."""
    key = '|'.join(str(site[name]) for name in ('sample_id', 'chrom', 'start', 'strand'))
    digest = hashlib.sha256(key.encode()).digest()
    seed = int.from_bytes(digest[:16], 'little') ^ sampling_seed
    return dict(reads=[], eligible_depth=0, deletions=0, skips=0, reference=None, repeat_mask=None,
                reservoir_seed=seed, random=None)


def reservoir_slot(bag, max_reads):
    """Choose a retained slot uniformly before constructing a read's features.

    Standard reservoir sampling gives every eligible read probability k/n.
    A rejected read still contributes to the full eligible_depth count.
    """
    bag['eligible_depth'] += 1
    if max_reads is None or bag['eligible_depth'] <= max_reads:
        return len(bag['reads'])
    if bag['random'] is None:
        bag['random'] = random.Random(bag['reservoir_seed'])
    slot = bag['random'].randrange(bag['eligible_depth'])
    return slot if slot < max_reads else None


class DeferredRead:
    """Share one alignment/CIGAR across reservoirs; decode only after final selection."""
    def __init__(self, read, coordinates):
        self.read = read
        self.coordinates = coordinates
        self.windows = {}
        self.needed_positions = set()
        self.modifications = None
        self.modes = None
        self.read_context = None
        self.metadata = None
        self.base_cache = {}

    def add_window(self, center, flank):
        """Save only CIGAR geometry while the reservoir is still changing."""
        if center not in self.windows:
            mapping, insertions = self.coordinates.window(center - flank, center + flank + 1)
            self.windows[center] = mapping, insertions
            self.needed_positions.update(query for query, _ in mapping.values() if query is not None)

    def features_for_site(self, site, reference, flank, counts, channels):
        """Called only for reads left in a completed site's final reservoir."""
        if self.modifications is None:
            self.modifications, self.modes = decode_mods(self.read, self.needed_positions)
            counts['modification_decodes'] += 1
            if channels is not None:
                channels.update(self.modes.keys())
            self.read_context = self.read.query_sequence, self.read.query_qualities
            self.metadata = dict(read_id=self.read.query_name, alignment_start=self.read.reference_start,
                                 cigar=self.read.cigarstring, mapq=self.read.mapping_quality,
                                 query_length=self.read.query_length)
        center = site['start']
        mapping, insertions = self.windows[center]
        feature = encode_read(self.read, site, flank, mapping, insertions,
            self.modifications, self.modes, reference, self.coordinates.junctions,
            self.read_context, self.base_cache, self.metadata)
        counts['encoded_read_site_windows'] += 1
        return feature  # The site/writer owns this value; do not cache completed-site features.


def collect_read_features(read, sites, positions, pending, flank, counts,
                          max_reads=None, sampling_seed=42, site_range=None):
    """Count eligible coverage and retain shared read references, without MM/features."""
    reference_end = read.reference_end
    if not read.query_length or reference_end is None:
        return
    first = bisect.bisect_left(positions, read.reference_start)
    last = bisect.bisect_left(positions, reference_end)
    if site_range is not None:
        first = max(first, site_range[0])
        last = min(last, site_range[1])
    matching = [index for index in range(first, last)
                if read.is_reverse == (sites[index]['strand'] == '-')]
    if not matching:
        return
    coordinates = CigarIndex(read)
    counts['coordinate_index_builds'] += 1
    retained = None
    chrom = read.reference_name
    for site_index in matching:
        site = sites[site_index]
        query, state = coordinates.reference_to_query(site['start'])
        key = chrom, site_index
        if key not in pending:
            pending[key] = new_site_bag(site, sampling_seed)
        bag = pending[key]
        if state == 'deletion':
            bag['deletions'] += 1
        elif state == 'refskip':
            bag['skips'] += 1
        if query is None:
            continue
        slot = reservoir_slot(bag, max_reads)
        if slot is None:
            continue
        if retained is None:
            retained = DeferredRead(read, coordinates)
        retained.add_window(site['start'], flank)
        if slot == len(bag['reads']):
            bag['reads'].append(retained)
        else:
            bag['reads'][slot] = retained
        counts['reservoir_admissions'] += 1
        counts['peak_retained_reads_per_site'] = max(counts['peak_retained_reads_per_site'], len(bag['reads']))
    counts['peak_active_sites'] = max(counts['peak_active_sites'], len(pending))


def write_site_features(site, bag, fasta, flank, read_writer, site_writer, summary_writer, counts,
                        site_sink=None, channels=None):
    """Write one complete site's contiguous read rows, offset and audit counts."""
    if bag is None:
        bag = dict(reads=[], deletions=0, skips=0, reference=None)
    retained = bag['reads']
    reference = bag['reference']
    repeat_mask = bag.get('repeat_mask')
    if reference is None or repeat_mask is None:
        reference, repeat_mask = reference_context(fasta, site, flank)
    reads = [read.features_for_site(site, reference, flank, counts, channels) for read in retained]
    retained.clear()  # Release discarded/shared contexts when no other unfinished site needs them.
    retained_depth = len(reads)
    eligible_depth = bag.get('eligible_depth', retained_depth)
    offset = counts['retained_read_site_pairs']
    for parquet_writer in (read_writer, site_writer):
        if isinstance(parquet_writer, PartitionedParquetWriter):
            parquet_writer.set_partition(site['chrom'], site['strand'])
    if site_sink is not None:
        site_sink.append_site(site, reads, reference, eligible_depth, bag['deletions'], bag['skips'],
                              sequence_repeat_mask=repeat_mask)
    else:
        site_writer.append(dict(
            site, reference_sequence=reference, eligible_depth=eligible_depth, retained_depth=retained_depth,
            central_deletions=bag['deletions'], central_refskips=bag['skips'], read_offset=offset))
        for feature in reads:
            row = dict(row_id=site['row_id'], **feature)
            row['mm_skip_modes'] = [feature['mm_skip_modes'].get(mod) for mod in MODIFICATIONS]
            read_writer.append(row)
    observed = []
    for channel in range(len(MODIFICATIONS)):
        observed.append(sum(read['modification_observed'][flank][channel] for read in reads))
    summary_writer.writerow([
        site['row_id'], eligible_depth, retained_depth, offset, bag['deletions'], bag['skips'], *observed])
    counts['sites'] += 1
    counts['retained_read_site_pairs'] += retained_depth
    counts['eligible_read_site_pairs'] += eligible_depth
    reads.clear()  # Arrow/writer buffers own remaining rows; release the site's bag.


def report_extraction_progress(out, sample_id, chrom, position, counts, started, read_writer, complete=False):
    """Persist a small heartbeat without retaining BAM records or feature tables."""
    elapsed = time.monotonic() - started
    total_records = counts['bam_records_total']
    grouped = 'candidate_groups_total' in counts
    scan_percent = 100 * counts['bam_records_scanned'] / total_records if total_records and not grouped else None
    report = dict(sample_id=sample_id, pid=os.getpid(), status='complete' if complete else 'running',
                  updated_unix=time.time(), chrom=chrom, position=position, elapsed_seconds=round(elapsed, 1),
                  counts=dict(counts), read_parquet_files=len(read_writer.files),
                  bam_scan_percent=round(scan_percent, 2) if scan_percent is not None else None,
                  reads_per_second=round(counts['bam_records_scanned'] / max(elapsed, .001), 1))
    if grouped:
        report['progress_unit'] = 'candidate_groups'
        total_groups = counts['candidate_groups_total']
        report['candidate_group_percent'] = round(100 * counts['candidate_groups_completed'] / total_groups, 2)
    partial = out / 'progress.json.partial'
    partial.write_text(json.dumps(report, indent=2), encoding='utf-8')
    partial.replace(out / 'progress.json')
    progress_label = (f"groups={counts['candidate_groups_completed']:,}/{counts['candidate_groups_total']:,} "
                      f"alignment visits={counts['bam_records_scanned']:,}" if grouped else
                      f"BAM records={counts['bam_records_scanned']:,}/{total_records:,}")
    print(f"{sample_id}: {chrom}:{position} | {progress_label} "
          f"sites={counts['sites']:,}/{counts['total_sites']:,} read-site rows={counts['retained_read_site_pairs']:,} "
          f"elapsed={elapsed / 60:.1f} min", flush=True)


def extract_bam_features(site_list, sample_id, bam_path, reference_fasta, out,
                         flank=9, min_mapq=0, parquet_batch_rows=1024, parquet_file_rows=100000,
                         partition_by_locus=False, progress_seconds=60, bam_threads=1,
                         candidate_group_gap=None, read_batch_size=256, group_batch_size=1,
                         max_fetch_span=10000000,
                         max_reads=None, sampling_seed=42, site_sink=None):
    """Extract all targets together using one scan or indexed group batches.

    Consecutive groups on both strands share one fetch per chromosome batch.
    A CIGAR index and MM/ML decoder are shared within that fetch. Spanning
    alignments may be reprocessed in another batch; no read context is kept
    for a future fetch.
    Once the next read starts after a site's center, that site's complete bag
    can be written and released. Optional reservoir sampling bounds each bag;
    eligible_depth always counts all eligible reads before sampling.
    """
    try:
        import pysam
    except ImportError as exc:
        raise ImportError(f'Cannot import pysam: {exc}. Check the Python environment; '
                          'site-list preparation uses only stdlib.') from exc
    if max_reads is not None and (type(max_reads) is not int or max_reads < 1):
        raise ValueError('max_reads must be a positive integer or None')
    if type(sampling_seed) is not int or sampling_seed < 0:
        raise ValueError('sampling_seed must be a nonnegative integer')
    if flank < 0 or not 0 <= min_mapq <= 255:
        raise ValueError('Invalid feature extraction options')
    if not isinstance(bam_threads, int) or bam_threads < 1 or progress_seconds <= 0:
        raise ValueError('bam_threads and progress_seconds must be positive')
    if not isinstance(read_batch_size, int) or isinstance(read_batch_size, bool) or read_batch_size < 1:
        raise ValueError('read_batch_size must be a positive integer')
    if type(group_batch_size) is not int or group_batch_size < 1:
        raise ValueError('group_batch_size must be a positive integer')
    if candidate_group_gap is not None:
        candidate_groups([], candidate_group_gap, max_fetch_span)  # Validate even with no groups.
    require_arrow()
    read_schema = feature_schema(flank, MODIFICATIONS)
    if parquet_batch_rows < 1 or parquet_file_rows < 1:
        raise ValueError('Parquet row limits must be positive')
    site_list = Path(site_list)
    out = Path(out)
    by_chrom = load_sample_sites(site_list, sample_id, flank)
    total_sites = sum(len(sites) for sites in by_chrom.values())
    input_info = {name: file_info(path) for name, path in
                  [('site_list', site_list), ('bam', bam_path), ('reference', reference_fasta)]}
    with pysam.AlignmentFile(str(bam_path), 'rb', require_index=True, threads=bam_threads) as bam, pysam.FastaFile(str(reference_fasta)) as fasta:
        validate_reference_sites(by_chrom, bam, fasta)
        positions = {chrom: [s['start'] for s in sites] for chrom, sites in by_chrom.items()}
        groups_by_chrom = ({chrom: candidate_groups(sites, candidate_group_gap, max_fetch_span) for chrom, sites in by_chrom.items()}
                           if candidate_group_gap is not None else None)
        cursors = {chrom: 0 for chrom in by_chrom}
        active = {}  # Only unfinished bags; the bulk read Parquet is never held in memory.
        out.mkdir(parents=True, exist_ok=False)
        (out / 'INCOMPLETE.txt').write_text('Use only after run.json has status=complete.', encoding='utf-8')
        if site_sink is not None:
            read_writer, site_writer = site_sink, None
        else:
            writer_class = PartitionedParquetWriter if partition_by_locus else ShardedParquetWriter
            read_writer = writer_class(out / 'reads', read_schema,
                                       batch_rows=parquet_batch_rows, file_rows=parquet_file_rows)
            site_writer = writer_class(out / 'sites', site_schema(),
                                       batch_rows=parquet_batch_rows, file_rows=parquet_file_rows)
        counts = Counter({name: 0 for name in ('sites', 'retained_read_site_pairs', 'bam_records_scanned',
                         'coordinate_index_builds', 'modification_decodes', 'peak_active_sites',
                         'reservoir_admissions', 'encoded_read_site_windows')})
        counts['bam_records_total'] = bam.mapped + bam.unmapped
        counts['total_sites'] = total_sites
        if groups_by_chrom is not None:
            counts['candidate_groups_total'] = sum(len(groups) for groups in groups_by_chrom.values())
            counts['candidate_groups_completed'] = 0
            counts['bam_records_fetched'] = 0
            counts['bam_fetches'] = 0
            counts['read_batches'] = 0
            with (out / 'candidate_groups.tsv').open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow(['chrom', 'group', 'start', 'end', 'candidate_sites'])
                for chrom in sorted(groups_by_chrom, key=bam.get_tid):
                    for index, (first, stop) in enumerate(groups_by_chrom[chrom]):
                        writer.writerow([chrom, index, positions[chrom][first], positions[chrom][stop-1] + 1, stop-first])
        channels = Counter()
        started = last_progress = time.monotonic()
        try:
            with (out / 'site_summary.tsv').open('w', newline='', encoding='utf-8') as summary:
                writer = csv.writer(summary, delimiter='\t')
                writer.writerow(['row_id', 'eligible_depth', 'retained_depth', 'read_offset', 'central_deletions',
                                 'central_refskips', *[f'{m}_center_observed' for m in MODIFICATIONS]])

                def flush_completed_sites(chrom, next_start):
                    """No later alignment can cover an earlier center in a sorted BAM."""
                    cursor = cursors[chrom]
                    sites = by_chrom[chrom]
                    while cursor < len(sites) and sites[cursor]['start'] < next_start:
                        site = sites[cursor]
                        bag = active.pop((chrom, cursor), None)
                        write_site_features(site, bag, fasta, flank, read_writer, site_writer, writer, counts, site_sink, channels=channels)
                        sites[cursor] = None  # This coordinate cannot occur in later alignments.
                        cursor += 1
                    cursors[chrom] = cursor

                previous = (-1, -1)
                current_chrom = None
                unmapped_tail = False
                if groups_by_chrom is not None:
                    for chrom in sorted(groups_by_chrom, key=bam.get_tid):
                        current_chrom = chrom
                        groups = groups_by_chrom[chrom]
                        fetch_batches = candidate_group_batches(by_chrom[chrom], groups, group_batch_size, max_fetch_span)
                        for first, stop, group_count in fetch_batches:
                            start, end = positions[chrom][first], positions[chrom][stop-1] + 1
                            previous_start = -1
                            counts['bam_fetches'] += 1
                            for batch in read_batches(bam.fetch(chrom, start, end), read_batch_size):
                                counts['read_batches'] += 1
                                counts['bam_records_fetched'] += len(batch)
                                for read in batch:
                                    if read.reference_start < previous_start:
                                        raise ValueError('BAM records must be sorted by alignment start')
                                    previous_start = read.reference_start
                                    counts['bam_records_scanned'] += 1
                                    flush_completed_sites(chrom, read.reference_start)
                                    if is_eligible_read(read, min_mapq):
                                        collect_read_features(read, by_chrom[chrom], positions[chrom], active,
                                                              flank, counts, max_reads, sampling_seed,
                                                              site_range=(first, stop))
                                read = None  # The batch must not leave its last alignment alive.
                                if time.monotonic() - last_progress >= progress_seconds:
                                    report_extraction_progress(out, sample_id, chrom, start, counts, started, read_writer)
                                    last_progress = time.monotonic()
                                batch.clear()  # All short-window features own their values now.
                            flush_completed_sites(chrom, end)
                            if active:
                                raise RuntimeError('Read reservoirs must be empty after completing a fetch batch')
                            counts['candidate_groups_completed'] += group_count
                            if time.monotonic() - last_progress >= progress_seconds:
                                report_extraction_progress(out, sample_id, chrom, end, counts, started, read_writer)
                                last_progress = time.monotonic()
                        flush_completed_sites(chrom, float('inf'))
                        del by_chrom[chrom], positions[chrom], cursors[chrom]
                    scan = ()
                else:
                    scan = bam.fetch(until_eof=True)
                for read in scan:
                    counts['bam_records_scanned'] += 1
                    if read.reference_id < 0:
                        unmapped_tail = True
                        continue
                    key = read.reference_id, read.reference_start
                    if key < previous or unmapped_tail:
                        raise ValueError('BAM records must be sorted by reference and alignment start')
                    previous = key
                    chrom = read.reference_name
                    if current_chrom != chrom:
                        if current_chrom in by_chrom:
                            flush_completed_sites(current_chrom, float('inf'))
                            del by_chrom[current_chrom]
                            del positions[current_chrom]
                            del cursors[current_chrom]
                        current_chrom = chrom
                    if chrom not in by_chrom:
                        continue
                    flush_completed_sites(chrom, read.reference_start)
                    if is_eligible_read(read, min_mapq):
                        collect_read_features(read, by_chrom[chrom], positions[chrom], active,
                                              flank, counts, max_reads, sampling_seed)
                    if counts['bam_records_scanned'] % 1000 == 0 and time.monotonic() - last_progress >= progress_seconds:
                        report_extraction_progress(out, sample_id, chrom, read.reference_start,
                                                   counts, started, read_writer)
                        last_progress = time.monotonic()
                for chrom in sorted(by_chrom, key=bam.get_tid):
                    flush_completed_sites(chrom, float('inf'))
                strategy = 'indexed candidate groups' if groups_by_chrom is not None else 'one BAM pass'
                print(f"{sample_id}: extracted {counts['sites']:,} sites using {strategy}", flush=True)
            active.clear()
            by_chrom.clear()
            positions.clear()
            cursors.clear()
            read_writer.close()
            if site_writer is not None:
                site_writer.close()
            if counts['sites'] != total_sites:
                raise ValueError('Not all candidate sites were written')
            for name, path in [('site_list', site_list), ('bam', bam_path), ('reference', reference_fasta)]:
                if file_info(path) != input_info[name]:
                    raise ValueError(f'{name} input changed during feature extraction')
            report = build_extraction_report(site_list, sample_id, bam_path, reference_fasta, bam,
                flank=flank, min_mapq=min_mapq, bam_threads=bam_threads,
                parquet_batch_rows=parquet_batch_rows, parquet_file_rows=parquet_file_rows,
                parquet_files=(dict(sites=site_sink.files) if site_sink is not None else
                               dict(reads=read_writer.files, sites=site_writer.files)), counts=counts,
                channels=channels, partition_by_locus=partition_by_locus,
                read_ranges=read_writer.read_ranges if partition_by_locus and site_sink is None else None,
                candidate_group_gap=candidate_group_gap, read_batch_size=read_batch_size,
                group_batch_size=group_batch_size, max_fetch_span=max_fetch_span)
            report['sampling_feature_order'] = 'final_reservoir_then_mm_features_v1'
            report['policies']['deferred_reads'] = 'reservoirs hold shared alignment/CIGAR references; final selected reads alone receive MM/ML, sequence, quality and feature conversion'
            report['max_reads'] = max_reads
            report['sampling_seed'] = sampling_seed
            report['policies']['modification_memory'] = 'decode only final-selected alignments; dictionary keeps CIGAR-window query positions; shared across final sites'
            report['policies']['numeric_cache'] = 'bounded 8192-entry value cache; no whole-BAM alignment or genomic coordinate cache'
            if max_reads is not None:
                report['policies']['sampling'] = 'uniform per-site reservoir sampling without replacement; no padding stored'
                report['policies']['center_observed_scope'] = 'retained reads only; unsampled site AF/call counts remain bedMethyl metadata'
                report['policies']['observed_mm_groups_scope'] = 'alignments retained in at least one final site reservoir; each decoded once; not a full-read census'
            if site_sink is not None:
                saved_schema = getattr(site_sink, 'schema', None)
                report['schema'] = (saved_schema.metadata[b'schema_version'].decode()
                                    if saved_schema is not None else 'site_bags_parquet_v3')
                report['partition_by_locus'] = True
                report['read_offset_mode'] = 'nested per-site read bags; no flat read offsets'
                report['policies']['storage'] = 'one row per site; only retained real reads; no padding'
                report['policies']['repeat_mask'] = ('binary lowercase acgt from the same reference FASTA window; '
                    'N/padding false; RNA orientation; stored with sequence and read tensors')
            if candidate_group_gap is not None:
                report['policies']['read_conversion'] = 'one CIGAR/MM conversion per relevant alignment within each fetch batch; spanning alignments may be reprocessed in later batches'
                report['policies']['group_memory'] = 'current fetch-batch reservoirs only; completed sites released immediately; no read context kept for future fetches or completed-site feature cache'
                report['policies']['group_reads'] = 'coordinate-sorted groups merge both strands/targets; one contiguous indexed fetch per chromosome batch including intervening gaps'
                report['policies']['fetch_limit'] = 'split at chromosome boundary, group_batch_size, or max_fetch_span; even a single oversized group is split by candidate centers'
                report['policies']['observed_mm_groups_scope'] = 'final-selected alignment union within each fetch batch; spanning alignments may be counted in multiple batches; not a full-read census'
            (out / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            (out / 'INCOMPLETE.txt').unlink()
            report_extraction_progress(out, sample_id, current_chrom, None, counts, started, read_writer, complete=True)
            return report
        except BaseException as exc:
            read_writer.abort()
            if site_writer is not None:
                site_writer.abort()
            (out / 'FAILED.txt').write_text(str(exc), encoding='utf-8')
            raise
