"""Zero only rejected modification ML bytes, keeping alignments and MM unchanged."""
from array import array
import bisect
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import sqlite3

from utils.io.artifacts import completed_run, file_info
from utils.io.cigar import CigarIndex
from utils.references.catalog import sha256
from prediction.evidence import evidence_values

TARGET_CODES = {'a': 'm6A', 'm': 'm5C', '17596': 'inosine', '17802': 'pseU'}
_MM_GROUP = re.compile(r'([ACGTUN])([+-])([a-z]+|[0-9]+)[.?]?((?:,[0-9]+)*)')


def read_site_decision(row):
    """Validate one prediction and return its exact site key and final decision."""
    start = int(row['start'])
    if (start < 0 or int(row['end']) != start + 1 or row['strand'] not in ('+', '-')
            or row['modification'] not in TARGET_CODES.values() or not row['chrom']):
        raise ValueError('Predictions must use single-base BED0 sites for the four targets')
    site = row['chrom'], start, row['strand'], row['modification']
    if row['final_keep'] not in ('0', '1') or row['rescued'] not in ('0', '1'):
        raise ValueError('Prediction final_keep/rescued must be 0 or 1')
    if row['rescued'] == '1' and row['final_keep'] != '1':
        raise ValueError('A rescued site must be retained')
    if 'flag_mettl16' in row:
        flags = evidence_values(row)
        if row['final_keep'] != str(int('1' in flags)):
            raise ValueError('Final decision must agree with independent evidence flags')
        if (row['modification'] in ('m5C','pseU') and '1' in flags[5:]
                and row.get('auxiliary_motif_min_depth') not in (None,'')):
            from prediction.auxiliary import auxiliary_motif_eligible
            if not auxiliary_motif_eligible(row,int(row['auxiliary_motif_min_depth']),float(row['auxiliary_motif_min_af'])):
                raise ValueError('Motif-retained m5C/psi decision violates its recorded auxiliary motif thresholds')
    if row['prediction_status'] == 'reference_only':
        if (row['modification'] not in ('m5C', 'pseU') or row['keep_score'] or row['nn_keep']
                or row['rescued'] != '0'):
            raise ValueError('Reference-only decisions require m5C/psi and no neural score')
        return site, 'retained' if row['final_keep'] == '1' else 'rejected'
    if row['prediction_status'] != 'predicted':
        if row['prediction_status'] not in ('no_reads', 'no_prediction') or row['keep_score']:
            raise ValueError('Unknown prediction status or score on an unpredicted site')
        return site, 'unpredicted'
    score = float(row['keep_score'])
    if not math.isfinite(score) or not 0 <= score <= 1 or row['nn_keep'] not in ('0', '1'):
        raise ValueError('Predicted sites need a finite score and a binary NN decision')
    if row.get('nn_class'):
        from utils.adenine import CLASSES
        if (row['modification'] not in ('m6A', 'inosine') or row['nn_class'] not in CLASSES
                or row['nn_keep'] != str(int(row['nn_class'] == row['modification']))):
            raise ValueError('Joint class and NN decision disagree')
        flags = evidence_values(row)
        if row['final_keep'] != str(int('1' in flags)):
            raise ValueError('Final decision must agree with independent evidence flags')
        return site, 'retained' if row['final_keep'] == '1' else 'rejected'
    if row['final_keep'] == '0':
        if row['nn_keep'] != '0' or row['rescued'] != '0':
            raise ValueError('Only an NN rejection surviving reference rescue may be zeroed')
        return site, 'rejected'
    return site, 'retained'


def load_rejected_sites(prediction_directories, sample_id):
    """Hash exact BED0/strand/target keys after rescue; unknown scores are kept."""
    if not sample_id:
        raise ValueError('Set an explicit sample_id for BAM output')
    rejected = {}
    seen_sites = set()
    sources = []
    counts = Counter(other_sample_rows=0, unpredicted_sites_preserved=0,
                     retained_sites_preserved=0, rejected_sites=0)
    for directory in prediction_directories:
        directory = Path(directory).resolve()
        completed_run(directory)
        path = directory / 'all_predictions.tsv'
        sources.append(dict(**file_info(path), sha256=sha256(path),
                            run_sha256=sha256(directory / 'run.json')))
        with path.open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                if row['sample_id'] != sample_id:
                    counts['other_sample_rows'] += 1
                    continue
                site, decision = read_site_decision(row)
                if site in seen_sites:
                    raise ValueError(f'Duplicate sample/site/target prediction: {site}')
                seen_sites.add(site)
                if decision == 'rejected':
                    rejected[site] = dict(observed=0, changed=0, already_zero=0)
                    counts['rejected_sites'] += 1
                elif decision == 'unpredicted':
                    counts['unpredicted_sites_preserved'] += 1
                else:
                    counts['retained_sites_preserved'] += 1
    if not sources or (not seen_sites and counts['other_sample_rows']):
        raise ValueError('No completed predictions for the requested sample')
    return rejected, sources, counts


def prediction_tables(directory):
    """Use completed chromosome/strand tables, or an older combined table."""
    directory = Path(directory).resolve()
    run = completed_run(directory)
    if not run.get('partitions'):
        return [(directory / 'all_predictions.tsv', None)]
    tables = []
    for partition in run['partitions']:
        path = (directory / partition['path']).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('Prediction partition path must stay inside its output directory')
        tables.append((path, (partition['chrom'], partition['strand'])))
    return tables


def build_decision_store(prediction_directories, sample_id, database):
    """Validate predictions once, retaining whole-genome decisions on disk."""
    if not sample_id:
        raise ValueError('Set an explicit sample_id for BAM output')
    database.execute('CREATE TABLE decisions (chrom TEXT, start INTEGER, strand TEXT, target TEXT, '
                     'decision TEXT, observed INTEGER DEFAULT 0, changed INTEGER DEFAULT 0, '
                     'already_zero INTEGER DEFAULT 0, PRIMARY KEY(chrom,start,strand,target))')
    counts = Counter(other_sample_rows=0, unpredicted_sites_preserved=0,
                     retained_sites_preserved=0, rejected_sites=0)
    sources = []
    for directory in prediction_directories:
        directory = Path(directory).resolve()
        run_hash = sha256(directory / 'run.json')
        for path, partition in prediction_tables(directory):
            sources.append(dict(**file_info(path), sha256=sha256(path), run_sha256=run_hash))
            with path.open(encoding='utf-8', newline='') as handle:
                for row in csv.DictReader(handle, delimiter='\t'):
                    if partition is not None and (row['chrom'], row['strand']) != partition:
                        raise ValueError('Prediction row disagrees with chromosome/strand partition')
                    if row['sample_id'] != sample_id:
                        counts['other_sample_rows'] += 1
                        continue
                    site, decision = read_site_decision(row)
                    try:
                        database.execute('INSERT INTO decisions(chrom,start,strand,target,decision) VALUES (?,?,?,?,?)',
                                         (*site, decision))
                    except sqlite3.IntegrityError as error:
                        raise ValueError(f'Duplicate sample/site/target prediction: {site}') from error
                    metric = {'rejected': 'rejected_sites', 'retained': 'retained_sites_preserved',
                              'unpredicted': 'unpredicted_sites_preserved'}[decision]
                    counts[metric] += 1
    if not sources or (not sum(counts[key] for key in ('rejected_sites', 'retained_sites_preserved',
                                                      'unpredicted_sites_preserved')) and counts['other_sample_rows']):
        raise ValueError('No completed predictions for the requested sample')
    database.commit()
    return sources, counts


def load_chromosome_rejections(database, chrom):
    """Only this chromosome's rejected sites enter the in-memory hash."""
    rows = database.execute('SELECT start,strand,target FROM decisions WHERE chrom=? AND decision=?',
                            (chrom, 'rejected'))
    return {(chrom, start, strand, target): dict(observed=0, changed=0, already_zero=0)
            for start, strand, target in rows}


def save_chromosome_audit(database, rejected):
    database.executemany('UPDATE decisions SET observed=?,changed=?,already_zero=? '
                         'WHERE chrom=? AND start=? AND strand=? AND target=?',
        [(counts['observed'], counts['changed'], counts['already_zero'], *site)
         for site, counts in rejected.items()])


def modification_entries(read, mm_tag, ml_length):
    """Return (ML offset, stored-query position, target) in raw MM/ML order.

    Multi-code groups interleave probabilities per position. Opposite-strand
    groups and other modifications still consume ML entries, but are not edited.
    Positions count the original sequenced orientation, including on FLAG 0x10.
    """
    sequence = read.get_forward_sequence()
    if sequence is None:
        raise ValueError(f'{read.query_name}: MM/ML without SEQ')
    if read.has_tag('MN') and read.get_tag('MN') != len(sequence):
        raise ValueError(f'{read.query_name}: MN/SEQ length mismatch')
    sequence = sequence.upper().replace('U', 'T')
    text = read.get_tag(mm_tag)
    if text and not text.endswith(';'):
        raise ValueError(f'{read.query_name}: MM must terminate each group with a semicolon')
    entries = []
    base_positions = {}
    group_ml_offset = 0
    groups = text.split(';')[:-1]
    for group in groups:
        match = _MM_GROUP.fullmatch(group)
        if match is None:
            raise ValueError(f'{read.query_name}: malformed MM group: {group}')
        base, strand, code_text, delta_text = match.groups()
        if code_text.isdigit():
            codes = [code_text]
        else:
            codes = list(code_text)
        deltas = list(map(int, delta_text.split(',')[1:]))
        targets = []
        if strand == '+':
            for code_index, code in enumerate(codes):
                if code in TARGET_CODES:
                    targets.append((code_index, TARGET_CODES[code]))
        if targets and deltas:
            base = base.replace('U', 'T')
            if base not in base_positions:
                if base == 'N':
                    base_positions[base] = range(len(sequence))
                else:
                    base_positions[base] = [i for i, value in enumerate(sequence) if value == base]
            occurrence = -1
            for event_index, delta in enumerate(deltas):
                occurrence += delta + 1
                if occurrence >= len(base_positions[base]):
                    raise ValueError(f'{read.query_name}: MM position outside SEQ')
                query_position = base_positions[base][occurrence]
                if read.is_reverse:
                    query_position = len(sequence) - 1 - query_position
                event_ml_offset = group_ml_offset + event_index * len(codes)
                for code_index, target in targets:
                    entries.append((event_ml_offset + code_index, query_position, target))
        group_ml_offset += len(deltas) * len(codes)
    if group_ml_offset != ml_length:
        raise ValueError(f'{read.query_name}: MM/ML entry counts differ')
    return entries


def zero_rejected_modifications(read, rejected, chrom):
    """Decode this read's CIGAR again for export, then edit only rejected ML bytes."""
    mm_tags = [tag for tag in ('MM', 'Mm') if read.has_tag(tag)]
    ml_tags = [tag for tag in ('ML', 'Ml') if read.has_tag(tag)]
    if not mm_tags or not ml_tags:
        return 0, 0
    if len(mm_tags) != 1 or len(ml_tags) != 1:
        raise ValueError(f'{read.query_name}: ambiguous uppercase/lowercase modification tags')
    ml_tag = ml_tags[0]
    raw = read.get_tag(ml_tag)
    if not isinstance(raw, array) or raw.typecode != 'B':
        raise ValueError(f'{read.query_name}: ML must be a B:C byte array')
    entries = modification_entries(read, mm_tags[0], len(raw))
    if not entries:
        return 0, 0
    cigar = CigarIndex(read)
    strand = '-' if read.is_reverse else '+'
    positions = {}
    changed = 0
    observed = 0
    for offset, q, target in entries:
        if q not in positions:
            positions[q] = cigar.query_to_reference(q)
        position = positions[q]
        if position is None:
            continue  # Insertions/soft clips have no genomic locus.
        site = rejected.get((chrom, position, strand, target))
        if site is None:
            continue
        site['observed'] += 1
        observed += 1
        if raw[offset]:
            raw[offset] = 0
            site['changed'] += 1
            changed += 1
        else:
            site['already_zero'] += 1
    if changed:
        read.set_tag(ml_tag, raw)
    return observed, changed


def copy_bam_with_filtered_scores(bam_path, output_path, database, counts, threads):
    """Copy every record in order, loading rejected sites one chromosome at a time."""
    import pysam
    with pysam.AlignmentFile(str(bam_path), 'rb', threads=threads) as source:
        if source.header.to_dict().get('HD', {}).get('SO') != 'coordinate':
            raise ValueError('Use a coordinate-sorted input BAM')
        contigs = {row[0] for row in database.execute('SELECT DISTINCT chrom FROM decisions WHERE decision=?', ('rejected',))}
        missing_contigs = contigs - set(source.references)
        if missing_contigs:
            raise ValueError(f'Prediction contigs absent from BAM: {sorted(missing_contigs)}')
        use_csi = any(length >= 2**29 for length in source.lengths)
        current_chrom, rejected, positions = None, {}, {}
        with pysam.AlignmentFile(str(output_path), 'wb', template=source, threads=threads) as dest:
            for read in source.fetch(until_eof=True):
                counts['records_written'] += 1
                if not read.is_unmapped and read.reference_end is not None:
                    chrom = source.get_reference_name(read.reference_id)
                    if chrom != current_chrom:
                        save_chromosome_audit(database, rejected)
                        rejected = load_chromosome_rejections(database, chrom)
                        positions = {'+': set(), '-': set()}
                        for _, start, strand, _ in rejected:
                            positions[strand].add(start)
                        positions = {strand: sorted(values) for strand, values in positions.items()}
                        current_chrom = chrom
                        counts['chromosomes_processed'] += 1
                        counts['peak_rejected_sites_per_chromosome'] = max(
                            counts['peak_rejected_sites_per_chromosome'], len(rejected))
                    strand = '-' if read.is_reverse else '+'
                    candidates = positions.get(strand, ())
                    first = bisect.bisect_left(candidates, read.reference_start)
                    if first < len(candidates) and candidates[first] < read.reference_end:
                        observed, changed = zero_rejected_modifications(read, rejected, chrom)
                        counts['rejected_ml_entries_observed'] += observed
                        counts['ml_entries_changed_to_zero'] += changed
                        counts['records_changed'] += bool(changed)
                dest.write(read)
        save_chromosome_audit(database, rejected)
        database.commit()
    return use_csi


def create_bam_index(bam_path, threads, use_csi):
    """Create BAI, or CSI when a contig exceeds the BAI coordinate range."""
    import pysam
    arguments = ['-@', str(threads)]
    suffix = '.bai'
    if use_csi:
        arguments.append('-c')
        suffix = '.csi'
    pysam.index(*arguments, str(bam_path))  # This also checks coordinate order.
    return Path(str(bam_path) + suffix)


def write_site_audit(path, database):
    """Keep per-site observed, changed and already-zero ML counts visible."""
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['chrom', 'start', 'end', 'strand', 'modification',
                         'ml_entries_observed', 'ml_entries_changed_to_zero', 'ml_entries_already_zero'])
        rows = database.execute('SELECT chrom,start,strand,target,observed,changed,already_zero '
                                'FROM decisions WHERE decision=? ORDER BY chrom,start,strand,target', ('rejected',))
        for chrom, start, strand, target, observed, changed, already_zero in rows:
            writer.writerow([chrom, start, start + 1, strand, target, observed, changed, already_zero])


def write_filtered_bam(bam_path, prediction_directories, sample_id, out, *, threads=2):
    """Load decisions, copy BAM with ML edits, then write its index and audit.

    Export reads the BAM again and builds its own CIGAR index. All mapped
    alignments are eligible, including supplementary/secondary/duplicate records.
    Unmapped records, MM '-' groups, rescued and unpredicted sites retain ML.
    """
    if not isinstance(threads, int) or threads < 1:
        raise ValueError('threads must be positive')
    bam_path = Path(bam_path).resolve()
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    source_bam = file_info(bam_path)
    out.mkdir(parents=True)
    (out / 'INCOMPLETE.txt').write_text('Filtered BAM is not complete.', encoding='utf-8')
    partial = out / 'filtered.partial.bam'
    result = out / 'filtered.bam'
    database_path = out / 'decisions.sqlite'
    database = sqlite3.connect(str(database_path))
    try:
        sources, counts = build_decision_store(prediction_directories, sample_id, database)
        counts.update(records_written=0, records_changed=0, rejected_ml_entries_observed=0,
                      ml_entries_changed_to_zero=0, chromosomes_processed=0, peak_rejected_sites_per_chromosome=0)
        use_csi = copy_bam_with_filtered_scores(bam_path, partial, database, counts, threads)
        if file_info(bam_path) != source_bam:
            raise ValueError('Input BAM changed during filtering')
        for entry in sources:
            expected = {key: entry[key] for key in ('path', 'size_bytes', 'mtime_ns')}
            if file_info(entry['path']) != expected:
                raise ValueError('Predictions changed during BAM filtering')
        partial_index = create_bam_index(partial, threads, use_csi)
        partial.rename(result)
        final_index = Path(str(result) + partial_index.suffix)
        partial_index.rename(final_index)
        write_site_audit(out / 'site_audit.tsv', database)
        report = dict(status='complete', schema='filtered_modification_bam_v1', sample_id=sample_id,
                      input_bam=source_bam, prediction_sources=sources, counts=dict(counts),
                      output_bam=file_info(result), output_index=file_info(final_index), threads=threads,
                      policies=dict(modification='explicit ML entries at NN- or policy-rejected sites become byte 0',
                                    probability='ML byte 0 denotes interval [0,1/256), not base-call Phred quality',
                                    match='exact GRCh38 BED0 chromosome/start/strand/target; no coordinate tolerance',
                                    missing='no_prediction/no_reads are preserved, never treated as negative evidence',
                                    alignments='all mapped records, including secondary/supplementary/duplicate; input order retained',
                                    tags='MM, MN, SEQ, QUAL and all other modification channels retained; MM minus-strand groups retained',
                                    memory='SQLite decision index; only one chromosome rejected-site hash in memory',
                                    mapping='one CIGAR index per applicable alignment; no intron-sized coordinate arrays'))
        (out / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as error:
        (out / 'FAILED.txt').write_text(str(error), encoding='utf-8')
        raise
    finally:
        database.close()
        database_path.unlink(missing_ok=True)
