"""Streaming ML distributions and advisory confidence thresholds.

This uses the maximum of canonical and all expected same-base modification
probabilities. It does not change calls, labels, BAM files, or NN thresholds.
"""
from array import array
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import re

from utils.features.bam import CANONICAL, MOD_CODES, MODIFICATIONS
from utils.io.artifacts import file_info
from utils.io.cigar import CigarIndex

BASES = 'ACGT'
PROBABILITY_UNITS = 512  # ML midpoint (N + 0.5) / 256 = (2N + 1) / 512.
MM_GROUP = re.compile(r'([ACGTUN])([+-])([A-Za-z]+|[0-9]+)([.?]?)((?:,[0-9]+)*)')
EXPECTED_CHANNELS = {
    base: {modification for modification in MODIFICATIONS if CANONICAL[modification] == base}
    for base in BASES
}


def histogram_quantile(histogram, percentile):
    """Empirical inverse CDF: the first bin reaching ceil(percentile * count)."""
    total = sum(histogram)
    if not total:
        return None
    rank = max(1, math.ceil(percentile * total))
    cumulative = 0
    for index, count in enumerate(histogram):
        cumulative += count
        if cumulative >= rank:
            return index
    raise ValueError('Invalid histogram')


def sample_read(read_name, fraction, seed):
    """Name-hash sampling is reproducible and independent of BAM record order."""
    if fraction == 1:
        return True
    identity = f'{seed}\0{read_name}'.encode('utf-8')
    rank = int.from_bytes(hashlib.sha256(identity).digest()[:8], 'big')
    return rank < fraction * 2**64


def read_probability_channels(read):
    """Return declared channels, explicit scores in original query coordinates,
    and bases whose unknown modification codes prevent canonical inference.
    """
    mm_tags = [tag for tag in ('MM', 'Mm') if read.has_tag(tag)]
    ml_tags = [tag for tag in ('ML', 'Ml') if read.has_tag(tag)]
    if len(mm_tags) > 1 or len(ml_tags) > 1:
        raise ValueError(f'{read.query_name}: ambiguous modification tag aliases')
    if not mm_tags:
        return {}, {}, set()
    if read.has_tag('MN') and read.get_tag('MN') != read.query_length:
        raise ValueError(f'{read.query_name}: MN/SEQ length mismatch')
    sequence = read.get_forward_sequence()
    if sequence is None:
        raise ValueError(f'{read.query_name}: MM/ML without SEQ')
    sequence = sequence.upper().replace('U', 'T')
    scores = read.get_tag(ml_tags[0]) if ml_tags else None
    if scores is not None and (not isinstance(scores, array) or scores.typecode != 'B'):
        raise ValueError(f'{read.query_name}: ML must be a B:C byte array')
    declarations = {}
    unsupported_bases = set()
    calls = {}
    base_positions = {}
    ml_offset = 0
    text = read.get_tag(mm_tags[0])
    if text and not text.endswith(';'):
        raise ValueError(f'{read.query_name}: MM groups must end with a semicolon')
    for group in text.split(';')[:-1]:
        parsed = MM_GROUP.fullmatch(group)
        if parsed is None:
            raise ValueError(f'{read.query_name}: malformed MM group')
        base, strand, codes, mode, delta_text = parsed.groups()
        base = base.replace('U', 'T')
        codes = [codes] if codes.isdigit() else list(codes)
        if strand == '+':
            for code in codes:
                modification = MOD_CODES.get(code)
                if modification is None or base != CANONICAL[modification]:
                    unsupported_bases.update(BASES if base == 'N' else base)
                    continue
                channels = declarations.setdefault(base, {})
                mode = mode or '.'
                if modification in channels and channels[modification] != mode:
                    raise ValueError(f'{read.query_name}: conflicting MM skip modes')
                channels[modification] = mode
        deltas = list(map(int, delta_text.split(',')[1:]))
        if deltas and base not in base_positions:
            base_positions[base] = range(len(sequence)) if base == 'N' else [
                query for query, nucleotide in enumerate(sequence) if nucleotide == base]
        occurrence = -1
        for delta in deltas:
            occurrence += delta + 1
            if occurrence >= len(base_positions[base]):
                raise ValueError(f'{read.query_name}: MM position outside SEQ')
            query = base_positions[base][occurrence]
            for code in codes:
                if scores is not None and ml_offset >= len(scores):
                    raise ValueError(f'{read.query_name}: MM/ML entry counts differ')
                modification = MOD_CODES.get(code)
                if strand == '+' and modification is not None:
                    key = query, modification
                    if key in calls:
                        raise ValueError(f'{read.query_name}: duplicate modification probability')
                    calls[key] = int(scores[ml_offset]) if scores is not None else None
                ml_offset += 1
    if scores is not None and ml_offset != len(scores):
        raise ValueError(f'{read.query_name}: MM/ML entry counts differ')
    return declarations, calls, unsupported_bases


def position_probabilities(query, channels, calls):
    """Probability units for every declared channel; None means unknown.

    Explicit values use SAM interval midpoints. Omitted '.' positions have an
    inferred modification probability of zero; '?' positions remain unknown.
    """
    probabilities = {}
    for modification, mode in channels.items():
        key = query, modification
        if key in calls:
            score = calls[key]
            if score is None:
                return None
            probabilities[modification] = 2 * score + 1
        elif mode == '.':
            probabilities[modification] = 0
        else:
            return None
    return probabilities


def collect_read_distributions(read, raw, confidence, winners, counts, only_aligned):
    """Decode one sampled read and update fixed-size histograms."""
    declarations, calls, unsupported_bases = read_probability_channels(read)
    counts['reads_with_unsupported_channels'] += bool(unsupported_bases)
    if not declarations and not calls:
        counts['reads_without_supported_channels'] += 1
        return
    sequence = read.get_forward_sequence().upper().replace('U', 'T')
    coordinates = CigarIndex(read) if only_aligned else None
    aligned = {}

    def position_is_included(query):
        if coordinates is None:
            return True
        if query not in aligned:
            stored_query = len(sequence) - 1 - query if read.is_reverse else query
            aligned[query] = coordinates.query_to_reference(stored_query) is not None
        return aligned[query]

    for (query, modification), score in calls.items():
        if not position_is_included(query):
            continue
        if score is None:
            counts['explicit_unknown_scores'] += 1
        else:
            raw[modification][score] += 1
            counts['observed_ml_values'] += 1

    for query, base in enumerate(sequence):
        if base not in BASES or not position_is_included(query):
            continue
        if base in unsupported_bases:
            counts['positions_with_unsupported_channels'] += 1
            continue
        channels = declarations.get(base, {})
        if set(channels) != EXPECTED_CHANNELS[base]:
            counts['positions_with_incomplete_channels'] += 1
            continue
        probabilities = position_probabilities(query, channels, calls)
        if probabilities is None:
            counts['positions_with_unknown_probabilities'] += 1
            continue
        # If even the lower ends of the ML intervals exceed one, no valid
        # canonical residual can be recovered. Do not silently renormalize.
        lower_sum = sum(value - 1 for value in probabilities.values() if value)
        if lower_sum > PROBABILITY_UNITS:
            counts['positions_with_probability_sum_above_one'] += 1
            continue
        midpoint_sum = sum(probabilities.values())
        if midpoint_sum > PROBABILITY_UNITS:
            counts['positions_with_quantization_clamp'] += 1
        canonical = max(0, PROBABILITY_UNITS - midpoint_sum)
        winning_state = 'canonical'
        winning_probability = canonical
        for modification, probability in probabilities.items():
            if probability > winning_probability:
                winning_state = modification
                winning_probability = probability
        confidence[base][winning_probability] += 1
        winners[base, winning_state][winning_probability] += 1
        counts['confidence_positions'] += 1


def write_histogram_table(path, histograms, columns, denominator, midpoint=False):
    """Write each bin, its proportion, and the cumulative empirical distribution."""
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow([*columns, 'bin', 'probability', 'count', 'fraction', 'cumulative_fraction'])
        for key, histogram in histograms.items():
            keys = key if isinstance(key, tuple) else (key,)
            total = sum(histogram)
            cumulative = 0
            for value, count in enumerate(histogram):
                cumulative += count
                probability = (value + 0.5) / denominator if midpoint else value / denominator
                fraction = count / total if total else ''
                cumulative_fraction = cumulative / total if total else ''
                writer.writerow([*keys, value, probability, count, fraction, cumulative_fraction])


def plot_distributions(path, raw, confidence, thresholds):
    """Plot measured ML distributions and canonical-inclusive confidence CDFs."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(4, 3, figsize=(15, 13))
    for axis, (modification, histogram) in zip(axes.flat, raw.items()):
        probabilities = [(value + 0.5) / 256 for value in range(256)]
        axis.bar(probabilities, histogram, width=1 / 256)
        axis.set_title(f'{modification}: explicit ML (n={sum(histogram):,})')
        axis.set_xlabel('Modification probability interval midpoint')
        axis.set_ylabel('Count')
        axis.set_xlim(0, 1)
    for axis, base in zip(list(axes.flat)[8:], BASES):
        histogram = confidence[base]
        probabilities = [value / PROBABILITY_UNITS for value in range(len(histogram))]
        axis.bar(probabilities, histogram, width=1 / PROBABILITY_UNITS)
        axis.set_title(f'{base}: winning-state confidence (n={sum(histogram):,})')
        axis.set_xlabel('Maximum canonical / modification probability')
        axis.set_ylabel('Count')
        axis.set_xlim(0, 1)
        total = sum(histogram)
        if total:
            cumulative = 0
            cdf = []
            for count in histogram:
                cumulative += count
                cdf.append(cumulative / total)
            cdf_axis = axis.twinx()
            cdf_axis.plot(probabilities, cdf, color='green', linewidth=1, label='CDF')
            cdf_axis.set_ylim(0, 1)
            cdf_axis.set_ylabel('CDF', color='green')
        if thresholds[base] is not None:
            axis.axvline(thresholds[base], color='red', linestyle='--', label=f"Threshold {thresholds[base]:.3f}")
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def estimate_quality(bam_path, out, *, percentile=0.1, sample_fraction=1.0, seed=42, only_aligned=True):
    """Inspect BAM probabilities and propose thresholds without modifying calls."""
    import pysam
    import matplotlib  # Fail before scanning a large BAM if plotting is unavailable.
    matplotlib.use('Agg')
    if not 0 <= percentile <= 1 or not 0 < sample_fraction <= 1:
        raise ValueError('Use percentile in 0..1 and sample_fraction in (0,1]')
    if not isinstance(seed, int) or not isinstance(only_aligned, bool):
        raise ValueError('seed must be an integer and only_aligned a boolean')
    bam_path = Path(bam_path).resolve()
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    source = file_info(bam_path)
    raw = {modification: [0] * 256 for modification in MODIFICATIONS}
    confidence = {base: [0] * (PROBABILITY_UNITS + 1) for base in BASES}
    winners = {}
    for base in BASES:
        winners[base, 'canonical'] = [0] * (PROBABILITY_UNITS + 1)
    for modification in MODIFICATIONS:
        winners[CANONICAL[modification], modification] = [0] * (PROBABILITY_UNITS + 1)
    counts = Counter(records_scanned=0, eligible_reads=0, sampled_reads=0,
                     observed_ml_values=0, confidence_positions=0)
    out.mkdir(parents=True)
    (out / 'INCOMPLETE.txt').write_text('Quality distributions are not complete.', encoding='utf-8')
    try:
        with pysam.AlignmentFile(str(bam_path), 'rb') as bam:
            for read in bam.fetch(until_eof=True):
                counts['records_scanned'] += 1
                if read.is_secondary or read.is_supplementary or read.is_duplicate or read.is_qcfail:
                    continue
                if only_aligned and read.is_unmapped:
                    continue
                if not read.query_sequence:
                    continue
                counts['eligible_reads'] += 1
                if not sample_read(read.query_name, sample_fraction, seed):
                    continue
                counts['sampled_reads'] += 1
                collect_read_distributions(read, raw, confidence, winners, counts, only_aligned)
        if file_info(bam_path) != source:
            raise ValueError('Input BAM changed during quality estimation')
        filter_thresholds = {}
        raw_quantiles = {}
        for base, histogram in confidence.items():
            value = histogram_quantile(histogram, percentile)
            filter_thresholds[base] = None if value is None else value / PROBABILITY_UNITS
        for modification, histogram in raw.items():
            value = histogram_quantile(histogram, percentile)
            raw_quantiles[modification] = None if value is None else (value + 0.5) / 256
        report = dict(status='complete', schema='modification_quality_distributions_v1',
                      source=source, percentile=percentile, sample_fraction=sample_fraction, seed=seed,
                      only_aligned=only_aligned, counts=dict(counts), pysam_version=pysam.__version__,
                      filter_thresholds=filter_thresholds,
                      inherited_modification_thresholds={
                          modification: filter_thresholds[CANONICAL[modification]] if sum(raw[modification]) else None
                          for modification in MODIFICATIONS},
                      raw_channel_quantiles=raw_quantiles,
                      method='empirical inverse CDF of max(canonical, all expected same-base modification probabilities)',
                      policies=dict(ml='SAM interval midpoint (byte+0.5)/256; raw histograms contain explicit ML values only',
                          channels='all eight supported channels retained; confidence requires every expected same-base channel to be declared in that read',
                          inherited_thresholds='base confidence thresholds inherited by observed modifications, not independently optimized modification thresholds; unobserved modifications remain null',
                          missing="MM '?' omitted or listed positions without ML are unknown; MM '.'/unflagged omitted positions infer modification zero",
                          invalid='unknown plus-strand modification codes block base confidence; probability lower-bound sum>1 excluded; midpoint rounding residual clamped to zero',
                          ties='canonical wins an exact tie, otherwise declaration order; values equal to threshold pass',
                          sampling='SHA256(seed + read name) selects a fraction across the entire BAM; record-order independent',
                          scope='advisory only; does not modify existing bedMethyl AF, labels, model scores or BAM calls',
                          compatibility='not a bit-for-bit replacement for modkit sampling or filtering'),
                      references=['https://nanoporetech.github.io/modkit/filtering_numeric_details.html',
                                  'https://samtools.github.io/hts-specs/SAMtags.pdf'])
        write_histogram_table(out / 'raw_ml_histograms.tsv', raw, ['modification'], 256, midpoint=True)
        write_histogram_table(out / 'confidence_histograms.tsv', confidence, ['base'], PROBABILITY_UNITS)
        write_histogram_table(out / 'winner_histograms.tsv', winners, ['base', 'winning_state'], PROBABILITY_UNITS)
        plot_distributions(out / 'histograms.png', raw, confidence, filter_thresholds)
        (out / 'thresholds.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        (out / 'run.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as error:
        (out / 'FAILED.txt').write_text(str(error), encoding='utf-8')
        raise
