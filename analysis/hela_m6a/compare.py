"""HeLa m6A set comparisons; overlaps are descriptive, not truth labels."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.io.artifacts import file_info
from utils.io.bed import load_bed_sites
from utils.io.excel import verify_workbook, write_workbook
from utils.references.catalog import sha256
from utils.references.genome import IndexedFasta, drach
from utils.references.transcripts import load_transcript_coordinates

PATHS = ('fasta', 'annotation', 'transcript_fasta', 'm6anet_sites', 'm6anet_reference_hashes',
         'predictions', 'prediction_run', 'glori', 'glori_provenance', 'coverage', 'coverage_run',
         'coverage_sources', 'previous_workbook', 'output', 'figures')
REGIONS = ('DRACH', 'non_DRACH', 'unknown')


def read_tsv(path):
    with Path(path).open(newline='') as handle:
        yield from csv.DictReader(handle, delimiter='\t')


def write_tsv(path, rows, fields):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def motif_group(sequence):
    if len(sequence) != 5 or any(base not in 'ACGT' for base in sequence):
        return 'unknown'
    return 'DRACH' if drach(sequence) else 'non_DRACH'


def collapse_isoforms(rows, threshold):
    """Any positive isoform calls the locus; never sum shared reads across isoforms."""
    loci = {}
    for row in rows:
        key = row['chrom'], row['start'], row['strand']
        record = loci.setdefault(key, dict(score=-1., n_reads_max=0, transcripts=set()))
        record['score'] = max(record['score'], row['probability_modified'])
        record['n_reads_max'] = max(record['n_reads_max'], row['n_reads'])
        record['transcripts'].add(row['transcript_id'])
    scored = set(loci)
    positive = {key for key, record in loci.items() if record['score'] >= threshold}
    return loci, scored, positive


def map_m6anet(config, output):
    if config.get('mapped_sites') is not None:
        return reuse_mapped_sites(config)
    import pysam
    with config['m6anet_sites'].open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    wanted = {row['transcript_id'] for row in rows}
    print(f'Mapping {len(rows):,} transcript sites from {len(wanted):,} transcripts', flush=True)
    coordinates = load_transcript_coordinates(config['annotation'], wanted)
    missing = wanted - set(coordinates)
    if missing:
        raise ValueError('Inference transcripts missing from annotation: ' + ','.join(sorted(missing)[:10]))
    mapped, counts, seen = [], Counter(), set()
    with pysam.FastaFile(str(config['transcript_fasta'])) as fasta:
        for source in rows:
            tx, position = source['transcript_id'], int(source['transcript_position'])
            if (tx, position) in seen:
                raise ValueError('Duplicate transcript/position in inference')
            seen.add((tx, position))
            model = coordinates[tx]
            if model.length != fasta.get_reference_length(tx):
                raise ValueError('Annotation/FASTA transcript length mismatch: ' + tx)
            kmer = source['kmer'].upper()
            if position < 2 or fasta.fetch(tx, position - 2, position + 3).upper() != kmer:
                raise ValueError('m6Anet position/k-mer mismatch: ' + tx + ':' + str(position))
            if len(kmer) != 5 or kmer[2] != 'A':
                raise ValueError('Expected central A in m6Anet k-mer')
            score, n_reads = float(source['probability_modified']), int(source['n_reads'])
            if not 0 <= score <= 1 or n_reads < 20:
                raise ValueError('Invalid m6Anet score or inference coverage')
            start = model.genomic_position(position)
            row = dict(transcript_id=tx, transcript_position=position, chrom=model.chrom,
                       start=start, end=start + 1, strand=model.strand, n_reads=n_reads,
                       probability_modified=score, transcript_kmer=kmer, mod_ratio=float(source['mod_ratio']),
                       junction_context=int(model.junction_context(position)))
            mapped.append(row)
            counts['transcript_rows'] += 1
            counts['transcript_DRACH'] += drach(kmer)
            counts['junction_context'] += row['junction_context']
    loci, scored, positive = collapse_isoforms(mapped, config['m6anet_threshold'])
    counts.update(unique_genomic_loci=len(scored), positive_genomic_loci=len(positive), transcripts=len(wanted))
    return mapped, loci, scored, positive, dict(counts)


def reuse_mapped_sites(config):
    """Reuse the checked coordinate conversion when only comparison thresholds change."""
    directory = config['mapped_sites']
    previous = json.loads((directory / 'run.json').read_text())
    if previous['status'] != 'complete' or (directory / 'INCOMPLETE.txt').exists():
        raise ValueError('Coordinate cache must be complete')
    for name in ('annotation', 'fasta', 'transcript_fasta', 'm6anet_sites'):
        current, saved = file_info(config[name]), previous['sources'][name]
        if any(current[key] != saved[key] for key in current):
            raise ValueError('Coordinate cache input changed: ' + name)
    mapped = list(read_tsv(directory / 'm6anet_transcript_to_genome.tsv'))
    for row in mapped:
        for name in ('transcript_position','start','end','n_reads','junction_context','site_positive'):
            row[name] = int(row[name])
        for name in ('probability_modified','mod_ratio'):
            row[name] = float(row[name])
    loci, scored, positive = collapse_isoforms(mapped, config['m6anet_threshold'])
    audit = dict(previous['mapping_audit'])
    if len(mapped) != audit['transcript_rows'] or len(scored) != audit['unique_genomic_loci']:
        raise ValueError('Coordinate cache row count changed')
    audit['positive_genomic_loci'] = len(positive)
    print(f'Reusing {len(mapped):,} checked coordinate mappings; rescoring saved probabilities only', flush=True)
    return mapped, loci, scored, positive, audit


def read_predictions(path):
    """Keep only small per-site reporting values; do not read BAM or tensors."""
    records, sets = {}, {name: set() for name in ('Dorado', 'NN', 'Final')}
    for row in read_tsv(path):
        key = row['chrom'], int(row['start']), row['strand']
        if key in records or row['sample_id'] != 'HeLa_WT' or row['modification'] != 'm6A':
            raise ValueError('Unexpected sample/target or duplicate prediction coordinate')
        values = {field: int(row[field]) for field in ('flag_nn', 'flag_db', 'flag_short_read', 'flag_drach', 'filter_pass')}
        if bool(values['filter_pass']) != any(values[field] for field in ('flag_nn','flag_db','flag_short_read','flag_drach')):
            raise ValueError('Final output does not follow independent-flag OR policy')
        values.update(motif=row['motif'], motif_group=row['motif_group'], af=float(row['af']),
                      depth=int(row['total_reads']), score=float(row['score_m6A']))
        records[key] = values
        sets['Dorado'].add(key)
        if values['flag_nn']:
            sets['NN'].add(key)
        if values['filter_pass']:
            sets['Final'].add(key)
    return records, sets


def checked_coverage(config, glori):
    """Reuse the completed original-pileup scan only after verifying its provenance."""
    prediction = json.loads(config['prediction_run'].read_text())
    coverage_run = json.loads(config['coverage_run'].read_text())
    coverage_sources = json.loads(config['coverage_sources'].read_text())
    if prediction['status'] != 'complete' or coverage_run['status'] != 'complete':
        raise ValueError('Incomplete upstream prediction/coverage')
    original = prediction['reports']['candidates']['source']
    if original != coverage_run['input']:
        raise ValueError('Coverage cache uses another bedMethyl input')
    source = next(ref for ref in coverage_sources['references'] if ref['id'] == 'HeLa.m6A.GLORI')
    if sha256(config['glori']) != source['sha256'] or len(glori) != source['unique_sites']:
        raise ValueError('Coverage cache uses another GLORI catalogue')
    minimum = coverage_run['thresholds']['min_depth']
    coverage = {}
    for row in read_tsv(config['coverage']):
        if row['modification'] == 'm6A':
            key = row['chrom'], int(row['start']), row['strand']
            if key not in glori or key in coverage or int(row['total_reads']) < minimum:
                raise ValueError('Unexpected/duplicate coverage row')
            coverage[key] = int(row['total_reads'])
    return coverage, minimum


def venn_regions(methods):
    """All disjoint nonempty membership patterns, including zero-count regions."""
    names = list(methods)
    universe = set().union(*methods.values())
    counts = Counter(tuple(int(site in methods[name]) for name in names) for site in universe)
    return [dict(pattern=''.join(map(str, bits)), count=counts[bits],
                 members=';'.join(name for name, bit in zip(names, bits) if bit))
            for bits in product((0, 1), repeat=len(names)) if any(bits)]


def comparison_views(regions, sets, m6anet_scored, covered_glori):
    views = []
    for region in REGIONS[:2]:
        names = ['GLORI', 'm6Anet', 'rnaModScan'] if region == 'DRACH' else ['GLORI', 'rnaModScan']
        for stage in ('NN', 'Dorado'):
            source = dict(GLORI=sets['GLORI'], m6Anet=sets['m6Anet'], rnaModScan=sets[stage])
            selected = {name: source[name] & regions[region] for name in names}
            views.append(dict(name=region + '__' + stage + '__full_catalog', region=region,
                              stage=stage, universe='full_catalog', methods=selected))
        for stage in ('NN',):
            universe = m6anet_scored if region == 'DRACH' else sets['Dorado'] | covered_glori
            scope = 'm6anet_scored' if region == 'DRACH' else 'DRS_depth_eligible'
            selected = {name: source_set & regions[region] & universe for name, source_set in
                        dict(GLORI=sets['GLORI'], m6Anet=sets['m6Anet'], rnaModScan=sets[stage]).items() if name in names}
            views.append(dict(name=region + '__' + stage + '__' + scope, region=region,
                              stage=stage, universe=scope, methods=selected))
    return views


def compare_tables(views, sets, regions, covered_glori, config):
    region_rows, totals, overlaps, recovery = [], [], [], []
    for view in views:
        identity = {field: view[field] for field in ('name', 'region', 'stage', 'universe', 'm6anet_threshold')}
        for row in venn_regions(view['methods']):
            region_rows.append(dict(identity, **row))
        for name, sites in view['methods'].items():
            totals.append(dict(identity, method=name, sites=len(sites)))
        names = list(view['methods'])
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                a, b = view['methods'][left], view['methods'][right]
                intersection = len(a & b)
                overlaps.append(dict(identity, method_a=left, method_b=right, count_a=len(a), count_b=len(b),
                    overlap=intersection, percent_of_a=100*intersection/len(a) if a else None,
                    percent_of_b=100*intersection/len(b) if b else None,
                    jaccard=intersection/len(a | b) if a | b else None))
    for region in REGIONS[:2]:
        reference = sets['GLORI'] & regions[region]
        covered = reference & covered_glori
        for method in ('Dorado', 'NN', 'm6Anet'):
            if method == 'm6Anet' and region != 'DRACH':
                continue
            overlap = len(sets[method] & reference)
            recovered = len(sets[method] & covered)
            recovery.append(dict(region=region, method=method, glori_catalogue=len(reference),
                glori_drs_depth_eligible=len(covered), catalogue_overlap=overlap, covered_overlap=recovered,
                covered_reference_recovery_pct=100*recovered/len(covered) if covered else None,
                interpretation='reference recovery; not independent recall or precision'))
    return region_rows, totals, overlaps, recovery


def save_membership(view, output, motifs, predictions, m6anet):
    methods = view['methods']
    def rows():
        for chrom, start, strand in sorted(set().union(*methods.values())):
            key = chrom, start, strand
            own, net = predictions.get(key, {}), m6anet.get(key, {})
            row = dict(chrom=chrom, start=start, end=start + 1, strand=strand, motif=motifs[key],
                reference_A=int(motifs[key][2]=='A'),
                **{name: int(key in sites) for name, sites in methods.items()},
                flag_nn=own.get('flag_nn'), flag_db=own.get('flag_db'), flag_short_read=own.get('flag_short_read'),
                flag_drach=own.get('flag_drach'), rnamodscan_m6a_score=own.get('score'),
                dorado_af=own.get('af'), dorado_depth=own.get('depth'), m6anet_max_score=net.get('score'),
                m6anet_n_reads_max=net.get('n_reads_max'), m6anet_isoforms=len(net['transcripts']) if net else None)
            yield row
    fields = ['chrom', 'start', 'end', 'strand', 'motif', 'reference_A', *methods, 'flag_nn', 'flag_db',
              'flag_short_read', 'flag_drach', 'rnamodscan_m6a_score', 'dorado_af', 'dorado_depth',
              'm6anet_max_score', 'm6anet_n_reads_max', 'm6anet_isoforms']
    write_tsv(output / ('membership_' + view['name'] + '.tsv'), rows(), fields)


def as_sheet(name, records, fields=None):
    fields = fields or list(records[0])
    return dict(name=name, headers=fields, rows=[[row.get(field) for field in fields] for row in records])


def save_workbook(config, tables, audit, notes):
    old = verify_workbook(config['previous_workbook'])
    sheets = [dict(name=name, headers=rows[0], rows=rows[1:]) for name, rows in old.items()]
    for name, rows in tables.items():
        sheets.append(as_sheet(name, rows))
    sheets.append(as_sheet('HeLa_mapping_QC', [dict(item=key, value=value) for key, value in audit.items()]))
    sheets.append(as_sheet('HeLa_comparison_notes', [dict(item=key, value=value) for key,value in notes.items()]))
    path = config['output'] / 'Figure_data_HeLa_m6A_20261108.xlsx'
    write_workbook(path, sheets, float_format='0.0000')
    restored = verify_workbook(path)
    if any(restored[name] != rows for name, rows in old.items()):
        raise ValueError('Previous figure values changed in cumulative workbook')
    return dict(file=path.name, original_sheets=len(old), sheets=len(sheets), previous_values_preserved=True)


def run(config):
    output, figure_dir = config['output'], config['figures']
    if output.exists() or figure_dir.exists():
        raise FileExistsError('Choose fresh output/figures directories')
    output.mkdir(parents=True)
    (output / 'INCOMPLETE.txt').write_text('Coordinate mapping/comparison is incomplete.\n')
    mapped, m6anet, scored, positive, audit = map_m6anet(config, output)
    predictions, sets = read_predictions(config['predictions'])
    glori = load_bed_sites(config['glori'])
    covered, minimum = checked_coverage(config, glori)
    sets.update(GLORI=glori, m6Anet=positive)
    print('Mapped transcript coordinates; loading genome motifs', flush=True)
    all_sites = set().union(*sets.values(), scored)
    motifs, regions = {}, {name: set() for name in REGIONS}
    genome = IndexedFasta(config['fasta'])
    try:
        for site in sorted(all_sites):
            motif = genome.motif(*site)
            motifs[site] = motif
            regions[motif_group(motif)].add(site)
    finally:
        genome.close()
    for site, row in predictions.items():
        if motifs[site] != row['motif'] or motif_group(motifs[site]) != row['motif_group']:
            raise ValueError('Prediction/genome motif mismatch: ' + str(site))
    for row in mapped:
        key = row['chrom'], row['start'], row['strand']
        row['genome_kmer'] = motifs[key]
        row['genome_motif_group'] = motif_group(motifs[key])
        row['site_positive'] = int(row['probability_modified'] >= config['m6anet_threshold'])
        if row['genome_kmer'][2] != 'A':
            raise ValueError('Mapped m6Anet center is not genome-strand A')
        if not row['junction_context'] and row['genome_kmer'] != row['transcript_kmer']:
            raise ValueError('Non-junction transcript/genome k-mer mismatch')
    write_tsv(output / 'm6anet_transcript_to_genome.tsv', mapped, list(mapped[0]))
    collapsed = [dict(chrom=key[0], start=key[1], end=key[1]+1, strand=key[2], motif=motifs[key],
                     motif_group=motif_group(motifs[key]), probability_modified_max=row['score'],
                     n_reads_max=row['n_reads_max'], transcript_count=len(row['transcripts']),
                     transcript_ids=';'.join(sorted(row['transcripts'])), positive=int(key in positive))
                 for key,row in sorted(m6anet.items())]
    write_tsv(output / 'm6anet_genomic_sites.tsv', collapsed, list(collapsed[0]))
    audit.update(genome_DRACH_scored=len(scored & regions['DRACH']),
                 genome_non_DRACH_scored=len(scored & regions['non_DRACH']),
                 genome_unknown_scored=len(scored & regions['unknown']),
                 positive_DRACH=len(positive & regions['DRACH']),
                 positive_non_DRACH=len(positive & regions['non_DRACH']),
                 positive_unknown=len(positive & regions['unknown']),
                 transcript_sequence_checked=len(mapped), genome_center_A_checked=len(mapped))
    views = comparison_views(regions, sets, scored, set(covered))
    for view in views:
        view['m6anet_threshold'] = config['m6anet_threshold']
    for cutoff in config['m6anet_sensitivity_thresholds']:
        sensitivity_sets = dict(sets, m6Anet={site for site,row in m6anet.items() if row['score'] >= cutoff})
        sensitivity_views = [view for view in comparison_views(regions, sensitivity_sets, scored, set(covered))
                             if view['stage']=='NN']
        for view in sensitivity_views:
            view['name'] += '__m6anet_sensitivity_' + f'{cutoff:.2f}'.replace('.', 'p')
            view['m6anet_threshold'] = cutoff
        views.extend(sensitivity_views)
    region_rows, totals, overlaps, recovery = compare_tables(views, sets, regions, set(covered), config)
    reference_a = {site for site,sequence in motifs.items() if sequence[2]=='A'}
    transitions = [dict(region=region, dorado=len(sets['Dorado'] & sites), nn=len(sets['NN'] & sites),
                        final=len(sets['Final'] & sites),
                        dorado_reference_A=len(sets['Dorado'] & sites & reference_a),
                        nn_reference_A=len(sets['NN'] & sites & reference_a),
                        final_reference_A=len(sets['Final'] & sites & reference_a),
                        nn_removed=len((sets['Dorado'] - sets['NN']) & sites),
                        final_removed=len((sets['Dorado'] - sets['Final']) & sites))
                   for region,sites in regions.items()]
    sensitivity = [dict(threshold=threshold, motif_group=region,
                        positive_genomic_sites=sum(row['score'] >= threshold for site,row in m6anet.items() if site in sites),
                        glori_overlap=sum(row['score'] >= threshold for site,row in m6anet.items() if site in sites and site in glori))
                   for threshold in (.5, .7, .8, .9, .95) for region,sites in regions.items()]
    tables = dict(HeLa_venn_regions=region_rows, HeLa_set_sizes=totals, HeLa_pairwise=overlaps,
                  HeLa_GLORI_recovery=recovery, HeLa_filter_transitions=transitions,
                  HeLa_m6anet_thresholds=sensitivity)
    notes = dict(assembly=config['assembly'], coordinates='BED0, exact chromosome/start/strand; no renaming or liftover',
        m6anet_threshold=config['m6anet_threshold'],
        m6anet_sensitivity_thresholds=','.join(str(value) for value in config['m6anet_sensitivity_thresholds']),
        threshold_choice='primary0.8 chosen for this descriptive comparison after inspecting0.9 counts; official recommended0.9 retained as sensitivity analysis; no threshold optimization or equal-FPR calibration claimed',
        isoforms='one genome coordinate/strand; maximum isoform probability; n_reads are NOT summed',
        transcript_position='m6Anet2.1.0 position is zero-based center (eventalign position+2); checked against every transcript 5-mer',
        motif='genomic strand-oriented 5-mer; genomic non-A centers stay in original non-DRACH category and are separately flagged; junction transcript DRACH may be genomic non-DRACH',
        final='text only, no final-rescue Venn: flag_db OR flag_short_read OR flag_nn OR flag_drach; GLORI is part of rescue, so final agreement is not independent validation',
        nn='m6A softmax argmax only, before reference/DRACH rescue; model development used pooled-cell CV',
        full_catalog='all called loci and all GLORI catalogue positives; coverage differences retained',
        m6anet_scored='supplementary DRACH comparison restricts all three sets to positions actually scored by m6Anet',
        DRS_depth_eligible=f'supplementary non-DRACH comparison uses DRS-depth-eligible GLORI plus threshold-passing Dorado candidates; minimum depth {minimum}',
        coverage='reuse verified original bedMethyl/GLORI coverage cache; missing GLORI coverage is not a negative label',
        interpretation='descriptive overlaps/reference recovery, not specificity or independently measured precision/recall; GLORI negatives were not supplied',
        m6anet_non_DRACH='not a genome non-DRACH comparator; junction effects audited but excluded from two-method non-DRACH panels',
        official_threshold_source='https://m6anet.readthedocs.io/en/latest/quickstart.html')
    for name, rows in tables.items():
        write_tsv(output / (name + '.tsv'), rows, list(rows[0]))
    for view in views:
        save_membership(view, output, motifs, predictions, m6anet)
    workbook = save_workbook(config, tables, audit, notes)
    from analysis.hela_m6a.plots import draw_figures
    figures = draw_figures(views, figure_dir, threshold=config['m6anet_threshold'],
                           sensitivity_thresholds=config['m6anet_sensitivity_thresholds'])
    sources = {name: dict(file_info(path), sha256=sha256(path)) for name,path in config.items()
               if name in PATHS and name not in ('output','figures','fasta','transcript_fasta')}
    for name in ('fasta','transcript_fasta'):
        sources[name] = dict(file_info(config[name]), fai_sha256=sha256(str(config[name])+'.fai'))
    report = dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                  sample=config['sample'], assembly=config['assembly'], sources=sources,
                  notes=notes, mapping_audit=audit, workbook=workbook, figures=figures,
                  set_sizes={name:len(sites) for name,sites in sets.items()})
    if config.get('mapped_sites') is not None:
        report['coordinate_cache'] = dict(path=str(config['mapped_sites']),
            run_sha256=sha256(config['mapped_sites']/'run.json'),
            mapping_sha256=sha256(config['mapped_sites']/'m6anet_transcript_to_genome.tsv'))
    (output / 'run.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    (output / 'INCOMPLETE.txt').unlink()
    print(json.dumps(dict(mapping=audit,transitions=transitions,workbook=workbook),indent=2),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--mapped-sites', type=Path, help='Verified completed comparison directory; reuse coordinate mapping only')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for name in PATHS:
        config[name] = (args.config.resolve().parent / config[name]).resolve()
    config['mapped_sites'] = args.mapped_sites.resolve() if args.mapped_sites is not None else None
    config.setdefault('m6anet_sensitivity_thresholds', [config.get('m6anet_sensitivity_threshold', .9)])
    if config['assembly'] != 'GRCh38' or config['sample'] != 'HeLa_WT':
        raise ValueError('This explicit comparison is for HeLa/GRCh38')
    cutoffs = [config['m6anet_threshold'], *config['m6anet_sensitivity_thresholds']]
    if any(not 0 <= value <= 1 for value in cutoffs) or len(set(cutoffs)) != len(cutoffs):
        raise ValueError('m6anet thresholds must be between0 and1')
    run(config)


if __name__ == '__main__':
    main()
