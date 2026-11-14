"""Route policy candidates around BAM/NN; score every m6A candidate."""
import csv
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from utils.features.index import INDEX_FIELDS
from utils.io.artifacts import completed_run, file_info
from utils.references.catalog import sha256
from utils.references.genome import IndexedFasta, drach, mettl16
from utils.adenine import FIELDS as ADENINE_FIELDS, MODIFICATIONS, combine_site, merged_sites
from prediction.confidence import AUDIT_FIELDS, combine_confidence
from prediction.motifs import POLICY_MOTIFS, routed_policy_motif
from prediction.auxiliary import auxiliary_motif_eligible

ROUTE_FIELDS = ['filter_mode', 'motif', 'motif_group', 'mettl16_motif',
                'policy_motif_code', 'policy_motif_sequence', 'auxiliary_motif_eligible']


def route_prediction_sites(candidate_root, out, targets, filtering, fasta_path=None, *, joint=False):
    candidate_root, out = Path(candidate_root), Path(out)
    completed_run(candidate_root)
    source_hash = sha256(candidate_root / 'run.json')
    source_sites = {target: sha256(candidate_root / target / 'sites.tsv') for target in targets}
    needs_motif = 'm6A' in targets or any(target in targets and filtering.get(option, False)
        for code, target, option, flag, pattern, index in POLICY_MOTIFS)
    reference = file_info(fasta_path) if needs_motif else None
    if out.exists():
        run = completed_run(out)
        if (run['candidate_run_sha256'] != source_hash or run['filtering'] != filtering
                or run['targets'] != list(targets) or run['reference'] != reference
                or run.get('candidate_sites_sha256') != source_sites or run.get('joint', False) != joint):
            raise ValueError('Prediction routing changed; use a new output directory')
        for name, digest in run['files_sha256'].items():
            if sha256(out / name) != digest:
                raise ValueError('Routed sites changed after preparation')
        return run
    out.mkdir(parents=True)
    (out / 'INCOMPLETE.txt').write_text('Candidate routing not complete.\n')
    genome = IndexedFasta(fasta_path) if needs_motif else None
    counts, files = Counter(), []
    try:
        for target in targets:
            source = candidate_root / target / 'sites.tsv'
            directory = out / target
            directory.mkdir()
            with source.open(encoding='utf-8', newline='') as handle, ExitStack() as stack:
                reader = csv.DictReader(handle, delimiter='\t')
                fields = list(dict.fromkeys(reader.fieldnames + INDEX_FIELDS + ROUTE_FIELDS + list(AUDIT_FIELDS)))
                writers = {}
                for name in ('nn_sites.tsv', 'policy_sites.tsv'):
                    destination = directory / name
                    output = stack.enter_context(destination.open('w', encoding='utf-8', newline=''))
                    writer = csv.DictWriter(output, fields, delimiter='\t')
                    writer.writeheader()
                    writers[name] = writer
                    files.append(destination)
                for row in reader:
                    if row['modification'] != target:
                        raise ValueError('Mixed modification in candidate list')
                    mode, motif, group, nonamer = 'nn', '', '', ''
                    policy_code, policy_sequence = 0, ''
                    eligible = ''
                    if target in filtering['reference_only_targets']:
                        mode = 'reference_only'
                        eligible = int(auxiliary_motif_eligible(row, filtering.get('auxiliary_motif_min_depth', 20),
                                                           filtering.get('auxiliary_motif_min_af', .40)))
                        if genome is not None and eligible:
                            policy_code, policy_sequence = routed_policy_motif(genome, row, filtering)
                    elif target == 'm6A' and genome is not None:
                        motif = genome.motif(row['chrom'], int(row['start']), row['strand'])
                        nonamer = genome.context(row['chrom'], int(row['start']), row['strand'], left=3, right=5)
                        group = 'DRACH' if drach(motif) else 'non_DRACH' if 'N' not in motif else 'unknown'
                        if group == 'DRACH' and filtering['m6a_drach_keep']:
                            mode = 'nn_drach_keep'
                        elif filtering.get('m6a_mettl16_keep', False) and mettl16(nonamer):
                            mode = 'nn_mettl16_keep'
                    row.update(filter_mode=mode, motif=motif, motif_group=group, mettl16_motif=nonamer,
                               policy_motif_code=policy_code, policy_motif_sequence=policy_sequence,
                               auxiliary_motif_eligible=eligible,
                               retained_depth=0, eligible_depth=0, read_offset=0,
                               features_dir='', reference_sequence='')
                    name = 'policy_sites.tsv' if mode == 'reference_only' else 'nn_sites.tsv'
                    writers[name].writerow(row)
                    counts[target, mode] += 1
        if joint:
            directory = out / 'adenine'
            directory.mkdir()
            paths = [out / target / 'nn_sites.tsv' for target in MODIFICATIONS]
            with paths[0].open(encoding='utf-8') as handle:
                fields = list(dict.fromkeys(next(csv.reader(handle, delimiter='\t')) + ADENINE_FIELDS))
            destination = directory / 'nn_sites.tsv'
            with destination.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fields, delimiter='\t', extrasaction='ignore')
                writer.writeheader()
                for rows in merged_sites(paths, sort_database=directory / 'sort.sqlite'):
                    row = combine_site(rows)
                    combine_confidence(row, rows)
                    row.update(filter_mode='nn', label='', label_reason='joint_prediction_candidate')
                    motif = genome.motif(row['chrom'], int(row['start']), row['strand'])
                    row['motif'] = motif
                    row['mettl16_motif'] = genome.context(row['chrom'], int(row['start']), row['strand'], left=3, right=5)
                    row['motif_group'] = 'DRACH' if drach(motif) else 'non_DRACH' if 'N' not in motif else 'unknown'
                    writer.writerow(row)
                    counts['adenine', 'nn'] += 1
            files.append(destination)
            destination = directory / 'policy_sites.tsv'
            with destination.open('w', encoding='utf-8', newline='') as handle:
                csv.writer(handle, delimiter='\t').writerow(fields)
            files.append(destination)
        if any(sha256(candidate_root / target / 'sites.tsv') != digest for target, digest in source_sites.items()):
            raise ValueError('Candidates changed during routing')
        run = dict(status='complete', schema='prediction_routing_v1', targets=list(targets), joint=joint,
                   candidate_run_sha256=source_hash, filtering=filtering, reference=reference,
                   candidate_sites_sha256=source_sites,
                   files_sha256={str(path.relative_to(out)): sha256(path) for path in files},
                   counts=[dict(modification=target, filter_mode=mode, sites=count)
                           for (target, mode), count in sorted(counts.items())])
        (out / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
        (out / 'INCOMPLETE.txt').unlink()
        return run
    except BaseException as error:
        (out / 'FAILED.txt').write_text(str(error) + '\n')
        raise
    finally:
        if genome is not None:
            genome.close()
