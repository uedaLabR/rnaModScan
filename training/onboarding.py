"""Create fresh training settings and check prerequisites without starting jobs."""
import argparse
import importlib
import json
import os
from pathlib import Path

from utils.config import build_sample_pairs, enabled_samples, load_config, training_site_source

PIPELINE_STAGES = ('sites', 'select', 'features', 'annotation', 'split', 'fit', 'final')
ROOT = Path(__file__).resolve().parents[1]


def planned_stages(config, stage='all'):
    """Annotation reports gene context; it is not a neural-model input."""
    if stage != 'all':
        return (stage,)
    return tuple(name for name in PIPELINE_STAGES
                 if name != 'annotation' or config['references']['gtf'] is not None)


def relative_path(path, directory):
    return os.path.relpath(Path(path).resolve(), Path(directory).resolve())


def initialize(output, *, cell='sample_cell', reference_cell=None, catalog=None,
               fasta=None, gtf=None, input_dir=None, run_dir=None,
               wt_id='sample_WT', ivt_id='sample_IVT'):
    """Write only a new YAML file. No data, environment or results are changed."""
    import yaml

    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Configuration exists: {output}; choose another --output')
    if not isinstance(cell, str) or not cell.strip():
        raise ValueError('Set a nonempty --cell')
    if any(not value or any(c.isspace() for c in value) for value in (wt_id, ivt_id)) or wt_id == ivt_id:
        raise ValueError('Set distinct --wt-id and --ivt-id without whitespace')
    reference_cell = reference_cell or cell
    if not reference_cell.strip():
        raise ValueError('Set a nonempty --reference-cell')
    directory = output.parent
    inputs = Path(input_dir).expanduser().resolve() if input_dir else directory / 'inputs'
    destination = Path(run_dir).expanduser().resolve() if run_dir else directory / 'runs' / output.stem
    if destination.exists():
        raise FileExistsError(f'Training output exists: {destination}; choose a new --run-dir')
    catalog = Path(catalog).expanduser().resolve() if catalog else ROOT / 'data/references/hg38/catalog.json'
    fasta = Path(fasta).expanduser().resolve() if fasta else inputs / 'reference.softmasked.fa'

    def path(value):
        return relative_path(value, directory) if value is not None else None

    config = dict(schema_version=1, mode='train', output_dir=path(destination),
                  targets=['m6A', 'inosine'],
                  selection=dict(enabled=True, source_dir=None, seed=42),
                  references=dict(fasta=path(fasta), gtf=path(gtf), catalog=path(catalog)),
                  samples=[dict(sample_id=sample_id, cell=cell, reference_cell=reference_cell,
                                condition=condition, bedmethyl=path(inputs / f'{sample_id}.bed'),
                                pileup_log=path(inputs / f'{sample_id}.pileup.log'),
                                bam=path(inputs / f'{sample_id}.bam'), enabled=True)
                           for sample_id, condition in [(wt_id, 'WT'), (ivt_id, 'IVT')]],
                  features=dict(flank=9, min_mapq=0, sample_workers=1, bam_threads=1,
                                group_workers=1, candidate_group_gap=10000,
                                group_batch_size=16, max_fetch_span=10000000, read_batch_size=256),
                  candidates=dict(min_depth=10, min_modified=3, min_af=0.1,
                                  ivt_min_modified=3, ivt_af_ratio=0.6),
                  cv=dict(seed=42, validation_fraction=0.1, require_all_samples=True),
                  nn=dict(task='adenine_three_class', batch_sites=16, max_reads=300,
                          sampling_seed=42, epochs=20, patience=3, learning_rate=0.0003,
                          folds=list(range(5)), target_settings=dict(adenine=dict(final_epochs=None))))
    prefix = ('# Fresh training configuration; paths are relative to this YAML.\n'
              '# Edit the input paths before running train check.\n'
              '# The default catalog is HUMAN GRCh38. For another assembly pass --catalog.\n'
              '# References provide positive evidence; unmatched sites are never auto-negatives.\n'
              '# Paired WT/IVT inputs and positive evidence for BOTH m6A and inosine are needed.\n'
              '# Gene annotation is optional: gtf:null skips annotation in --stage all.\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        handle.write(prefix + yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return dict(status='created', config=str(output), output_dir=str(destination),
                reference_catalog=str(catalog), cell=cell, reference_cell=reference_cell,
                next_command=f'rnamodscan train check --config {output}',
                note='Only the new YAML was written. Replace input paths; no training has started.')


def _dependencies(stages):
    names = ['yaml']
    if set(stages) & {'features', 'quality', 'fit', 'final', 'inspect'}:
        names.append('ctypes')
    if set(stages) & {'features', 'quality', 'fit', 'final'}:
        names.append('pysam')
    if set(stages) & {'features', 'fit', 'final'}:
        names.append('pyarrow')
    if set(stages) & {'fit', 'final', 'inspect'}:
        names.extend(['tensorflow', 'matplotlib'])
    return names


def check_training(config, stages, *, skip_dependencies=False):
    """Inspect paths, small manifests and optional headers, never scan alignments."""
    stages = tuple(stages)
    errors, warnings, dependencies, checked = [], [], [], []

    def issue(setting, message, path=None):
        errors.append(dict(setting=setting, message=message,
                           **(dict(path=str(path)) if path is not None else {})))

    def require(path, setting):
        if path is None:
            issue(setting, 'Set this path in the configuration')
            return False
        if not Path(path).is_file():
            issue(setting, 'Input file is missing', path)
            return False
        checked.append(dict(setting=setting, path=str(path)))
        return True

    if not skip_dependencies:
        for name in _dependencies(stages):
            try:
                module = importlib.import_module(name)
                dependencies.append(dict(package=name, status='available', version=getattr(module, '__version__', None)))
            except (ImportError, OSError, RuntimeError, ValueError, AttributeError, TypeError) as error:
                dependencies.append(dict(package=name, status='unavailable'))
                issue('dependencies.' + name, str(error))
    else:
        warnings.append('Dependency imports and BAM header checks were skipped.')
    if 'final' in stages and 'fit' in stages and set(config['nn']['folds']) != set(range(5)):
        issue('nn.folds', 'The final refit requires all five completed CV folds; use folds [0,1,2,3,4] for the all-stage workflow.')
    pairs = build_sample_pairs(config['samples'])
    fresh_sites = 'sites' in stages and training_site_source(config) == config['output_dir'] / 'sites'
    references = config['references']
    if fresh_sites:
        for sample in config['samples']:
            for name in ['bedmethyl', 'pileup_log']:
                require(sample[name], f'samples.{sample["sample_id"]}.{name}')
    if set(stages) & {'features', 'quality'}:
        for sample in enabled_samples(config):
            bam = sample['bam']
            if require(bam, f'samples.{sample["sample_id"]}.bam'):
                indices = [Path(str(bam)+'.bai'), Path(str(bam)+'.csi'), bam.with_suffix('.bai'), bam.with_suffix('.csi')]
                if not any(p.is_file() for p in indices):
                    issue(f'samples.{sample["sample_id"]}.bam_index', 'A BAM BAI/CSI index is required', bam)
    if 'features' in stages:
        if require(references['fasta'], 'references.fasta'):
            require(Path(str(references['fasta'])+'.fai'), 'references.fasta_index')
    if 'annotation' in stages:
        require(references['gtf'], 'references.gtf')

    catalog_assembly = None
    if fresh_sites and require(references['catalog'], 'references.catalog'):
        try:
            path = references['catalog']
            manifest = json.loads(path.read_text(encoding='utf-8-sig'))
            catalog_assembly = manifest['assembly']
            if not isinstance(catalog_assembly, str) or not catalog_assembly.strip():
                raise ValueError('Declare the assembly explicitly in the reference catalog')
            sources = manifest['sources']
            if not isinstance(sources, list):
                raise ValueError('Catalog sources must be a list')
            if any(not isinstance(source, dict) for source in sources):
                raise ValueError('Catalog source entries must be mappings')
            identifiers = set()
            for source in sources:
                identifier = source.get('id')
                if (not isinstance(identifier, str) or not identifier
                        or identifier in identifiers or identifier in ('short_read_any', 'database_any')):
                    issue('references.catalog', 'Use unique nonempty, non-reserved source IDs')
                if isinstance(identifier, str):
                    identifiers.add(identifier)
                if source.get('kind') not in ('short_read', 'database') or source.get('modification') not in ('m6A', 'inosine', 'm5C', 'pseU'):
                    issue('references.catalog', f'Source {identifier} has an unsupported kind or modification')
                if not source.get('cell') or (source.get('kind') == 'short_read' and source['cell'] == '*'):
                    issue('references.catalog', f'Source {identifier} needs an explicit cell/tissue scope; only database entries may use *')
                if source.get('status', 'ready') not in ('ready', 'unavailable'):
                    issue('references.catalog', f'Source {identifier} status must be ready or unavailable')
                if source.get('assembly') != catalog_assembly or source.get('coordinates') != 'BED0':
                    issue('references.catalog', f'Source {source.get("id")} must use the declared assembly and BED0 coordinates')
                if source.get('status', 'ready') != 'ready':
                    continue
                if source.get('path'):
                    require((path.parent/source['path']).resolve(), 'reference_source.'+str(source.get('id')))
                else:
                    issue('references.catalog', f'Ready source {source.get("id")} has no path')
                if source.get('provenance'):
                    require((path.parent/source['provenance']).resolve(), 'reference_provenance.'+str(source.get('id')))
            for pair in pairs:
                for target in ['m6A', 'inosine']:
                    available = [source for source in sources if source.get('modification') == target
                                 and source.get('status', 'ready') == 'ready'
                                 and source.get('cell') in (pair['reference_cell'], '*')]
                    if not available:
                        message = f'No ready {target} positive-reference source applies to {pair["reference_cell"]}; the joint model needs both modification classes.'
                        if set(stages) & {'fit', 'final'}:
                            issue('references.catalog', message)
                        else:
                            warnings.append(message)
            warnings.append('Reference contents/checksums and candidate-label class counts are checked by preparation/fitting; this check reads only catalog metadata and file paths.')
        except (KeyError, TypeError, ValueError, OSError) as error:
            issue('references.catalog', str(error))

    # The established launcher checks completion markers and stage-specific reuse
    # rules; it does not open BAM alignments or start a worker.
    from training.run_training import check_inputs, check_outputs
    standard = tuple(stage for stage in stages if stage in PIPELINE_STAGES)
    if not errors and standard:
        try:
            check_inputs(config, standard)
        except (ValueError, OSError, KeyError, TypeError) as error:
            issue('prepared_inputs', str(error))
    if standard:
        try:
            check_outputs(config, standard)
        except (ValueError, OSError, KeyError, TypeError) as error:
            issue('output_dir', str(error))
    if not skip_dependencies and 'features' in stages and not errors:
        try:
            import pysam
            lengths = {}
            with Path(str(references['fasta'])+'.fai').open() as handle:
                for line in handle:
                    fields = line.rstrip().split('\t')
                    lengths[fields[0]] = int(fields[1])
            for sample in enabled_samples(config):
                with pysam.AlignmentFile(str(sample['bam']), 'rb') as bam:
                    for chrom, length in zip(bam.references, bam.lengths):
                        if lengths.get(chrom) != length:
                            issue('sample_reference_match', f'{sample["sample_id"]}: BAM contig {chrom} length {length} does not match FASTA/FAI')
        except (ValueError, OSError, KeyError, IndexError) as error:
            issue('sample_reference_match', str(error))
    return dict(status='ready' if not errors else 'not_ready', mode='train',
                config=str(config['config_path']), output_dir=str(config['output_dir']),
                planned_stages=list(stages),
                skipped_optional_stages=['annotation'] if 'annotation' not in stages and references['gtf'] is None else [],
                catalog_assembly=catalog_assembly, checked_files=checked,
                dependencies=dependencies, warnings=warnings, errors=errors,
                label_policy='Positive reference matches plus paired WT/IVT observations; missing reference overlap is not a negative label.',
                read_only=True, jobs_started=False)


def init_main(argv=None):
    parser = argparse.ArgumentParser(description='Create a fresh WT/IVT training YAML; existing files are never replaced')
    parser.add_argument('--output', type=Path, required=True, help='New YAML file')
    parser.add_argument('--cell', default='sample_cell', help='Explicit cell/tissue identity')
    parser.add_argument('--reference-cell', help='Exact cell/tissue scope in the catalog; defaults to --cell')
    parser.add_argument('--catalog', type=Path, help='Custom positive-reference catalog; defaults to the included HUMAN GRCh38 catalog')
    parser.add_argument('--fasta', type=Path, help='Matching softmasked FASTA; FAI is also required')
    parser.add_argument('--gtf', type=Path, help='Optional same-assembly GTF/GFF3 for reporting annotation')
    parser.add_argument('--input-dir', type=Path, help='Directory containing sample BAM/bedMethyl/pileup log inputs')
    parser.add_argument('--run-dir', type=Path, help='New output directory for this training run')
    parser.add_argument('--wt-id', default='sample_WT', help='WT sample ID and default filename stem')
    parser.add_argument('--ivt-id', default='sample_IVT', help='IVT sample ID and default filename stem')
    arguments = parser.parse_args(argv)
    try:
        report = initialize(**vars(arguments))
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, OSError, TypeError) as error:
        parser.exit(2, f'error: {error}\n')


def check_main(argv=None):
    from utils.config import TRAIN_STAGES
    parser = argparse.ArgumentParser(description='Read-only training preflight; no extraction, fitting or output writing')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('all', *TRAIN_STAGES), default='all')
    parser.add_argument('--skip-dependencies', action='store_true', help='Check paths/manifests without importing native/NN dependencies or opening BAM headers')
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config, 'train')
        report = check_training(config, planned_stages(config, arguments.stage),
                                skip_dependencies=arguments.skip_dependencies)
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 0 if report['status'] == 'ready' else 2
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as error:
        parser.exit(2, f'error: {error}\n')
