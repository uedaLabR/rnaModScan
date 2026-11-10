"""Re-extract capped, per-site Parquet from BAM, preserving verified labels and CV policy."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import time

from training.selection_validation import verify_selected_sites, verify_source_report, DEFAULT_SOURCE_REPORT
from utils.config import load_config, enabled_samples, index_paths
from utils.io.artifacts import completed_run
from utils.references.catalog import sha256

ROOT = Path(__file__).resolve().parents[1]


def rebuild_site_parquet(config_path, selection_dir, *, source_report=None, dry_run=False):
    config = load_config(config_path, 'train')
    root = config['output_dir']
    selection_dir = Path(selection_dir).resolve()
    site_list = selection_dir / 'adenine/training_sites.tsv'
    nn, features = config['nn'], config['features']
    if dry_run:
        return dict(status='planned', source_selection=str(selection_dir), site_list=str(site_list),
                    output_dir=str(root), stages=['features', 'split', 'cell_dataset'],
                    max_reads=nn['max_reads'], disk_padding=False, neural_fitting=False,
                    candidate_group_gap=features['candidate_group_gap'], read_batch_size=features['read_batch_size'],
                    sample_workers=features['sample_workers'], group_workers=features['group_workers'],
                    group_batch_size=features['group_batch_size'], max_fetch_span=features['max_fetch_span'],
                    samples=[s['sample_id'] for s in enabled_samples(config)])
    if nn['max_reads'] != 300:
        raise ValueError('Use max_reads300; site Parquet stores the four NN inputs together')
    if selection_dir == root / 'selection':
        raise ValueError('Use a separate verified source selection and a fresh output directory')
    site_list, selected_run, sample_counts, class_counts = verify_selected_sites(config, selection_dir)
    source_report = Path(source_report) if source_report is not None else selection_dir / 'source_site_run.json'
    if not source_report.exists():
        source_report = DEFAULT_SOURCE_REPORT
    verify_source_report(config, source_report)
    for sample in enabled_samples(config):
        bam = sample['bam']
        if bam is None or not bam.is_file():
            raise FileNotFoundError(f"Missing BAM for {sample['sample_id']}")
        if not any(path.is_file() for path in (Path(str(bam)+'.bai'), Path(str(bam)+'.csi'), bam.with_suffix('.bai'))):
            raise FileNotFoundError(f'Missing BAM index: {bam}')
    reference = config['references']['fasta']
    if not reference.is_file() or not Path(str(reference)+'.fai').is_file():
        raise FileNotFoundError('Set an indexed reference FASTA')
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / 'preparation_run.json'
    identity = dict(schema='site_bag_preparation_v1', selected_sites_sha256=sha256(site_list),
                    source_selection=str(selection_dir), config_sha256=sha256(config['config_path']))
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError('Preparation settings changed; choose a new output directory')
        if previous['status'] == 'complete':
            return previous
    report = dict(identity, status='running', started_unix=time.time(), completed_stages=[],
                  sample_sites=sample_counts, class_counts=class_counts, neural_fitting=False)
    def save():
        partial = report_path.with_suffix('.tmp')
        partial.write_text(json.dumps(report, indent=2)+'\n')
        partial.replace(report_path)
    save()
    try:
        from utils.features.site_bag_pipeline import extract_site_bags_and_index
        from training.cross_validation import build_cross_validation
        from training.cell_datasets import build_cell_datasets
        report['current_stage'] = 'features'
        save()
        extract_site_bags_and_index(site_list, enabled_samples(config), reference, root / 'features',
            config['references']['catalog'], mode='training', max_reads=300,
            sampling_seed=nn['sampling_seed'], sample_workers=features['sample_workers'],
            flank=features['flank'], min_mapq=features['min_mapq'], bam_threads=features['bam_threads'],
            progress_seconds=features['progress_seconds'],
            candidate_group_gap=features['candidate_group_gap'], read_batch_size=features['read_batch_size'],
            group_workers=features['group_workers'], group_batch_size=features['group_batch_size'],
            max_fetch_span=features['max_fetch_span'])
        report['completed_stages'].append('features')
        report['current_stage'] = 'split'
        save()
        if not (root / 'splits/adenine').exists():
            build_cross_validation(index_paths(config), root / 'splits/summary', flank=features['flank'],
                seed=config['cv']['seed'], validation_fraction=config['cv']['validation_fraction'],
                target_outputs={'adenine': root / 'splits/adenine'})
        else:
            completed_run(root / 'splits/adenine')
        report['completed_stages'].append('split')
        report['current_stage'] = 'cell_dataset'
        save()
        if not (root / 'cell_datasets').exists():
            build_cell_datasets(index_paths(config), config['samples'], root / 'cell_datasets',
                               flank=features['flank'], **config['cell_dataset'])
        else:
            completed_run(root / 'cell_datasets')
        report['completed_stages'].append('cell_dataset')
        report.update(status='complete', current_stage=None, finished_unix=time.time())
        save()
        return report
    except BaseException as error:
        report.update(status='failed', error=str(error), finished_unix=time.time())
        save()
        raise


def start_background(config_path, selection_dir, source_report=None):
    """Detach the CPU preparation; create a separate log, never start NN fitting."""
    config = load_config(config_path, 'train')
    root = config['output_dir']
    root.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(ROOT / 'training/test/00_rebuild_site_parquet.py'),
               '--config', str(config['config_path']), '--selection-dir', str(Path(selection_dir).resolve())]
    if source_report is not None:
        command += ['--source-report', str(Path(source_report).resolve())]
    with (root / 'rebuild.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    report = dict(status='starting', pid=process.pid, command=command, log=str(root / 'rebuild.log'))
    (root / 'background_run.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'training/config.yaml')
    parser.add_argument('--selection-dir', type=Path, default=ROOT / 'results/training_v5/selection')
    parser.add_argument('--source-report', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--background', action='store_true', help='Continue preparation after terminal disconnection')
    args = parser.parse_args(argv)
    if args.background and not args.dry_run:
        report = start_background(args.config, args.selection_dir, args.source_report)
    elif args.dry_run:
        report = rebuild_site_parquet(args.config, args.selection_dir, source_report=args.source_report, dry_run=True)
    else:
        import fcntl
        config = load_config(args.config, 'train')
        config['output_dir'].mkdir(parents=True, exist_ok=True)
        with (config['output_dir'] / '.preparation.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('Another preparation process is already running') from error
            report = rebuild_site_parquet(args.config, args.selection_dir, source_report=args.source_report)
    print(json.dumps(report, indent=2))
