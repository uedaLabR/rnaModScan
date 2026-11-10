"""Optional modkit pileup for prediction; candidate depth/AF filtering follows it."""
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from utils.io.artifacts import file_info
from utils.references.catalog import sha256


def require_file(path, description):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{description}: {path}')
    return path


def bam_index_path(bam):
    for path in (Path(str(bam) + '.bai'), Path(str(bam) + '.csi'),
                 bam.with_suffix('.bai'), bam.with_suffix('.csi')):
        if path.is_file():
            return path
    raise FileNotFoundError(f'BAM index required for modkit pileup: {bam}')


def probability(value):
    if isinstance(value, bool):
        raise ValueError('Modkit confidence thresholds must be probabilities in [0, 1]')
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError('Modkit confidence thresholds must be probabilities in [0, 1]')
    return str(number)


def threshold_arguments(filter_threshold, mod_thresholds):
    """Accept a global probability or explicit per-base thresholds; never AF."""
    arguments = []
    if filter_threshold is None:
        thresholds = []
    elif isinstance(filter_threshold, dict):
        thresholds = [f'{base}:{probability(value)}' for base, value in filter_threshold.items()]
    elif isinstance(filter_threshold, (list, tuple)):
        thresholds = filter_threshold
    else:
        thresholds = [filter_threshold]
    seen = set()
    for value in thresholds:
        text = str(value)
        if ':' in text:
            base, value = text.split(':', 1)
            if base not in 'ACGT' or len(base) != 1:
                raise ValueError('Per-base modkit thresholds must use A, C, G or T')
            formatted = base + ':' + probability(value)
        else:
            base, formatted = 'global', probability(value)
        if base in seen:
            raise ValueError(f'Duplicate modkit confidence threshold: {base}')
        seen.add(base)
        arguments += ['--filter-threshold', formatted]
    if mod_thresholds is not None and not isinstance(mod_thresholds, dict):
        raise ValueError('mod_thresholds must map modification codes to probabilities')
    for code, value in (mod_thresholds or {}).items():
        code = str(code)
        if not re.fullmatch(r'[a-z]|[0-9]+', code):
            raise ValueError('Use a SAM modification code, such as a, m, 17596 or 17802')
        arguments += ['--mod-threshold', code + ':' + probability(value)]
    return arguments


def run_modkit_pileup(bam_path, output_dir, *, executable='modkit', threads=2,
                      reference_fasta=None, filter_threshold=None, mod_thresholds=None,
                      include_bed=None, seed=None):
    """Run pileup once into a new directory, preserving command/version/logs.

    Omitting confidence settings retains modkit's sampled threshold estimation.
    All modification codes and both strands are retained. This does not modify
    the input BAM, apply site AF filters, or use call-mods' probability clamping.
    """
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError('modkit threads must be a positive integer')
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError('modkit seed must be a nonnegative integer or null')
    resolved_executable = shutil.which(str(executable))
    if resolved_executable is None:
        raise FileNotFoundError(f'modkit executable not found: {executable}')
    bam = require_file(bam_path, 'modkit input BAM')
    sources = [file_info(bam), file_info(bam_index_path(bam))]
    out = Path(output_dir).resolve()
    bed, log = out / 'all_modifications.bed', out / 'pileup.log'
    command = [resolved_executable, 'pileup', str(bam), str(bed),
               '--threads', str(threads), '--log-filepath', str(log)]
    if reference_fasta is not None:
        reference = require_file(reference_fasta, 'modkit reference FASTA')
        index = require_file(str(reference) + '.fai', 'modkit reference FASTA index')
        sources += [file_info(reference), file_info(index)]
        command += ['--reference', str(reference)]
    if include_bed is not None:
        included = require_file(include_bed, 'modkit include BED')
        sources.append(file_info(included))
        command += ['--include-bed', str(included)]
    if seed is not None:
        command += ['--seed', str(seed)]
    command += threshold_arguments(filter_threshold, mod_thresholds)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE.txt').write_text('modkit pileup not complete.\n')
    version_command = [resolved_executable, '--version']
    (out / 'command.json').write_text(json.dumps(dict(argv=command, version_argv=version_command), indent=2) + '\n')
    try:
        version_result = subprocess.run(version_command, capture_output=True, text=True)
        (out / 'modkit_version.txt').write_text(version_result.stdout + version_result.stderr)
        if version_result.returncode:
            raise RuntimeError(f'modkit --version failed with exit code {version_result.returncode}')
        version = version_result.stdout.strip() or version_result.stderr.strip()
        if not version:
            raise RuntimeError('modkit --version did not report a version')
        with (out / 'stdout.log').open('w') as stdout, (out / 'stderr.log').open('w') as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, text=True)
        if result.returncode:
            raise RuntimeError(f'modkit pileup failed with exit code {result.returncode}; see {out / "stderr.log"}')
        require_file(bed, 'modkit pileup output bedMethyl')
        require_file(log, 'modkit pileup run log')
        if any(file_info(source['path']) != source for source in sources):
            raise ValueError('A modkit input changed during pileup')
        report = dict(status='complete', schema='modkit_prediction_pileup_v1',
                      bedmethyl=str(bed), pileup_log=str(log), argv=command,
                      modkit_version=version, version_argv=version_command, inputs=sources,
                      bedmethyl_info=file_info(bed), pileup_log_sha256=sha256(log),
                      policies=dict(confidence='modkit defaults unless explicitly set in argv; not site AF',
                                    include_bed='when configured, restricts both threshold estimation and output positions',
                                    channels='all available modification codes; no collapsing or strand combination',
                                    bam='original MM/ML probabilities preserved; no call-mods preprocessing'))
        (out / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
        (out / 'INCOMPLETE.txt').unlink()
        return report
    except BaseException as error:
        (out / 'FAILED.txt').write_text(str(error) + '\n')
        raise
