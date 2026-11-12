"""Check a prediction installation without sample data or scientific evaluation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_manifest(directory):
    count = 0
    for line in (directory / 'SHA256SUMS').read_text().splitlines():
        if not line.strip():
            continue
        expected, filename = line.split(maxsplit=1)
        path = (directory / filename.lstrip('*')).resolve()
        if directory.resolve() not in path.parents:
            raise ValueError('Checksum path escapes the bundle')
        if file_digest(path) != expected:
            raise ValueError(f'Checksum mismatch: {path.name}')
        count += 1
    return count


def check_catalog(directory):
    config = json.loads((directory / 'catalog.json').read_text())
    if config['assembly'] != 'GRCh38':
        raise ValueError('Expected the bundled human catalog')
    count = 0
    for source in config['sources']:
        if source.get('status', 'ready') != 'ready':
            continue
        path = (directory / source['path']).resolve()
        if directory.resolve() not in path.parents:
            raise ValueError('Reference path escapes the portable bundle')
        if file_digest(path) != source['sha256']:
            raise ValueError('Reference checksum mismatch')
        with gzip.open(path, 'rt') as handle:
            first = handle.readline().split('\t')
        if len(first) < 3 or int(first[1]) < 0 or int(first[2]) <= int(first[1]):
            raise ValueError('Invalid first reference BED record')
        if source.get('provenance'):
            provenance = (directory / source['provenance']).resolve()
            if directory.resolve() not in provenance.parents:
                raise ValueError('Provenance path escapes the portable bundle')
            json.loads(provenance.read_text(encoding='utf-8-sig'))
        count += 1
    return count


def check_model(directory, device):
    # Configure devices in this fresh process before TensorFlow loads.
    from prediction.runtime import configure_inference
    runtime = configure_inference(device, 1)
    from prediction.predict import _load_model
    import numpy as np

    model, _, run = _load_model(directory / 'adenine')
    shape = run['model']
    reads, window = shape['max_reads'], shape['window']
    inputs = {
        'sequence_tokens': np.zeros((2, window - 2), dtype=np.int32),
        'sequence_repeat_mask': np.zeros((2, window), dtype=bool),
        'read_features': np.zeros((2, reads, window, shape['read_channels']), dtype=np.float32),
        'read_mask': np.zeros((2, reads), dtype=bool),
    }
    # Two artificial one-read bags exercise both reference/repeat and read paths.
    # They have no biological labels and provide no accuracy estimate.
    inputs['read_mask'][:, 0] = True
    inputs['sequence_repeat_mask'][1] = True
    inputs['read_features'][:, 0, :, 10] = 1.0  # MATCH
    inputs['read_features'][:, 0, :, 8] = 1.0 / 21.0
    inputs['read_features'][:, 0, :, 16:18] = 1.0
    inputs['read_features'][0, 0, :, 0] = 0.8
    inputs['read_features'][1, 0, :, 1] = 0.8
    scores = np.asarray(model(inputs, training=False))
    if scores.shape != (2, 3) or not np.isfinite(scores).all():
        raise ValueError('Saved model returned invalid scores')
    if not np.allclose(scores.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError('Saved model softmax does not sum to one')
    if model.count_params() != 23731:
        raise ValueError('Unexpected saved-model parameter count')
    return dict(runtime=runtime, model_sha256=run['model_sha256'],
                parameters=model.count_params(), synthetic_bags=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    args = parser.parse_args()
    root = args.root.resolve()
    for command in ([], ['predict']):
        subprocess.run([sys.executable, '-m', 'rnamodscan', *command, '--help'],
                       check=True, stdout=subprocess.DEVNULL)
    from utils.config import load_config
    config = load_config(root / 'docker/predict.yaml', 'predict')
    if config['modkit']['run_pileup']:
        raise ValueError('The example must use prepared modkit outputs')
    model = root / 'models/pooled_HEK293T_HeLa_20261111'
    catalog = root / 'data/references/hg38'
    report = dict(status='PASS', purpose='installation smoke test only',
                  model_files_checked=check_manifest(model),
                  reference_files_checked=check_manifest(catalog),
                  reference_sources_checked=check_catalog(catalog),
                  **check_model(model, args.device))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
