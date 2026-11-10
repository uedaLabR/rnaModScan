"""Small file metadata and completion checks shared by all pipeline stages."""
import json
from pathlib import Path


def file_info(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def completed_run(directory):
    directory = Path(directory)
    if any((directory / name).exists() for name in ('INCOMPLETE.txt', 'FAILED.txt')):
        raise ValueError(f'Incomplete output: {directory}')
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    if run.get('status') != 'complete':
        raise ValueError(f'Not complete: {directory}')
    return run
