"""Choose a species-matched reference catalog without touching BAM or NN inputs."""
import json


def checked_catalog_path(config):
    """Other genomes have no references; known genomes must match the catalog."""
    genome = config['genome']
    if genome == 'other':
        return None
    path = config['references']['catalog']
    if path is None or not path.is_file():
        hint = (' Run: python analysis/reference_catalog/prepare_mouse_reference_catalog.py'
                if genome == 'mm10' else '')
        raise FileNotFoundError(f'Missing references.catalog: {path}.{hint}')
    manifest = json.loads(path.read_text(encoding='utf-8-sig'))
    expected = 'GRCh38' if genome == 'hg38' else 'mm10'
    if manifest.get('assembly') != expected:
        raise ValueError(f'genome={genome} requires a {expected} catalog; got {manifest.get("assembly")}')
    return path
