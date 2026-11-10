"""Explicit, assembly-aware reference catalog. BED coordinates are zero-based."""
import hashlib
import json
from pathlib import Path

MODIFICATIONS = ('m6A', 'm5C', 'inosine', 'pseU')


from utils.io.bed import load_bed_sites


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ReferenceCatalog:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve() if manifest is not None else None
        self.sources = []
        self.sites = {}
        self._scopes = {}
        self.assembly = None
        if self.path is None:
            return
        config = json.loads(self.path.read_text(encoding='utf-8-sig'))
        self.assembly = config['assembly']
        if self.assembly not in ('GRCh38', 'mm10'):
            raise ValueError('Catalog must use GRCh38 or mm10')
        ids = set()
        for source in config['sources']:
            s = dict(source)
            if s['id'] in ids or s['id'] in ('short_read_any', 'database_any'):
                raise ValueError('Duplicate or reserved source ID')
            ids.add(s['id'])
            if s['kind'] not in ('short_read', 'database') or s['modification'] not in MODIFICATIONS:
                raise ValueError(f'Invalid reference source: {s}')
            if s['assembly'] != self.assembly or s.get('coordinates') != 'BED0':
                hint = 'lift hg19 first' if self.assembly == 'GRCh38' else 'convert other assemblies first'
                raise ValueError(f'References must be {self.assembly} BED0; {hint}')
            if self.assembly == 'mm10' and not (s['kind'] == 'database'
                    and s.get('database') == 'RMBase' and str(s.get('version')) == '3.0'):
                continue  # Mouse uses RMBase 3.0 only; never open short-read files.
            if not s.get('cell') or (s['kind'] == 'short_read' and s['cell'] == '*'):
                raise ValueError('Short-read sources require an explicit cell')
            if s.get('status', 'ready') not in ('ready', 'unavailable'):
                raise ValueError('Reference status must be ready or unavailable')
            if s.get('status', 'ready') == 'unavailable':
                s['status'] = 'unavailable'
                self.sources.append(s)
                continue
            if s['kind'] == 'database' and s.get('database') == 'RMBase' and s.get('version') == '2.0':
                if s.get('original_assembly') != 'hg19' or not s.get('conversion_provenance'):
                    raise ValueError('RMBase 2.0 requires hg19-to-GRCh38 conversion provenance')
            p = (self.path.parent / s['path']).resolve()
            digest = sha256(p)
            if s.get('sha256') and s['sha256'] != digest:
                raise ValueError(f'Reference checksum mismatch: {p}')
            self.sites[s['id']] = load_bed_sites(p)
            if s.get('provenance'):
                provenance = (self.path.parent / s['provenance']).resolve()
                s.update(provenance=str(provenance), provenance_sha256=sha256(provenance))
            s.update(path=str(p), status='ready', sha256=digest, unique_sites=len(self.sites[s['id']]))
            self.sources.append(s)

    def _scope(self, reference_cell, modification):
        """Cache the small source list, without duplicating millions of BED keys."""
        scope = reference_cell, modification
        if scope not in self._scopes:
            sources = tuple(s for s in self.sources if s['modification'] == modification
                            and s['cell'] in (reference_cell, '*'))
            ready = tuple((s['id'], self.sites[s['id']]) for s in sources if s['status'] == 'ready')
            self._scopes[scope] = sources, ready
        return self._scopes[scope]

    def applicable(self, reference_cell, modification):
        return list(self._scope(reference_cell, modification)[0])

    def matches(self, reference_cell, modification, chrom, start, strand):
        """Exact hashed BED-key membership in each applicable source, in source order."""
        key = (chrom, start, strand)
        return [source_id for source_id, sites in self._scope(reference_cell, modification)[1]
                if key in sites]
