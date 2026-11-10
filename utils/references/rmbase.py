"""Import RMBase 3.0 hg38/mm10 BED archives without extracting tar paths."""
import csv
import io
import json
import os
import tarfile
from collections import Counter
from pathlib import Path
from utils.references.catalog import sha256

ARCHIVES = {
    'm6A': ('m6A', 'm6A'),
    'm5C': ('m5C', 'm5C'),
    'inosine': ('RNA-editing', 'A-I'),
    'pseU': ('Pseudo', 'Y'),
}
GENOMES = {
    'hg38': dict(species='Homo sapiens', prefix='human.hg38', assembly='GRCh38', columns=29,
                 member_suffix='result.col29.bed'),
    'mm10': dict(species='Mus musculus', prefix='mouse.mm10', assembly='mm10', columns=25,
                 member_suffix='allSite.uniqID.anno.score.conserved.guide.resort.col25.bed'),
}


def import_rmbase3(source_dir, out, *, genome='hg38'):
    if genome not in GENOMES:
        raise ValueError('RMBase import supports hg38 or mm10')
    settings = GENOMES[genome]
    source_dir, out = Path(source_dir), Path(out)
    paths = {mod: source_dir / f'{genome}.{name}.tar.gz' for mod, (name, _) in ARCHIVES.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    out.mkdir(parents=True, exist_ok=False)
    reports = []
    try:
        for mod, (name, accepted_type) in ARCHIVES.items():
            path = paths[mod]
            expected_member = f"{settings['prefix']}.{name}.{settings['member_suffix']}"
            seen, observed, member_count = set(), Counter(), 0
            bed = out / f"{mod}.{settings['assembly']}.bed"
            with bed.open('w', encoding='utf-8') as dest, \
                    (out / f'{mod}.provenance.tsv').open('w', encoding='utf-8', newline='') as audit, \
                    tarfile.open(path, 'r:gz') as archive:
                writer = csv.writer(audit, delimiter='\t')
                writer.writerow(['chrom', 'start', 'end', 'strand', 'site_id', 'modification_type',
                                 'project_ids', 'dataset_ids', 'pubmed_ids', 'biosample_text', 'method_text'])
                for member in archive:
                    if not member.isfile():
                        continue
                    if member.name != expected_member:
                        raise ValueError(f'Unexpected RMBase archive member: {member.name}')
                    member_count += 1
                    if member_count != 1:
                        raise ValueError('Duplicate archive member')
                    with io.TextIOWrapper(archive.extractfile(member), encoding='utf-8-sig') as handle:
                        for number, line in enumerate(handle, 1):
                            if not line.strip() or line.startswith('#'):
                                continue
                            c = line.rstrip('\r\n').split('\t')
                            if len(c) != settings['columns']:
                                raise ValueError(f"{path}:{number}: expected RMBase 3.0 {settings['columns']}-column BED")
                            start, end = int(c[1]), int(c[2])
                            if start < 0 or end != start + 1 or c[5] not in ('+', '-'):
                                raise ValueError(f'{path}:{number}: invalid single-base BED coordinates')
                            observed[c[6]] += 1
                            if c[6] != accepted_type:
                                # Never silently equate all RNA editing with inosine.
                                continue
                            key = c[0], start, c[5]
                            writer.writerow([c[0], start, end, c[5], c[3], c[6], *c[8:13]])
                            if key not in seen:
                                dest.write(f'{c[0]}\t{start}\t{end}\t{c[3]}\t0\t{c[5]}\n')
                                seen.add(key)
            if member_count != 1 or not seen:
                raise ValueError(f'{path}: no accepted modification sites')
            reports.append(dict(modification=mod, archive=str(path.resolve()), archive_sha256=sha256(path),
                                member=expected_member, accepted_type=accepted_type, observed_types=dict(observed),
                                unique_sites=len(seen), duplicate_rows=observed[accepted_type]-len(seen),
                                bed=bed.name, bed_sha256=sha256(bed)))
        report = dict(database='RMBase', version='3.0', species=settings['species'], assembly=settings['assembly'],
                      coordinates='BED0', original_assembly=genome,
                      conversion='none (hg38 = GRCh38)' if genome == 'hg38' else 'none (native mm10)',
                      source_url='https://rna.sysu.edu.cn/rmbase3/download.php', files=reports,
                      cell_scope=f"all {settings['species']} biosamples; not sample-specific ground truth")
        (out / 'import.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return report
    except Exception as exc:
        (out / 'FAILED.txt').write_text(str(exc), encoding='utf-8')
        raise


def prepare_rmbase3_catalog(source_dir, out, catalog_path, *, genome='mm10'):
    """Build reusable BEDs and a database-only catalog; preserve ready imports."""
    if genome not in GENOMES:
        raise ValueError('RMBase import supports hg38 or mm10')
    out, catalog_path = Path(out), Path(catalog_path)
    if out.exists():
        if (out / 'FAILED.txt').exists():
            raise ValueError(f'Incomplete RMBase import: {out}; choose a new output directory')
        report = json.loads((out / 'import.json').read_text())
        if report['version'] != '3.0' or report['assembly'] != GENOMES[genome]['assembly']:
            raise ValueError('Existing RMBase import uses another version or assembly')
        for item in report['files']:
            if sha256(out / item['bed']) != item['bed_sha256']:
                raise ValueError('Existing RMBase BED checksum mismatch')
    else:
        report = import_rmbase3(source_dir, out, genome=genome)
    sources = []
    for item in report['files']:
        sources.append(dict(id=f"RMBase3.{genome}.{item['modification']}", kind='database',
            database='RMBase', version='3.0', cell='*', modification=item['modification'],
            assembly=report['assembly'], original_assembly=genome, coordinates='BED0',
            path=Path(os.path.relpath(out / item['bed'], catalog_path.parent)).as_posix(),
            sha256=item['bed_sha256'],
            provenance=Path(os.path.relpath(out / 'import.json', catalog_path.parent)).as_posix()))
    manifest = dict(assembly=report['assembly'], sources=sources)
    if catalog_path.exists():
        if json.loads(catalog_path.read_text()) != manifest:
            raise ValueError(f'Existing catalog differs; choose a new catalog path: {catalog_path}')
    else:
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return report
