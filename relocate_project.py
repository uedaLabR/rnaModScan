"""One-time, verified consolidation requested by the project owner."""
import hashlib
import json
import os
import stat
from pathlib import Path

DEST = Path(r'C:\project\rnaMosScan').resolve()
SOURCES = [Path(r'C:\Users\hirok\OneDrive\デスクトップ\project\rnaModScan').resolve(),
           Path(r'C:\Users\hirok\OneDrive\ドキュメント\ChatGPT\rnaModScan').resolve()]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert str(DEST) == r'C:\project\rnaMosScan'
    records = []
    planned = {}
    for root in SOURCES:
        assert root.name == 'rnaModScan' and root != DEST and root not in DEST.parents
        for path in root.rglob('*'):
            if path.is_symlink():
                raise ValueError(f'Refusing symbolic link: {path}')
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            target = (DEST / relative).resolve()
            if not target.is_relative_to(DEST) or not path.resolve().is_relative_to(root):
                raise ValueError('Path escaped project bounds')
            other = target if target.exists() else planned.get(str(relative).lower())
            if other is not None and digest(path) != digest(other):
                raise ValueError(f'Conflicting file, nothing moved: {path} / {other}')
            planned[str(relative).lower()] = path
    for root in SOURCES:
        paths = list(root.rglob('*'))
        for path in paths:
            if not path.is_file():
                continue
            target = (DEST / path.relative_to(root)).resolve()
            if not path.resolve().is_relative_to(root) or not target.is_relative_to(DEST):
                raise ValueError('Path escaped project bounds')
            before = digest(path)
            size = path.stat().st_size
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target) != before:
                    raise ValueError(f'Conflict: {target}')
                try:
                    path.unlink()
                except PermissionError:
                    path.chmod(stat.S_IWRITE)
                    path.unlink()
                action = 'removed_identical_duplicate'
            else:
                os.rename(path, target)
                if digest(target) != before:
                    raise ValueError(f'Post-move checksum mismatch: {target}')
                action = 'moved'
            records.append(dict(source=str(path), destination=str(target), bytes=size, sha256=before, action=action))
        for folder in sorted((p for p in paths if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            if not folder.resolve().is_relative_to(root):
                raise ValueError('Directory escaped project bounds')
            try:
                folder.rmdir()
            except PermissionError:
                folder.chmod(stat.S_IWRITE)
                folder.rmdir()
        try:
            root.chmod(stat.S_IWRITE)
            root.rmdir()
        except OSError:
            # An empty project root can be held open by the desktop app.
            if any(root.iterdir()):
                raise
    catalog_path = DEST / 'data/references/catalog.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    short = DEST / 'data/reference_sites_GRCh38'
    for entry in catalog['sources']:
        if entry['kind'] == 'short_read':
            entry['path'] = str(short / 'by_method' / Path(entry['path']).name)
            entry['provenance'] = str(short / 'sources.json')
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding='utf-8')
    script = DEST / 'prepare_reference_catalog.py'
    text = script.read_text(encoding='utf-8')
    text = text.replace("SHORT_READ_DIR = Path(r'C:\\Users\\hirok\\OneDrive\\デスクトップ\\project\\rnaModScan\\data\\reference_sites_GRCh38')",
                        "SHORT_READ_DIR = ROOT / 'data/reference_sites_GRCh38'")
    script.write_text(text, encoding='utf-8')
    for rel in ['docs/CALL_SUMMARY.md', 'data/references/README.md']:
        p = DEST / rel
        text = p.read_text(encoding='utf-8').replace('デスクトップ側プロジェクトの `data/reference_sites_GRCh38`',
            '`C:\\project\\rnaMosScan\\data\\reference_sites_GRCh38`').replace(
            'Short-read references remain in the original desktop project directory.',
            r'Short-read references: `C:\project\rnaMosScan\data\reference_sites_GRCh38`.')
        p.write_text(text, encoding='utf-8')
    handoff = DEST / 'HANDOFF.md'
    text = handoff.read_text(encoding='utf-8')
    note = ('\n## 2026-10-23 保存先移動\n\n'
            '現在のプロジェクト本体は `C:\\project\\rnaMosScan`。OneDrive上の元プロジェクトと作業コピーは内容確認後に統合・移動済み。\n'
            'short-read参照もこのプロジェクトの `data/reference_sites_GRCh38` に移動。原本・整形済みRMBase・集計出力も同じプロジェクト内。\n'
            '今後はこのフォルダーを開いて作業すること。`TRANSFER_MANIFEST.json` は元の転送時点の記録であり現在のファイル一覧ではない。\n')
    first, remainder = text.split('\n', 1)
    handoff.write_text(first + '\n' + note + '\n' + remainder, encoding='utf-8')
    log = DEST / 'migration_2026-10-23.json'
    log.write_text(json.dumps(dict(destination=str(DEST), records=records), indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(files_verified=len(records), source_bytes_removed=sum(r['bytes'] for r in records), destination=str(DEST))))


if __name__ == '__main__':
    main()
