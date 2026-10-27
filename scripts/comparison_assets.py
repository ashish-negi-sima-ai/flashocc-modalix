"""Create or restore a checksummed comparison asset bundle outside Git."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'benchmarks/flashocc_20261027'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def entries():
    files = {r['path']: r for r in json.loads((REFERENCE / 'assets.json').read_text())['files']}
    for r in json.loads((ROOT / 'models/manifest.json').read_text())['models'].values():
        files[r['path']] = r
    return list(files.values())


def verify(root, files):
    for row in files:
        path = root / row['path']
        if not path.is_file() or path.stat().st_size != row['size_bytes'] or digest(path) != row['sha256']:
            raise ValueError('Missing or changed asset: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['archive', 'restore', 'verify'])
    parser.add_argument('--asset-root', type=Path, default=ROOT)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    files = entries()
    if args.action == 'verify':
        verify(args.asset_root, files)
        print(f'Verified {len(files)} assets.')
        return
    if not args.archive:
        parser.error('--archive is required')
    sidecar = Path(str(args.archive) + '.json')
    if args.action == 'archive':
        if args.archive.exists() or sidecar.exists():
            parser.error('Choose a new archive path; existing bundles are immutable')
        verify(args.asset_root, files)
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(str(args.archive) + '.partial')
        try:
            with tarfile.open(partial, 'w:gz', compresslevel=1, dereference=True) as archive:
                for row in files:
                    archive.add(args.asset_root / row['path'], arcname=row['path'], recursive=False)
            partial.rename(args.archive)
        finally:
            partial.unlink(missing_ok=True)
        sidecar.write_text(json.dumps({'sha256': digest(args.archive), 'files': files}, indent=2) + '\n')
    else:
        info = json.loads(sidecar.read_text())
        if digest(args.archive) != info['sha256']:
            raise ValueError('Bundle checksum mismatch')
        if args.asset_root.exists() and any(args.asset_root.iterdir()):
            parser.error('Restore into a new or empty directory')
        args.asset_root.mkdir(parents=True, exist_ok=True)
        expected = {row['path'] for row in info['files']}
        with tarfile.open(args.archive) as archive:
            members = archive.getmembers()
            if {m.name for m in members} != expected or len(members) != len(expected):
                raise ValueError('Archive inventory mismatch')
            for member in members:
                relative = Path(member.name)
                if not member.isfile() or relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Unexpected archive member')
                target = args.asset_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
                target.chmod(member.mode & 0o777)
        verify(args.asset_root, info['files'])
    print(f'{args.action}: {args.archive}; {len(files)} checksummed assets')


if __name__ == '__main__':
    main()
