"""Verify the bundled demo payloads using only the Python standard library."""
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]


def main():
    demo = ROOT / 'demo'
    manifest = json.loads((demo / 'manifest.json').read_text())
    for row in manifest['files']:
        path = demo / row['path']
        if not path.is_file() or path.stat().st_size != row['size_bytes']:
            raise ValueError(f'Missing or changed asset: {path}')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != row['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')
    config = json.loads((demo / 'inputs/config.json').read_text())
    for clip in config['clips']:
        assert (demo / 'inputs' / clip['file']).is_file(), clip
    for frame in config['frames']:
        path = Path(frame['geometry'])
        assert not path.is_absolute(), 'Bundled geometry paths must be relative'
        assert (demo / 'inputs' / path).is_file(), path
    for key in ('camera_model', 'bev_model'):
        path = Path(config['execution'][key])
        assert not path.is_absolute(), 'Bundled model paths must be relative'
        assert (ROOT / path).is_file(), path
    header = (demo / 'bin/flashocc_insight').read_bytes()[:20]
    assert header[:6] == b'\x7fELF\x02\x01' and struct.unpack_from('<H', header, 18)[0] == 183, 'Expected an ARM64 ELF'
    total = sum(row['size_bytes'] for row in manifest['files'])
    print(f'Verified {len(manifest["files"])} assets ({total / 1024**2:.2f} MiB), relative paths and ARM64 executable.')


if __name__ == '__main__':
    main()
