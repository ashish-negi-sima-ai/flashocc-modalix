"""Stream official mini archive, retaining metadata and keyframe camera images only."""
import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

URL = 'https://www.nuscenes.org/data/v1.0-mini.tgz'


class Reader:
    def __init__(self, prefix, response):
        self.prefix, self.response, self.network_bytes = prefix, response, 0

    def read(self, size=-1):
        data = self.prefix.read(size) if self.prefix else b''
        if data:
            return data
        data = self.response.read(size)
        self.network_bytes += len(data)
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, help='Optional already-downloaded archive prefix')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    offset = args.prefix.stat().st_size if args.prefix else 0
    request = urllib.request.Request(URL, headers={'Range': f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        if offset and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise RuntimeError('Server did not honor resume offset')
        prefix = args.prefix.open('rb') if args.prefix else None
        reader = Reader(prefix, response)
        metadata, required, extracted = set(), None, set()
        try:
            with tarfile.open(fileobj=reader, mode='r|gz') as archive:
                for member in archive:
                    name = member.name.removeprefix('./')
                    keep = (name.startswith('v1.0-mini/') and name.endswith('.json')) or name.startswith('samples/CAM_')
                    if not member.isfile() or not keep:
                        continue
                    destination = args.output/name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    data = archive.extractfile(member).read()
                    if len(data) != member.size:
                        raise RuntimeError(f'Truncated member: {name}')
                    destination.write_bytes(data)
                    if name.startswith('v1.0-mini/'):
                        metadata.add(name)
                        if name.endswith('/sample_data.json'):
                            required = {r['filename'] for r in json.loads(data)
                                        if r['is_key_frame'] and r['filename'].startswith('samples/CAM_')}
                    else:
                        extracted.add(name)
                        if len(extracted) % 400 == 0:
                            print(f'{len(extracted)} camera images extracted', flush=True)
                    if required and required.issubset(extracted) and len(metadata) >= 13:
                        break
        finally:
            if prefix:
                prefix.close()
    if not required or not required.issubset(extracted):
        raise RuntimeError('Incomplete camera subset')
    result = dict(source=URL, scope='mini metadata and all keyframe camera images; no sweeps or LiDAR',
                  camera_images=len(required), metadata_files=len(metadata),
                  archive_prefix_bytes=offset, additional_network_bytes=reader.network_bytes)
    (args.output/'download-manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result, flush=True)


if __name__ == '__main__':
    main()
