"""Install pinned Three.js browser assets locally, retaining its MIT license."""
import base64
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

VERSION='0.180.0'
ROOT=Path(__file__).resolve().parents[1]

def main():
    with urllib.request.urlopen(f'https://registry.npmjs.org/three/{VERSION}',timeout=30) as response:
        metadata=json.load(response)
    with urllib.request.urlopen(metadata['dist']['tarball'],timeout=60) as response:data=response.read()
    algorithm,digest=metadata['dist']['integrity'].split('-',1)
    if algorithm!='sha512' or base64.b64encode(hashlib.sha512(data).digest()).decode()!=digest:
        raise RuntimeError('Three.js archive integrity check failed')
    destination=ROOT/'viewer/vendor';destination.mkdir(parents=True,exist_ok=True)
    files={'build/three.module.min.js':'three.module.min.js','build/three.core.min.js':'three.core.min.js',
           'examples/jsm/controls/OrbitControls.js':'OrbitControls.js','LICENSE':'LICENSE-three.txt'}
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        for source,target in files.items():
            (destination/target).write_bytes(archive.extractfile('package/'+source).read())
    (destination/'version.json').write_text(json.dumps(dict(package='three',version=VERSION,
        integrity=metadata['dist']['integrity'],source=metadata['dist']['tarball']),indent=2)+'\n')
    print(f'Installed Three.js {VERSION} into {destination}')

if __name__=='__main__':main()
