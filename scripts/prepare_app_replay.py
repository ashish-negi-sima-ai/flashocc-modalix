"""Package already-preprocessed six-camera samples for the C++ replay application."""
import json
import struct
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'artifacts/real-inputs/manifest.json').read_text())
    destination=ROOT/'artifacts/app-replay-inputs';destination.mkdir(exist_ok=True)
    records=[]
    for record in [r for r in manifest['samples'] if r['split']=='validation']:
        token=record['token']
        fp32=np.load(ROOT/'artifacts/real-predictions'/f'{token}.npy').astype(np.uint8)
        sdk_path=ROOT/'artifacts/real-predictions'/f'{token}-compile-real-compile-real-mixed.npy'
        sdk=np.load(sdk_path).astype(np.uint8) if sdk_path.exists() else np.empty(0,np.uint8)
        with np.load(ROOT/'artifacts/real-inputs'/record['path']) as inputs:
            images=inputs['images'].transpose(0,2,3,1).copy()
            rd=inputs['ranks_depth'];rf=inputs['ranks_feat'];rb=inputs['ranks_bev']
            assert images.shape==(6,256,704,3) and np.isfinite(images).all()
            path=destination/f'{token}.frame'
            with path.open('wb') as f:
                f.write(b'FOCCAPP1')
                f.write(struct.pack('<6Q',6,256,704,len(rd),fp32.size,sdk.size))
                for array,dtype in [(images,'<f4'),(rd,'<i4'),(rf,'<i4'),(rb,'<i4'),(fp32,'u1'),(sdk,'u1')]:
                    f.write(np.asarray(array,dtype=dtype).tobytes(order='C'))
        records.append(dict(token=token,scene=record['scene'],file=path.name,host_sdk_reference=bool(sdk.size)))
        print(records[-1])
    (destination/'manifest.json').write_text(json.dumps(dict(camera_order=manifest['camera_order'],
                preprocessing='upstream resize/crop/normalization and calibrated geometry already applied',
                reference_kind='FP32 predictions, plus host quantized predictions where available; not ground truth',
                samples=records),indent=2)+'\n')


if __name__=='__main__':main()
