"""Export real NHWC camera outputs, remapped ranks and NumPy pooling references."""
import argparse
import json
import struct
from pathlib import Path
import numpy as np
import onnxruntime as ort
from pool_reference import pool

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--count',type=int,default=8)
    args=parser.parse_args()
    options=ort.SessionOptions();options.intra_op_num_threads=4
    camera=ort.InferenceSession(str(ROOT/'artifacts/onnx-modalix/camera.onnx'),options,providers=['CPUExecutionProvider'])
    manifest=json.loads((ROOT/'artifacts/real-inputs/manifest.json').read_text())
    destination=ROOT/'artifacts/pool-fixtures';destination.mkdir(exist_ok=True)
    records=[]
    for record in [r for r in manifest['samples'] if r['split']=='validation'][:args.count]:
        with np.load(ROOT/'artifacts/real-inputs'/record['path']) as inputs:
            outputs=[camera.run(None,{'camera_input':inputs['images'][i:i+1]}) for i in range(6)]
            depth=np.concatenate([o[0] for o in outputs])
            features=np.concatenate([o[1] for o in outputs])
            expected=pool(depth,features,inputs['ranks_depth'],inputs['ranks_feat'],inputs['ranks_bev'])
            # Validate saved FP32 BEV cache as a second integration reference.
            cached=np.load(ROOT/'artifacts/real-calibration/bev/validation'/f"{record['token']}.npy")
            np.testing.assert_array_equal(expected,cached)
            rd=inputs['ranks_depth'].astype(np.int64)
            mapped=(rd//(44*704))*(44*704)+(rd%704)*44+(rd%(44*704))//704
            # Verify remapping against the original depth indexing for every contribution.
            depth_nhwc=depth.transpose(0,2,3,1).copy()
            np.testing.assert_array_equal(depth_nhwc.ravel()[mapped],depth.ravel()[rd])
            path=destination/f"{record['token']}.pool"
            with path.open('wb') as f:
                f.write(b'FOCCP01\0')
                f.write(struct.pack('<5Q',depth.size,6*16*44,40000,64,len(rd)))
                for value,dtype in [(depth_nhwc,'<f4'),(features.transpose(0,2,3,1),'<f4'),
                                    (mapped,'<i4'),(inputs['ranks_feat'],'<i4'),(inputs['ranks_bev'],'<i4'),
                                    (expected.transpose(0,2,3,1),'<f4')]:
                    f.write(np.asarray(value,dtype=dtype).tobytes(order='C'))
            records.append(dict(token=record['token'],scene=record['scene'],file=path.name,
                                contributions=len(rd),depth_layout='NHWC',features_layout='NHWC',output_layout='NHWC'))
            print(records[-1],flush=True)
    (destination/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
