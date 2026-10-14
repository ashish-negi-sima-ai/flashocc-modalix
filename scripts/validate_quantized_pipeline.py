"""Host-side quantized six-camera pipeline comparison with FP32 predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
from afe.apis.model import Model
from pool_reference import pool


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--count',type=int,default=2)
    parser.add_argument('--camera-dir',type=Path)
    parser.add_argument('--bev-dir',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--executor',choices=['normal','jax'],default='jax')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    camera_directory=args.camera_dir or (root/'artifacts/compile-real/camera')
    camera=Model.load('flashocc_m0_camera',str(camera_directory))
    bev_directory=args.bev_dir or (root/'artifacts/compile-real/bev')
    bev=Model.load('flashocc_m0_bev',str(bev_directory))
    manifest=json.loads((root/'artifacts/real-inputs/manifest.json').read_text())
    records=[r for r in manifest['samples'] if r['split']=='validation']
    # Spread the small check across both holdout scenes.
    indexes=np.linspace(0,len(records)-1,args.count,dtype=int)
    results=[]
    for index in indexes:
        r=records[index]
        with np.load(root/'artifacts/real-inputs'/r['path']) as inputs:
            outputs=[]
            for i in range(6):
                result=camera.execute(inputs={'camera_input':inputs['images'][i:i+1].transpose(0,2,3,1)},use_jax=args.executor=='jax')
                if [a.shape for a in result]!=[(1,16,44,44),(1,16,44,64)]:
                    raise ValueError(f'Unexpected SDK camera output layout: {[a.shape for a in result]}')
                outputs.append(result)
            depth=np.concatenate([o[0].transpose(0,3,1,2) for o in outputs])
            features=np.concatenate([o[1].transpose(0,3,1,2) for o in outputs])
            pooled=pool(depth,features,inputs['ranks_depth'],inputs['ranks_feat'],inputs['ranks_bev'])
        logits=bev.execute(inputs={'bev_input':pooled.transpose(0,2,3,1)},use_jax=args.executor=='jax')[0]
        if logits.shape!=(1,200,200,288):
            raise ValueError(f'Unexpected SDK BEV output layout: {logits.shape}')
        labels=logits.transpose(0,2,1,3).reshape(1,200,200,16,18).argmax(-1).astype(np.uint8)
        reference=np.load(root/'artifacts/real-predictions'/f"{r['token']}.npy")
        occupied=(reference!=17)|(labels!=17)
        result=dict(token=r['token'],scene=r['scene'],
                    all_voxel_label_agreement=float(np.mean(labels==reference)),
                    occupied_union_label_agreement=float(np.mean((labels==reference)[occupied])),
                    occupancy_binary_agreement=float(np.mean((labels==17)==(reference==17))))
        results.append(result)
        np.save(root/'artifacts/real-predictions'/f"{r['token']}-{camera_directory.parent.name}-{bev_directory.parent.name}.npy",labels)
        print(result,flush=True)
    output=args.output or (root/'artifacts/compile-real/pipeline-validation.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    precision=json.loads((camera_directory/'status.json').read_text()).get('precision','int8')
    bev_precision=json.loads((bev_directory/'status.json').read_text()).get('precision','int8')
    output.write_text(json.dumps(
        dict(comparison='host SDK quantized vs ONNX FP32',camera_precision=precision,bev_precision=bev_precision,
             executor=args.executor,ground_truth_accuracy=False,hardware_validated=False,samples=results),indent=2)+'\n')


if __name__=='__main__':
    main()
