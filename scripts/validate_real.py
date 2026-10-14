"""Build real calibration tensors and validate the split FP32 pipeline."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch
from export_flashocc import M0, ROOT
from pool_reference import pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, default=ROOT/'artifacts/real-inputs')
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = M0().eval()
    checkpoint = torch.load(ROOT/'artifacts/flashocc-r50-M0.pth', weights_only=True, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    camera = ort.InferenceSession(str(ROOT/'artifacts/onnx-modalix/camera.onnx'), options, providers=['CPUExecutionProvider'])
    bev = ort.InferenceSession(str(ROOT/'artifacts/onnx-modalix/bev.onnx'), options, providers=['CPUExecutionProvider'])
    manifest = json.loads((args.inputs/'manifest.json').read_text())
    cache = ROOT/'artifacts/real-calibration'
    for stage in ['camera', 'bev']:
        for split in ['calibration', 'validation']:
            (cache/stage/split).mkdir(parents=True, exist_ok=True)
    reports = []
    for record in manifest['samples']:
        token, split = record['token'], record['split']
        with np.load(args.inputs/record['path']) as inp:
            images = inp['images']
            outputs = [camera.run(None, {'camera_input': images[i:i+1]}) for i in range(6)]
            depth = np.concatenate([o[0] for o in outputs])
            features = np.concatenate([o[1] for o in outputs])
            with torch.no_grad():
                ref_depth, ref_features = model.camera(torch.from_numpy(images))
            np.testing.assert_allclose(depth, ref_depth.numpy(), atol=1e-3, rtol=1e-3)
            np.testing.assert_allclose(features, ref_features.numpy(), atol=1e-3, rtol=1e-3)
            pooled = pool(depth, features, inp['ranks_depth'], inp['ranks_feat'], inp['ranks_bev'])
            # Independent torch accumulation of the weighted contributions.
            f = torch.from_numpy(features).permute(0,2,3,1).reshape(-1,64)
            d = torch.from_numpy(depth).reshape(-1)
            rd, rf, rb = [torch.from_numpy(inp[n].astype(np.int64)) for n in ['ranks_depth','ranks_feat','ranks_bev']]
            ref_pool = torch.zeros(40000,64)
            ref_pool.index_add_(0, rb, d[rd,None]*f[rf])
            ref_pool = ref_pool.reshape(200,200,64).permute(2,0,1)[None]
            np.testing.assert_allclose(pooled,ref_pool.numpy(),rtol=1e-4,atol=1e-4)
        logits4 = bev.run(None, {'bev_input': pooled})[0]
        logits = logits4.transpose(0,3,2,1).reshape(1,200,200,16,18)
        with torch.no_grad():
            reference = model.bev(torch.from_numpy(pooled)).numpy()
        np.testing.assert_allclose(logits, reference, rtol=1e-3, atol=1e-3)
        labels = logits.argmax(-1).astype(np.uint8)
        for i in range(6):
            np.save(cache/'camera'/split/f'{token}-{i}.npy', images[i:i+1])
        np.save(cache/'bev'/split/f'{token}.npy', pooled)
        if split == 'validation':
            (ROOT/'artifacts/real-predictions').mkdir(exist_ok=True)
            np.save(ROOT/'artifacts/real-predictions'/f'{token}.npy', labels)
        report = dict(token=token, scene=record['scene'], split=split,
                      camera_max_abs_error=float(np.max(np.abs(features-ref_features.numpy()))),
                      depth_max_abs_error=float(np.max(np.abs(depth-ref_depth.numpy()))),
                      pooling_max_abs_error=float(np.max(np.abs(pooled-ref_pool.numpy()))),
                      logits_max_abs_error=float(np.max(np.abs(logits-reference))),
                      label_agreement=float(np.mean(labels==reference.argmax(-1))),
                      bev_min=float(pooled.min()),bev_max=float(pooled.max()))
        reports.append(report)
        print(json.dumps(report), flush=True)
    result = dict(dataset=manifest['dataset'], comparison='ONNX vs dependency-adapted PyTorch; pooling vs torch index_add',
                  accuracy_ground_truth_available=False, samples=reports)
    (cache/'validation.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
