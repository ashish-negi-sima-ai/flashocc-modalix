"""Run the exported M0 split pipeline on preprocessed inputs and pooling ranks."""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from pool_reference import pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True,
                        help='NPZ: images [6,3,256,704], ranks_depth, ranks_feat, ranks_bev')
    parser.add_argument('--models', type=Path, default=Path('artifacts/onnx-modalix'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/prediction.npz'))
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    camera = ort.InferenceSession(str(args.models/'camera.onnx'), options, providers=['CPUExecutionProvider'])
    bev = ort.InferenceSession(str(args.models/'bev.onnx'), options, providers=['CPUExecutionProvider'])
    with np.load(args.inputs, allow_pickle=False) as inputs:
        images = inputs['images']
        if images.shape != (6, 3, 256, 704) or images.dtype != np.float32:
            raise ValueError('images must be float32 [6,3,256,704], in upstream camera order')
        results = [camera.run(None, {'camera_input': images[i:i+1]}) for i in range(6)]
        depth = np.concatenate([r[0] for r in results])
        features = np.concatenate([r[1] for r in results])
        pooled = pool(depth, features, inputs['ranks_depth'], inputs['ranks_feat'], inputs['ranks_bev'])
    logits_nchw = bev.run(None, {'bev_input': pooled})[0]
    logits = logits_nchw.transpose(0, 3, 2, 1).reshape(1, 200, 200, 16, 18)
    labels = logits.argmax(-1).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, labels=labels, bev_features=pooled)
    print(f'Saved labels {labels.shape} and BEV features to {args.output}')


if __name__ == '__main__':
    main()
