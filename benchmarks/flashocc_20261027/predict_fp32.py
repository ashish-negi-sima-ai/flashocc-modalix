"""Original FP32 ONNX predictions on the supplied GT input tensors."""
from pathlib import Path
import argparse
import struct
import sys
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gt-dir', type=Path, default=ROOT / 'gt')
parser.add_argument('--camera-model', type=Path, default=ROOT / 'artifacts/onnx-modalix/camera.onnx')
parser.add_argument('--bev-model', type=Path, default=ROOT / 'artifacts/onnx-modalix/bev.onnx')
parser.add_argument('--results', type=Path, default=ROOT / 'artifacts/qat-comparison-repeat')
parser.add_argument('--replay-input-dir', type=Path)
args = parser.parse_args()
OUT = args.results.resolve() / 'gt-replay/fp32'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'scripts'))
from pool_reference import pool

options = ort.SessionOptions()
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
camera = ort.InferenceSession(str(args.camera_model),
                              options, providers=['CPUExecutionProvider'])
bev = ort.InferenceSession(str(args.bev_model),
                           options, providers=['CPUExecutionProvider'])
for p in sorted(args.gt_dir.glob('*.npz')):
    with np.load(p) as a:
        outputs = [camera.run(None, {'camera_input': a['images'][i:i+1]})
                   for i in range(6)]
        depth = np.concatenate([o[0] for o in outputs])
        features = np.concatenate([o[1] for o in outputs])
        pooled = pool(depth, features, a['ranks_depth'], a['ranks_feat'], a['ranks_bev'])
        replay_arrays = ([a['images'].transpose(0, 2, 3, 1), a['ranks_depth'],
                          a['ranks_feat'], a['ranks_bev']] if args.replay_input_dir else None)
    logits = bev.run(None, {'bev_input': pooled})[0]
    assert np.isfinite(logits).all()
    labels = logits.transpose(0, 3, 2, 1).reshape(200, 200, 16, 18).argmax(-1).astype(np.uint8)
    labels.tofile(OUT / (p.stem + '.labels.u8'))
    if replay_arrays is not None:
        args.replay_input_dir.mkdir(parents=True, exist_ok=True)
        with (args.replay_input_dir / (p.stem + '.frame')).open('wb') as stream:
            stream.write(b'FOCCAPP1')
            stream.write(struct.pack('<6Q', 6, 256, 704, len(replay_arrays[1]), labels.size, 0))
            for array, dtype in zip(replay_arrays, ['<f4', '<i4', '<i4', '<i4']):
                stream.write(np.asarray(array, dtype=dtype).tobytes(order='C'))
            stream.write(labels.tobytes())
    reference = ROOT / 'artifacts/real-predictions' / (p.stem + '.npy')
    if reference.exists():
        np.testing.assert_array_equal(labels, np.load(reference).reshape(labels.shape))
    print(p.stem, flush=True)
