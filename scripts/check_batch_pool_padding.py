"""Measure whether the compiler's observed padding change explains batch-six errors."""
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def relative_rmse(actual, expected):
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'artifacts/compile-batch6/diagnostics/compiler-source'
    output.mkdir(parents=True, exist_ok=True)
    model = onnx.load(root / 'artifacts/onnx-modalix/camera.onnx')
    pools = [n for n in model.graph.node if n.op_type == 'MaxPool']
    assert len(pools) == 1
    pads = next(a for a in pools[0].attribute if a.name == 'pads')
    assert list(pads.ints) == [1, 1, 1, 1]
    pads.ints[:] = [0, 0, 1, 1]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(model.SerializeToString(), options,
                                   providers=['CPUExecutionProvider'])
    manifest = json.loads((root / 'artifacts/app-replay-inputs/manifest.json').read_text())
    records = []
    for sample in manifest['samples']:
        token = sample['token']
        views = []
        for camera in range(6):
            value = np.load(root / f'artifacts/real-calibration/camera/validation/{token}-{camera}.npy')
            views.append(session.run(None, {'camera_input': value}))
        record = {'token': token}
        for index, name in enumerate(('depth', 'features')):
            shifted = np.concatenate([v[index].transpose(0, 2, 3, 1) for v in views])
            actual = np.fromfile(root / f'artifacts/compile-batch6/batch6/{token}.{name}.f32', '<f4').reshape(shifted.shape)
            original = np.fromfile(root / f'artifacts/compile-batch6/reference/{token}.{name}.f32', '<f4').reshape(shifted.shape)
            record[name] = {'mla_vs_original_fp32': relative_rmse(actual, original),
                            'mla_vs_shifted_pool_fp32': relative_rmse(actual, shifted),
                            'shifted_pool_fp32_vs_original': relative_rmse(shifted, original)}
        records.append(record)
        print(json.dumps(record), flush=True)
    (output / 'padding-causality.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
