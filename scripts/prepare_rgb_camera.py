"""Create an equivalent RGB-input camera ONNX for EV74's native NV12/RGB path."""
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'artifacts/onnx-modalix/camera.onnx'
    output = root / 'artifacts/onnx-rgb'
    output.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    first = next(node for node in model.graph.node if node.op_type == 'Conv')
    assert first.input[0] == model.graph.input[0].name
    weight = next(w for w in model.graph.initializer if w.name == first.input[1])
    value = numpy_helper.to_array(weight)
    assert value.ndim == 4 and value.shape[1] == 3
    weight.CopyFrom(numpy_helper.from_array(value[:, ::-1].copy(), weight.name))
    onnx.checker.check_model(model)
    path = output / 'camera.onnx'
    onnx.save(model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    old = ort.InferenceSession(str(source), options, providers=['CPUExecutionProvider'])
    new = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    validation = sorted((root / 'artifacts/real-calibration/camera/validation').glob('*.npy'))
    reports = []
    for p in validation[::4]:
        x = np.load(p)
        a = old.run(None, {'camera_input': x})
        b = new.run(None, {'camera_input': x[:, ::-1].copy()})
        errors = []
        for ref, actual in zip(a, b):
            np.testing.assert_allclose(ref, actual, atol=1e-3, rtol=1e-3)
            errors.append(float(np.max(np.abs(ref - actual))))
        reports.append({'input': str(p), 'max_absolute_errors': errors})
    (output / 'equivalence.json').write_text(json.dumps(reports, indent=2) + '\n')
    for split in ['calibration', 'validation']:
        destination = root / 'artifacts/real-calibration-rgb/camera' / split
        destination.mkdir(parents=True, exist_ok=True)
        for p in sorted((root / 'artifacts/real-calibration/camera' / split).glob('*.npy')):
            np.save(destination / p.name, np.load(p)[:, ::-1].copy())
    print('RGB ONNX equivalence passed on', len(reports), 'holdout images', flush=True)


if __name__ == '__main__':
    main()
