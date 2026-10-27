"""Inspect and check guarded spatial camera packing in the supplied ONNX."""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


def main():
    root = _PROJECT_ROOT
    path = root / 'artifacts/model_prepared.onnx'
    destination = root / 'artifacts/model-prepared-analysis'
    destination.mkdir(parents=True, exist_ok=True)
    packed = onnx.load(path)
    onnx.checker.check_model(packed)
    # Compare the raw head, before the source mask and depth sigmoid.
    head = next(n.input[0] for n in packed.graph.node if n.name == 'Mul_131')
    del packed.graph.output[:]
    packed.graph.output.append(helper.make_tensor_value_info(head, onnx.TensorProto.FLOAT,
                                                            [1, 123, 16, 274]))
    single = copy.deepcopy(packed)
    del single.graph.input[:]
    single.graph.input.append(helper.make_tensor_value_info('camera_input', onnx.TensorProto.FLOAT,
                                                            [1, 3, 256, 704]))
    replace = {'packed_images_with_guards': 'camera_input'}
    nodes = []
    for node in single.graph.node[5:]:
        for i, name in enumerate(node.input):
            node.input[i] = replace.get(name, name)
        if node.op_type == 'Mul':
            replace[node.output[0]] = node.input[0]
        else:
            nodes.append(node)
    del single.graph.node[:]
    single.graph.node.extend(nodes)
    # The FPN's nearest-neighbor resize has a constant output width.
    for tensor in single.graph.initializer:
        if tensor.name == '515':
            tensor.CopyFrom(numpy_helper.from_array(np.array([1, 256, 16, 44], np.int64), '515'))
    single.graph.output[0].type.tensor_type.shape.dim[3].dim_value = 44
    del single.graph.value_info[:]
    # Keep only nodes and constants needed to produce the raw head.
    needed = {head}
    kept = []
    for node in reversed(single.graph.node):
        if any(x in needed for x in node.output):
            kept.append(node)
            needed.update(node.input)
    del single.graph.node[:]
    single.graph.node.extend(reversed(kept))
    constants = [t for t in single.graph.initializer if t.name in needed]
    del single.graph.initializer[:]
    single.graph.initializer.extend(constants)
    onnx.checker.check_model(single)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    packed_session = ort.InferenceSession(packed.SerializeToString(), options,
                                          providers=['CPUExecutionProvider'])
    single_session = ort.InferenceSession(single.SerializeToString(), options,
                                          providers=['CPUExecutionProvider'])
    manifest = json.loads((root / 'artifacts/app-replay-inputs/manifest.json').read_text())
    records = []
    for sample in manifest['samples']:
        token = sample['token']
        cameras = [np.load(root / f'artifacts/real-calibration/camera/validation/{token}-{i}.npy')
                   for i in range(6)]
        # The supplied graph crops rows 140:396. Populate these with our saved
        # normalized images. This checks graph equivalence, not checkpoint accuracy.
        value = np.zeros((1, 6, 3, 396, 704), np.float32)
        value[0, :, :, 140:, :] = np.concatenate(cameras, axis=0)
        actual = packed_session.run([head], {'resized_cameras': value})[0]
        per_camera = []
        for i, camera in enumerate(cameras):
            expected = single_session.run([head], {'camera_input': camera})[0]
            view = actual[:, :, :, 46*i:46*i+44]
            difference = view - expected
            per_camera.append({'camera': i,
                               'relative_rmse': float(np.linalg.norm(difference) / np.linalg.norm(expected)),
                               'max_abs_error': float(np.abs(difference).max())})
        record = {'token': token, 'cameras': per_camera}
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {'model': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'operators': dict(Counter(n.op_type for n in packed.graph.node)),
              'input': [1, 6, 3, 396, 704], 'network_batch': 1,
              'packed_nchw': [1, 3, 256, 4384], 'input_guard_width': 32,
              'output_guard_width': 2, 'raw_head': head, 'depth_channels': 59,
              'feature_channels': 64, 'groups': records,
              'comparison': 'Packed raw head versus per-camera FP32 with the same weights and operators',
              'ground_truth_accuracy_measured': False, 'mla_compiled_or_validated': False}
    (destination / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
