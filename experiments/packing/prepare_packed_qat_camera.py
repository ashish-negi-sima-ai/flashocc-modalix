"""Pack the QAT-trained FlashOCC camera ONNX for two, three or six cameras per call.

The source is the Q/DQ-stripped FP32 export that produced the deployed W8A8 PTQ
camera archive. Its export computes the FPN Resize size at runtime from a Shape
chain, keeps dead Identity/Constant leftovers and names the depth output after
its Softmax. These are normalized first: the static size is folded into an
initializer, unreachable nodes are pruned and the output is renamed `depth`.
All learned tensors and remaining nodes are unchanged. The normalized graph is
then passed to the existing guarded spatial packing adapter.

Every output is checked with ONNX Runtime against six separate FP32 calls of
the original stripped graph on held-out QAT validation samples.
"""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
_project_sys.path.insert(0, str(_PROJECT_ROOT / "experiments/packing"))
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

from prepare_packed_camera import build, pack_images, unpack_output


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(source):
    model = onnx.load(source)
    onnx.checker.check_model(model)
    graph = model.graph
    if [i.name for i in graph.input] != ['camera_input'] or len(graph.output) != 2:
        raise ValueError('Expected the QAT FlashOCC camera export')
    resizes = [n for n in graph.node if n.op_type == 'Resize']
    if len(resizes) != 1 or len(resizes[0].input) != 4:
        raise ValueError('Expected one Resize with a sizes input')
    resize = resizes[0]
    # The input shape is static, so the Shape->Slice->Concat size is a constant.
    probe = copy.deepcopy(model)
    probe.graph.output.append(helper.make_tensor_value_info(resize.input[3], onnx.TensorProto.INT64, None))
    session = ort.InferenceSession(probe.SerializeToString(), providers=['CPUExecutionProvider'])
    sizes = session.run([resize.input[3]], {'camera_input': np.zeros((1, 3, 256, 704), np.float32)})[0]
    sizes_name = 'flashocc_qat/resize_sizes'
    graph.initializer.append(numpy_helper.from_array(sizes.astype(np.int64), sizes_name))
    resize.input[3] = sizes_name

    old_depth = graph.output[0].name
    if graph.output[1].name != 'features' or old_depth not in {'depth', '/Softmax_output_0'}:
        raise ValueError(f'Unexpected outputs: {[o.name for o in graph.output]}')
    for node in graph.node:
        node.output[:] = ['depth' if name == old_depth else name for name in node.output]
        node.input[:] = ['depth' if name == old_depth else name for name in node.input]
    graph.output[0].name = 'depth'

    needed = {o.name for o in graph.output}
    kept = []
    for node in reversed(graph.node):
        if any(name in needed for name in node.output):
            kept.append(node)
            needed.update(name for name in node.input if name)
    kept.reverse()
    removed = Counter(n.op_type for n in graph.node) - Counter(n.op_type for n in kept)
    del graph.node[:]
    graph.node.extend(kept)
    initializers = [t for t in graph.initializer if t.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    del graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)

    original = onnx.load(source)
    old_init = {t.name: t.SerializeToString() for t in original.graph.initializer}
    for tensor in model.graph.initializer:
        if tensor.name != sizes_name and old_init[tensor.name] != tensor.SerializeToString():
            raise ValueError(f'Learned parameter changed: {tensor.name}')
    return model, {'resize_sizes': sizes.tolist(), 'removed_nodes': dict(sorted(removed.items()))}


def compare(actual, expected):
    difference = actual.astype(np.float64) - expected
    return {'bit_exact': bool(np.array_equal(actual, expected)),
            'relative_rmse': float(np.linalg.norm(difference) / np.linalg.norm(expected)),
            'max_abs_error': float(np.abs(difference).max())}


def main():
    root = _PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root/'artifacts/qat-runs/m0-full/onnx-stripped/camera.fp32.onnx')
    parser.add_argument('--normalized-dir', type=Path, default=root/'artifacts/onnx-qat')
    parser.add_argument('--output-root', type=Path, default=root/'artifacts',
                        help='Writes onnx-qat-packedN/ for every requested pack size')
    parser.add_argument('--cameras', type=int, nargs='+', choices=[2, 3, 6], default=[2, 3, 6])
    parser.add_argument('--guard-width', type=int, default=32)
    parser.add_argument('--inputs', type=Path, default=root/'artifacts/qat-inputs')
    parser.add_argument('--validation-samples', type=int, default=8)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()

    normalized, changes = normalize(args.source)
    args.normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = args.normalized_dir/'camera.onnx'
    onnx.save(normalized, normalized_path)

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads

    def session(model):
        data = model.SerializeToString() if isinstance(model, onnx.ModelProto) else str(model)
        return ort.InferenceSession(data, options, providers=['CPUExecutionProvider'])

    manifest = json.loads((args.inputs/'manifest.json').read_text())
    records = [r for r in manifest['samples'] if r['split'] == 'validation'][:args.validation_samples]
    groups = []
    for record in records:
        with np.load(args.inputs/record['path']) as data:
            groups.append((record['token'], data['images'].astype(np.float32)))

    print('Running original stripped FP32 reference', flush=True)
    source_session, normalized_session = session(args.source), session(normalized_path)
    references = {}
    for token, views in groups:
        outputs = [source_session.run(None, {'camera_input': view[None]}) for view in views]
        normalized_outputs = [normalized_session.run(None, {'camera_input': view[None]}) for view in views]
        for a, b in zip(outputs, normalized_outputs):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
        references[token] = [np.concatenate([o[i] for o in outputs]) for i in range(2)]
    (args.normalized_dir/'normalization.json').write_text(json.dumps({
        'source': str(args.source), 'source_sha256': sha256(args.source),
        'normalized': str(normalized_path), 'normalized_sha256': sha256(normalized_path),
        'changes': changes, 'learned_parameters_unchanged': True,
        'fp32_bit_exact_vs_source': True, 'checked_camera_images': 6*len(groups)}, indent=2)+'\n')

    for cameras in args.cameras:
        scene, core, guards, convolutions = build(normalized_path, cameras, args.guard_width)
        output_dir = args.output_root/f'onnx-qat-packed{cameras}'
        output_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, model in [('camera.onnx', scene), ('camera_spatial.onnx', core)]:
            onnx.save(model, output_dir/name)
            files[name] = sha256(output_dir/name)
        gap = args.guard_width//16
        metadata = {'source': str(normalized_path), 'source_sha256': sha256(normalized_path),
                    'qat_stripped_source': str(args.source), 'qat_stripped_source_sha256': sha256(args.source),
                    'files_sha256': files, 'cameras': cameras, 'network_batch': 1,
                    'scene_input_nkchw': [1, cameras, 3, 256, 704],
                    'spatial_input_nchw': [1, 3, 256, 704*cameras+args.guard_width*(cameras-1)],
                    'input_name': 'camera_input', 'input_guard_width': args.guard_width,
                    'output_guard_width': gap,
                    'output_camera_columns': [[i*(44+gap), i*(44+gap)+44] for i in range(cameras)],
                    'depth_channels': 44, 'feature_channels': 64, 'depth_activation': 'Softmax(axis=1)',
                    'outputs_nchw': {name: [1, channels, 16, 44*cameras+gap*(cameras-1)]
                                     for name, channels in [('depth', 44), ('features', 64)]},
                    'convolutions_with_unchanged_weights_biases_parameters': convolutions,
                    'guards': guards, 'operator_audit': 'not run; AFE W8A8 compilation is the check',
                    'mla_compiled': False, 'hardware_validated': False}

        print(f'Validating {cameras}-camera packing', flush=True)
        scene_session, spatial_session = session(scene), session(core)
        samples = []
        for token, views in groups:
            packs = [views[first:first+cameras] for first in range(0, 6, cameras)]
            scene_outputs = [scene_session.run(None, {'camera_input': pack[None]}) for pack in packs]
            spatial_outputs = [spatial_session.run(None, {'camera_input': pack_images(pack, args.guard_width)})
                               for pack in packs]
            record = {'token': token}
            for i, name in enumerate(('depth', 'features')):
                for a, b in zip(scene_outputs, spatial_outputs, strict=True):
                    np.testing.assert_array_equal(a[i], b[i])
                unpacked = np.concatenate([unpack_output(o[i], cameras, args.guard_width) for o in spatial_outputs])
                np.testing.assert_allclose(unpacked, references[token][i], atol=1e-6, rtol=1e-5)
                record[name] = compare(unpacked, references[token][i])
            samples.append(record)
            print(json.dumps(record), flush=True)
        report = {'comparison': 'Packed QAT FP32 ONNX versus six separate calls of the stripped QAT FP32 ONNX',
                  'cameras_per_input': cameras, 'calls_per_six_camera_group': 6//cameras,
                  'atol': 1e-6, 'rtol': 1e-5, 'groups': len(samples), 'camera_images': 6*len(samples),
                  'scene_vs_spatial_bit_exact': True,
                  'all_outputs_bit_exact': all(r[n]['bit_exact'] for r in samples for n in ('depth', 'features')),
                  'passed': True, 'samples': samples}
        (output_dir/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
        metadata.update(fp32_validated=True, validation_report='validation.json')
        (output_dir/'packing.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'samples'}), flush=True)


if __name__ == '__main__':
    main()
