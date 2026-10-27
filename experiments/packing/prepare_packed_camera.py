"""Pack the existing FlashOCC camera ONNX spatially, keeping network batch one."""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def pack_images(images, guard_width=32):
    """Normalized NCHW camera images -> one NCHW image with zero guards."""
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[1:] != (3, 256, 704):
        raise ValueError('Expected normalized [cameras,3,256,704] images')
    if guard_width < 32 or guard_width % 32:
        raise ValueError('Guard width must be a positive multiple of 32')
    gap = np.zeros((1, 3, 256, guard_width), dtype=images.dtype)
    parts = []
    for index in range(len(images)):
        if index:
            parts.append(gap)
        parts.append(images[index:index+1])
    return np.concatenate(parts, axis=3)


def unpack_output(value, cameras=6, guard_width=32, layout='NCHW'):
    """Packed depth/features -> contiguous per-camera output for BEV pooling.

    NCHW is the ONNX layout; NHWC is the SDK/runtime layout. The returned
    tensor uses the same layout and has camera index as its first dimension.
    """
    value = np.asarray(value)
    if cameras < 1 or guard_width < 32 or guard_width % 32:
        raise ValueError('Invalid camera count or input guard width')
    gap = guard_width // 16
    width = 44*cameras + gap*(cameras-1)
    if layout == 'NCHW':
        if value.ndim != 4 or value.shape[0] != 1 or value.shape[2:] != (16, width):
            raise ValueError('Unexpected packed NCHW output shape')
        return np.ascontiguousarray(np.concatenate(
            [value[:, :, :, i*(44+gap):i*(44+gap)+44] for i in range(cameras)], axis=0))
    if layout == 'NHWC':
        if value.ndim != 4 or value.shape[:3] != (1, 16, width):
            raise ValueError('Unexpected packed NHWC output shape')
        return np.ascontiguousarray(np.concatenate(
            [value[:, :, i*(44+gap):i*(44+gap)+44, :] for i in range(cameras)], axis=0))
    raise ValueError('layout must be NCHW or NHWC')


def build(source, cameras, guard):
    original = onnx.load(source)
    onnx.checker.check_model(original)
    inferred = onnx.shape_inference.infer_shapes(original, strict_mode=True)
    shapes = {v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
              for v in list(inferred.graph.input) + list(inferred.graph.value_info)
              + list(inferred.graph.output)}
    input_name = original.graph.input[0].name
    if len(original.graph.input) != 1 or shapes[input_name] != (1, 3, 256, 704):
        raise ValueError('This adapter requires our static FlashOCC M0 camera input')
    if [(v.name, shapes[v.name]) for v in original.graph.output] != [
            ('depth', (1, 44, 16, 44)), ('features', (1, 64, 16, 44))]:
        raise ValueError('Unexpected original depth/feature contract')
    constants = {t.name: numpy_helper.to_array(t) for t in original.graph.initializer}
    core = copy.deepcopy(original)
    initializers = list(core.graph.initializer)
    nodes = []
    masks = {}
    guard_sites = []
    prefix = 'flashocc_packing/'

    def packed_width(width):
        if guard * width % 704:
            raise ValueError('Guard does not align with an intermediate feature map')
        return cameras*width + (cameras-1)*(guard*width//704)

    for node in core.graph.node:
        if node.op_type == 'Conv':
            weight = constants[node.input[1]]
            spatial = any(size > 1 for size in weight.shape[2:])
        else:
            spatial = node.op_type == 'MaxPool'
        # The first convolution already receives explicit zero input guards.
        if spatial and node.input[0] != input_name:
            name = node.input[0]
            _, _, height, width = shapes[name]
            key = (height, width)
            if key not in masks:
                mask_name = prefix + f'mask_h{height}_w{width}'
                mask = np.zeros((1, 1, height, packed_width(width)), np.float32)
                step = width + guard*width//704
                for camera in range(cameras):
                    mask[:, :, :, camera*step:camera*step+width] = 1
                initializers.append(numpy_helper.from_array(mask, mask_name))
                masks[key] = mask_name
            result = prefix + f'guarded_{len(guard_sites)}'
            nodes.append(helper.make_node('Mul', [name, masks[key]], [result],
                                          name=prefix + f'guard_{len(guard_sites)}'))
            node.input[0] = result
            guard_sites.append({'consumer': node.name, 'original_input': name,
                                'original_hw': [height, width],
                                'packed_width': packed_width(width)})
        if node.op_type == 'Resize':
            # Retain nearest-neighbor resize semantics; change only target width.
            sizes = constants[node.input[3]].copy()
            sizes[3] = packed_width(int(sizes[3]))
            size_name = prefix + 'resize_sizes'
            initializers.append(numpy_helper.from_array(sizes, size_name))
            node.input[3] = size_name
        nodes.append(node)
    del core.graph.node[:]
    core.graph.node.extend(nodes)
    used = {name for node in nodes for name in node.input}
    del core.graph.initializer[:]
    core.graph.initializer.extend(t for t in initializers if t.name in used)
    del core.graph.value_info[:]
    core.graph.name = 'flashocc_m0_camera_spatial_packed'
    core.graph.input[0].type.tensor_type.shape.dim[3].dim_value = packed_width(704)
    for value in core.graph.output:
        value.type.tensor_type.shape.dim[3].dim_value = packed_width(44)
    core = onnx.shape_inference.infer_shapes(core, strict_mode=True)
    onnx.checker.check_model(core, full_check=True)

    # Five-dimensional scene/camera wrapper. Its CNN is exactly the 4D core.
    scene = copy.deepcopy(core)
    packed_name = prefix + 'packed_images'
    for node in scene.graph.node:
        for index, name in enumerate(node.input):
            if name == input_name:
                node.input[index] = packed_name
    del scene.graph.input[:]
    scene.graph.input.append(helper.make_tensor_value_info(
        input_name, onnx.TensorProto.FLOAT, [1, cameras, 3, 256, 704]))
    transpose_name, rows_name = prefix+'channel_major', prefix+'rows'
    rows_shape, split_sizes, gap_name = prefix+'rows_shape', prefix+'split_sizes', prefix+'input_gap'
    scene.graph.initializer.extend([
        numpy_helper.from_array(np.array([1, 3, cameras*256, 704], np.int64), rows_shape),
        numpy_helper.from_array(np.full(cameras, 256, np.int64), split_sizes),
        numpy_helper.from_array(np.zeros((1, 3, 256, guard), np.float32), gap_name)])
    views = [prefix+f'camera_{i}' for i in range(cameras)]
    parts = []
    for i, view in enumerate(views):
        if i:
            parts.append(gap_name)
        parts.append(view)
    wrapper = [
        helper.make_node('Transpose', [input_name], [transpose_name],
                         perm=[0, 2, 1, 3, 4], name=prefix+'camera_to_channel_major'),
        helper.make_node('Reshape', [transpose_name, rows_shape], [rows_name], name=prefix+'reshape_rows'),
        helper.make_node('Split', [rows_name, split_sizes], views, axis=2, name=prefix+'split_rows'),
        helper.make_node('Concat', parts, [packed_name], axis=3, name=prefix+'concat_with_guards')]
    body = list(scene.graph.node)
    del scene.graph.node[:]
    scene.graph.node.extend(wrapper + body)
    del scene.graph.value_info[:]
    scene.graph.name = f'flashocc_m0_camera_{cameras}_view_packed'
    scene = onnx.shape_inference.infer_shapes(scene, strict_mode=True)
    onnx.checker.check_model(scene, full_check=True)
    # Verify all learned convolution weights and biases are unchanged byte for byte.
    old_nodes = [n for n in original.graph.node if n.op_type == 'Conv']
    new_nodes = [n for n in core.graph.node if n.op_type == 'Conv']
    old_init = {t.name: t.SerializeToString() for t in original.graph.initializer}
    new_init = {t.name: t.SerializeToString() for t in core.graph.initializer}
    for old, new in zip(old_nodes, new_nodes, strict=True):
        if old.input[1:] != new.input[1:] or old.attribute != new.attribute:
            raise ValueError('Convolution parameters changed')
        for name in old.input[1:]:
            if old_init[name] != new_init[name]:
                raise ValueError(f'Learned parameter changed: {name}')
    if len(guard_sites) != 18:
        raise ValueError(f'Unexpected guard-site count: {len(guard_sites)}')
    return scene, core, guard_sites, len(old_nodes)


def main():
    root = _PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root/'artifacts/onnx-modalix/camera.onnx')
    parser.add_argument('--output-dir', type=Path, help='default: artifacts/onnx-packed, or onnx-packedN for N cameras')
    parser.add_argument('--cameras', type=int, default=6)
    parser.add_argument('--guard-width', type=int, default=32)
    args = parser.parse_args()
    if args.cameras < 2 or args.guard_width < 32 or args.guard_width % 32:
        parser.error('Use at least two cameras and guard width a positive multiple of 32')
    if args.output_dir is None:
        args.output_dir = root/'artifacts'/('onnx-packed' if args.cameras == 6 else f'onnx-packed{args.cameras}')
    # Audit sidecars are produced by the model-surgery guard before surgery.
    for dtype in ('int8', 'bfloat16'):
        audit = json.loads(args.source.with_suffix('.'+dtype+'.audit.json').read_text())
        if audit['unsupported'] or audit['unknown']:
            raise ValueError(f'Resolve {dtype} operator audit before surgery')
    scene, core, guards, convolutions = build(args.source, args.cameras, args.guard_width)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, model in [('camera.onnx', scene), ('camera_spatial.onnx', core)]:
        path = args.output_dir/name
        onnx.save(model, path)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'source': str(args.source), 'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
                'files_sha256': files, 'cameras': args.cameras, 'network_batch': 1,
                'scene_input_nkchw': [1, args.cameras, 3, 256, 704],
                'spatial_input_nchw': [1, 3, 256, 704*args.cameras+args.guard_width*(args.cameras-1)],
                'input_name': 'camera_input', 'input_guard_width': args.guard_width,
                'output_guard_width': args.guard_width//16,
                'output_camera_columns': [[i*(44+args.guard_width//16), i*(44+args.guard_width//16)+44]
                                          for i in range(args.cameras)],
                'depth_channels': 44, 'feature_channels': 64, 'depth_activation': 'Softmax(axis=1)',
                'outputs_nchw': {name: [1, channels, 16, 44*args.cameras+(args.guard_width//16)*(args.cameras-1)]
                                 for name, channels in [('depth', 44), ('features', 64)]},
                'unpacked_outputs_nchw': {name: [args.cameras, channels, 16, 44]
                                          for name, channels in [('depth', 44), ('features', 64)]},
                'convolutions_with_unchanged_weights_biases_parameters': convolutions,
                'guards': guards, 'fp32_validated': False, 'mla_compiled': False, 'hardware_validated': False}
    (args.output_dir/'packing.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'guards'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
