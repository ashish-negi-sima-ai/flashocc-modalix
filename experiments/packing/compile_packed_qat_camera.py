"""W8A8 PTQ compilation and simulator validation of packed QAT camera models.

This applies the deployed QAT camera recipe (PTQ on Q/DQ-stripped QAT weights,
default W8A8, batch one) to the packed exports from prepare_packed_qat_camera.py.
`--cameras 1` compiles the unpacked normalized export with the same calibration
images, as a control. Calibration uses all six views of the first
`--calibration-samples` QAT calibration samples, grouped like the runtime.

Validation runs the quantized camera model in the AFE simulator on the QAT
validation samples, unpacks its outputs, pools them and runs the deployed
QAT PTQ BEV model. Metrics follow reports/qat/m0-full/ptq-compile/accuracy.json.
Camera outputs and labels are saved so that runs can be compared to each other;
`--reference` compares against such a saved run. `--validate-only` with
`--load-dir` re-evaluates an existing quantized model, such as the deployed one.
"""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
_project_sys.path.insert(0, str(_PROJECT_ROOT / "experiments/packing"))
import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from afe.apis.defines import default_quantization, gen2_target
from afe.apis.loaded_net import load_model
from afe.apis.model import Model
from afe.load.importers.general_importer import ImporterParams, ModelFormat

from compile_qat_ptq_w8a8 import confusion_matrix, occupancy_metrics, OCCUPANCY_CLASSES
from pool_reference import pool as bev_pool
from prepare_packed_camera import pack_images, unpack_output

MODEL_NAME = 'flashocc_m0_camera'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(inputs, split, count=None):
    manifest = json.loads((inputs/'manifest.json').read_text())
    return [r for r in manifest['samples'] if r['split'] == split][:count]


def camera_inputs(images, cameras, guard_width):
    """Six normalized NCHW views -> NHWC model inputs, in runtime call order."""
    if cameras == 1:
        return [view[None].transpose(0, 2, 3, 1) for view in images]
    return [pack_images(images[first:first+cameras], guard_width).transpose(0, 2, 3, 1)
            for first in range(0, 6, cameras)]


def run_camera(model, images, cameras, guard_width):
    """Returns NCHW depth [6,44,16,44] and features [6,64,16,44]."""
    depth, features = [], []
    for value in camera_inputs(images, cameras, guard_width):
        outputs = {np.asarray(o).shape[-1]: np.asarray(o) for o in
                   model.execute(inputs={'camera_input': value}, use_jax=True)}
        for channels, destination in [(44, depth), (64, features)]:
            output = outputs[channels]
            if cameras > 1:
                output = unpack_output(output, cameras, guard_width, layout='NHWC')
            destination.append(output.transpose(0, 3, 1, 2))
    return np.concatenate(depth), np.concatenate(features)


def relative_rmse(actual, expected):
    return float(np.linalg.norm(actual.astype(np.float64)-expected)/np.linalg.norm(expected))


def compile_model(args, model_path):
    print(f'Loading {model_path}', flush=True)
    net = load_model(ImporterParams(format=ModelFormat.onnx, file_paths=[str(model_path)]), target=gen2_target)
    calibration_records = records(args.inputs, 'calibration', args.calibration_samples)
    calibration = []
    for record in calibration_records:
        with np.load(args.inputs/record['path']) as data:
            images = data['images'].astype(np.float32)
        calibration += [{'camera_input': value} for value in camera_inputs(images, args.cameras, args.guard_width)]
    print(f'W8A8 PTQ with {len(calibration)} calibration tensors '
          f'({6*len(calibration_records)} camera images)', flush=True)
    start = time.time()
    quant = net.quantize(calibration_data=calibration, quantization_config=default_quantization,
                         model_name=MODEL_NAME, log_level=logging.INFO)
    quantize_seconds = time.time()-start
    quant.save(model_name=MODEL_NAME, output_directory=str(args.build_dir))
    start = time.time()
    quant.compile(output_path=str(args.build_dir), batch_size=1, log_level=logging.INFO)
    archive = args.build_dir/f'{MODEL_NAME}_mpk.tar.gz'
    status = {'model': str(model_path), 'model_sha256': sha256(model_path),
              'cameras_per_input': args.cameras, 'guard_width': args.guard_width,
              'quantization': 'w8a8 default_quantization, PTQ on QAT weights',
              'calibration_tensors': len(calibration), 'calibration_camera_images': 6*len(calibration_records),
              'calibration_tokens': [r['token'] for r in calibration_records],
              'quantize_seconds': quantize_seconds, 'compile_seconds': time.time()-start,
              'batch_size': 1, 'archive': str(archive), 'archive_sha256': sha256(archive),
              'hardware_validated': False}
    (args.build_dir/'compile_status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps({k: v for k, v in status.items() if k != 'calibration_tokens'}, indent=2), flush=True)
    return quant


def validate(args, camera, fp32_path):
    bev = Model.load('flashocc_m0_bev', str(args.bev_dir))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    fp32 = ort.InferenceSession(str(fp32_path), options, providers=['CPUExecutionProvider'])
    reference = np.load(args.reference/'outputs.npz') if args.reference else None
    masked = np.zeros((OCCUPANCY_CLASSES, OCCUPANCY_CLASSES), np.int64)
    every = np.zeros_like(masked)
    per_sample, saved = [], {}
    for record in records(args.inputs, 'validation', args.validation_samples):
        token = record['token']
        with np.load(args.inputs/record['path']) as data:
            images = data['images'].astype(np.float32)
            ranks = [data[k] for k in ('ranks_depth', 'ranks_feat', 'ranks_bev')]
            semantics = data['semantics'].astype(np.int64)
            mask_camera = data['mask_camera'].astype(bool)
        depth, features = run_camera(camera, images, args.cameras, args.guard_width)
        expected = [np.concatenate(o) for o in zip(*[fp32.run(None, {'camera_input': v[None]}) for v in images])]
        pooled = bev_pool(depth, features, *ranks)
        logits = bev.execute(inputs={'bev_input': pooled.transpose(0, 2, 3, 1)}, use_jax=True)[0]
        # AFE NHWC is (N,H,W,C); swap H and W to match Occ3D (W,H,Z) labels.
        labels = logits.transpose(0, 2, 1, 3).reshape(1, 200, 200, 16, 18).argmax(-1).astype(np.int64)[0]
        sample_masked = confusion_matrix(semantics[mask_camera], labels[mask_camera])
        masked += sample_masked
        every += confusion_matrix(semantics, labels)
        metrics = occupancy_metrics(sample_masked)
        result = {'token': token, 'scene': record['scene'],
                  'camera_masked_accuracy': metrics['accuracy'],
                  'camera_masked_miou': metrics['miou_excluding_free'],
                  'depth_relative_rmse_vs_fp32': relative_rmse(depth, expected[0]),
                  'features_relative_rmse_vs_fp32': relative_rmse(features, expected[1])}
        if reference is not None:
            result.update(depth_relative_rmse_vs_reference=relative_rmse(depth, reference[f'{token}/depth']),
                          features_relative_rmse_vs_reference=relative_rmse(features, reference[f'{token}/features']),
                          label_agreement_vs_reference=float(np.mean(labels == reference[f'{token}/labels'])),
                          depth_bit_exact_vs_reference=bool(np.array_equal(depth, reference[f'{token}/depth'])),
                          features_bit_exact_vs_reference=bool(np.array_equal(features, reference[f'{token}/features'])))
        saved.update({f'{token}/depth': depth, f'{token}/features': features,
                      f'{token}/labels': labels.astype(np.uint8)})
        per_sample.append(result)
        print(json.dumps(result), flush=True)
    np.savez_compressed(args.build_dir/'outputs.npz', **saved)

    def mean(key):
        values = [r[key] for r in per_sample if key in r]
        return float(np.mean(values)) if values else None
    summary = {'protocol': 'Occ3D camera mask; aggregate confusion matrix; mIoU classes 0-16',
               'camera_model': 'AFE simulator (use_jax) of the saved quantized camera model',
               'bev_model': str(args.bev_dir), 'cameras_per_input': args.cameras,
               'camera_masked': occupancy_metrics(masked), 'all_voxels_diagnostic': occupancy_metrics(every),
               'samples': len(per_sample),
               'mean_depth_relative_rmse_vs_fp32': mean('depth_relative_rmse_vs_fp32'),
               'mean_features_relative_rmse_vs_fp32': mean('features_relative_rmse_vs_fp32')}
    if reference is not None:
        summary.update(reference=str(args.reference),
                       mean_depth_relative_rmse_vs_reference=mean('depth_relative_rmse_vs_reference'),
                       mean_features_relative_rmse_vs_reference=mean('features_relative_rmse_vs_reference'),
                       mean_label_agreement_vs_reference=mean('label_agreement_vs_reference'))
    (args.build_dir/'accuracy.json').write_text(json.dumps({'summary': summary, 'per_sample': per_sample}, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    root = _PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cameras', type=int, choices=[1, 2, 3, 6], required=True)
    parser.add_argument('--guard-width', type=int, default=32)
    parser.add_argument('--model', type=Path, help='default: onnx-qat-packedN/camera_spatial.onnx or onnx-qat/camera.onnx')
    parser.add_argument('--build-dir', type=Path, help='default: artifacts/compile-qat-packedN or compile-qat-sequential')
    parser.add_argument('--inputs', type=Path, default=root/'artifacts/qat-inputs')
    parser.add_argument('--calibration-samples', type=int, default=64)
    parser.add_argument('--validation-samples', type=int)
    parser.add_argument('--bev-dir', type=Path, default=root/'artifacts/qat-runs/m0-full/ptq-compile/bev')
    parser.add_argument('--fp32-model', type=Path, default=root/'artifacts/onnx-qat/camera.onnx')
    parser.add_argument('--reference', type=Path, help='Build directory whose outputs.npz is the comparison reference')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--load-dir', type=Path, help='Quantized model directory for --validate-only; default: build dir')
    parser.add_argument('--no-validate', action='store_true')
    args = parser.parse_args()
    if args.model is None:
        args.model = root/('artifacts/onnx-qat/camera.onnx' if args.cameras == 1
                           else f'artifacts/onnx-qat-packed{args.cameras}/camera_spatial.onnx')
    if args.build_dir is None:
        args.build_dir = root/'artifacts'/('compile-qat-sequential' if args.cameras == 1
                                           else f'compile-qat-packed{args.cameras}')
    args.build_dir = args.build_dir/'camera'
    args.build_dir.mkdir(parents=True, exist_ok=True)
    if args.cameras > 1:
        packing = json.loads((args.model.parent/'packing.json').read_text())
        if (packing['cameras'], packing['input_guard_width']) != (args.cameras, args.guard_width) \
                or not packing['fp32_validated'] or sha256(args.model) != packing['files_sha256'][args.model.name]:
            raise ValueError('Model does not match its validated packing metadata')
    if args.validate_only:
        camera = Model.load(MODEL_NAME, str(args.load_dir or args.build_dir))
    else:
        camera = compile_model(args, args.model)
    if not args.no_validate:
        validate(args, camera, args.fp32_model)


if __name__ == '__main__':
    main()
