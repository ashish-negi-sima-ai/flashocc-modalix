"""Compile QAT QDQ ONNX models in w8a8 and report AFE simulated quantized accuracy.

Uses SkipCalibration so that the scale/zero-point from the QDQ ONNX nodes are
used directly — no calibration dataset required.

Outputs to:
  qat/runs/m0-full/afe-compile/{camera,bev}/

Accuracy is reported against the 32 validation samples in
  artifacts/qat-inputs/validation/
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np

from afe.apis.defines import default_quantization, gen2_target
from afe.apis.loaded_net import load_model
from afe.apis.model import Model
from afe.load.importers.general_importer import ImporterParams, ModelFormat

ROOT = Path(__file__).resolve().parents[1]
QAT_DIR = ROOT / 'qat/runs/m0-full'
ONNX_DIR = QAT_DIR / 'onnx'
OUT_DIR = QAT_DIR / 'afe-compile'
INPUTS_DIR = ROOT / 'artifacts/qat-inputs'

# For QAT QDQ ONNX: use default w8a8 quantization but pass a single synthetic calibration
# sample.  AFE's _QDQObserver uses the embedded QDQ scales and ignores the actual tensor
# values, so the synthetic data only serves to trigger InsertNodeObservers.
W8A8 = default_quantization

_CAMERA_SHAPE_NHWC = (1, 256, 704, 3)  # input (1,3,256,704) NCHW → NHWC
_BEV_SHAPE_NHWC = (1, 200, 200, 64)    # input (1,64,200,200) NCHW → NHWC
_STAGE_SHAPES = {'camera': ('camera_input', _CAMERA_SHAPE_NHWC),
                 'bev': ('bev_input', _BEV_SHAPE_NHWC)}
OCCUPANCY_CLASSES = 18
FREE_CLASS = 17


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    # Promote uint8 labels before multiplying by the class count.
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    bins = (labels * OCCUPANCY_CLASSES + predictions).reshape(-1)
    return np.bincount(
        bins, minlength=OCCUPANCY_CLASSES**2
    ).reshape(OCCUPANCY_CLASSES, OCCUPANCY_CLASSES)


def occupancy_metrics(confusion: np.ndarray) -> dict:
    diagonal = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - diagonal
    present = union[:FREE_CLASS] > 0
    return {
        'accuracy': float(diagonal.sum() / max(confusion.sum(), 1)),
        'miou_excluding_free': float(np.mean(
            diagonal[:FREE_CLASS][present] / union[:FREE_CLASS][present]
        )),
        'voxels': int(confusion.sum()),
    }


def compile_stage(stage: str) -> Model:
    model_path = ONNX_DIR / f'{stage}.qdq.onnx'
    out = OUT_DIR / stage
    out.mkdir(parents=True, exist_ok=True)
    print(f'[{stage}] loading {model_path}', flush=True)
    net = load_model(
        ImporterParams(format=ModelFormat.onnx, file_paths=[str(model_path)]),
        target=gen2_target,
    )
    inp_name, shape = _STAGE_SHAPES[stage]
    rng = np.random.default_rng(17)
    calib = [{inp_name: rng.standard_normal(shape).astype(np.float32)}]
    print(f'[{stage}] quantizing (w8a8, QDQ hints + 1 synthetic sample)', flush=True)
    quant = net.quantize(
        calibration_data=calib,
        quantization_config=W8A8,
        model_name=f'flashocc_m0_{stage}',
        log_level=logging.INFO,
    )
    quant.save(model_name=f'flashocc_m0_{stage}', output_directory=str(out))
    print(f'[{stage}] compiling', flush=True)
    quant.compile(output_path=str(out), batch_size=1, log_level=logging.INFO)
    (out / 'compile_status.json').write_text(json.dumps(
        {'stage': stage, 'quantization': 'w8a8',
         'calibration': 'QDQ hints (synthetic trigger sample)',
         'compiled': True}, indent=2))
    print(f'[{stage}] done -> {out}', flush=True)
    return quant


def validate(camera_model: Model, bev_model: Model, count: int | None = None) -> dict:
    from pool_reference import pool as bev_pool

    manifest = json.loads((INPUTS_DIR / 'manifest.json').read_text())
    records = [r for r in manifest['samples'] if r['split'] == 'validation']
    if count is not None:
        records = records[:count]

    results = []
    masked_confusion = np.zeros(
        (OCCUPANCY_CLASSES, OCCUPANCY_CLASSES), dtype=np.int64
    )
    all_confusion = np.zeros_like(masked_confusion)
    for r in records:
        npz_path = INPUTS_DIR / r['path']
        with np.load(npz_path) as data:
            images = data['images']               # (6, 3, 256, 704) float32 NCHW
            ranks_depth = data['ranks_depth']
            ranks_feat = data['ranks_feat']
            ranks_bev = data['ranks_bev']
            semantics = data['semantics'].astype(np.int64)
            mask_camera = data['mask_camera'].astype(bool)

        # Run camera for all 6 images
        depths, features = [], []
        for i in range(6):
            img_nhwc = images[i:i+1].transpose(0, 2, 3, 1)  # (1, 256, 704, 3) NHWC
            out = camera_model.execute(inputs={'camera_input': img_nhwc}, use_jax=True)
            # outputs are NHWC; convert back to NCHW for pooling
            depths.append(out[0].transpose(0, 3, 1, 2))    # (1, 44, 16, 44)
            features.append(out[1].transpose(0, 3, 1, 2))  # (1, 64, 16, 44)

        depth_cat = np.concatenate(depths)     # (6, 44, 16, 44)
        feat_cat = np.concatenate(features)    # (6, 64, 16, 44)
        pooled = bev_pool(depth_cat, feat_cat, ranks_depth, ranks_feat, ranks_bev)
        # pooled: (1, 64, 200, 200) NCHW → NHWC for BEV model
        bev_in = pooled.transpose(0, 2, 3, 1)

        bev_out = bev_model.execute(inputs={'bev_input': bev_in}, use_jax=True)
        logits = bev_out[0]  # (1, 200, 200, 288) NHWC = (N, H, W, C)
        # Restore occupancy layout: NCHW original was (N,288,H,W); restore_logits did
        # permute(0,3,2,1) → (N,W,H,C).  AFE gives NHWC=(N,H,W,C), so swap H,W first.
        labels = (
            logits.transpose(0, 2, 1, 3)
            .reshape(1, 200, 200, 16, 18)
            .argmax(-1)
            .astype(np.int64)
        )
        labels = labels[0]  # (200, 200, 16) = (W, H, depth)

        sample_masked_confusion = confusion_matrix(
            semantics[mask_camera], labels[mask_camera]
        )
        sample_all_confusion = confusion_matrix(semantics, labels)
        masked_confusion += sample_masked_confusion
        all_confusion += sample_all_confusion
        masked_metrics = occupancy_metrics(sample_masked_confusion)
        all_acc = float(np.mean(labels == semantics))
        # occupied = any voxel not class 17 (free)
        occupied = (semantics != 17) | (labels != 17)
        occ_acc = float(np.mean((labels == semantics)[occupied])) if occupied.any() else float('nan')
        bin_acc = float(np.mean((labels == 17) == (semantics == 17)))

        rec = dict(token=r['token'], scene=r['scene'],
                   camera_masked_accuracy=masked_metrics['accuracy'],
                   camera_masked_miou=masked_metrics['miou_excluding_free'],
                   all_voxel_label_accuracy=all_acc,
                   occupied_union_label_accuracy=occ_acc,
                   occupancy_binary_accuracy=bin_acc)
        results.append(rec)
        print(rec, flush=True)

    summary = dict(
        protocol=('Occ3D camera mask; aggregate confusion matrix; mIoU over '
                  'classes 0-16 with free class 17 excluded'),
        camera_masked=occupancy_metrics(masked_confusion),
        all_voxels_diagnostic=occupancy_metrics(all_confusion),
        all_voxel_label_accuracy_unweighted_sample_mean=float(
            np.mean([r['all_voxel_label_accuracy'] for r in results])
        ),
        occupied_union_label_accuracy=float(np.nanmean([r['occupied_union_label_accuracy'] for r in results])),
        occupancy_binary_accuracy=float(np.mean([r['occupancy_binary_accuracy'] for r in results])),
        samples=len(results),
    )
    return dict(summary=summary, per_sample=results)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true',
                        help='Skip compilation and load saved models for validation')
    args = parser.parse_args()

    if args.validate_only:
        camera_quant = Model.load('flashocc_m0_camera', str(OUT_DIR / 'camera'))
        bev_quant = Model.load('flashocc_m0_bev', str(OUT_DIR / 'bev'))
    else:
        camera_quant = compile_stage('camera')
        bev_quant = compile_stage('bev')

    # validate
    print('\n=== AFE quantized validation ===', flush=True)
    results = validate(camera_quant, bev_quant)
    out_path = OUT_DIR / 'accuracy.json'
    out_path.write_text(json.dumps(results, indent=2) + '\n')
    print(f'\nSummary:', flush=True)
    for k, v in results['summary'].items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}', flush=True)
        elif isinstance(v, dict):
            print(f'  {k}: {json.dumps(v, sort_keys=True)}', flush=True)
    print(f'  samples: {results["summary"]["samples"]}', flush=True)
    print(f'\nResults written to {out_path}', flush=True)


if __name__ == '__main__':
    main()
