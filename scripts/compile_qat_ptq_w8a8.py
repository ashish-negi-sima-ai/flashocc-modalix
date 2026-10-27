"""PTQ compilation of QAT-trained weights in w8a8.

Pipeline:
  1. Strip QDQ nodes from camera.qdq.onnx / bev.qdq.onnx → clean FP32 ONNX
     preserving the QAT-optimised float32 weight values.
  2. Compile each stage with AFE PTQ using real calibration data from
     artifacts/qat-inputs/calibration/ (512 samples; we use up to --calib-count).
  3. Validate on the 32 validation samples and report accuracy vs ground truth.

Outputs go to artifacts/qat-runs/m0-full/ptq-compile/{camera,bev}/.
"""
import argparse
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
QAT_DIR = ROOT / 'artifacts/qat-runs/m0-full'
ONNX_DIR = QAT_DIR / 'onnx'
STRIPPED_DIR = QAT_DIR / 'onnx-stripped'
OUT_DIR = QAT_DIR / 'ptq-compile'
INPUTS_DIR = ROOT / 'artifacts/qat-inputs'

OCCUPANCY_CLASSES = 18
FREE_CLASS = 17


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
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


_STAGE_INPUT = {
    'camera': ('camera_input', (1, 3, 256, 704)),   # NCHW
    'bev':    ('bev_input',    (1, 64, 200, 200)),   # NCHW
}


def _strip_qdq(stage: str) -> Path:
    """Strip QDQ nodes and return path to the cleaned FP32 ONNX."""
    import subprocess, sys
    STRIPPED_DIR.mkdir(parents=True, exist_ok=True)
    src = ONNX_DIR / f'{stage}.qdq.onnx'
    dst = STRIPPED_DIR / f'{stage}.fp32.onnx'
    if dst.exists():
        print(f'[{stage}] stripped ONNX already exists, reusing', flush=True)
        return dst
    print(f'[{stage}] stripping QDQ nodes...', flush=True)
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/strip_qdq.py'), str(src), str(dst)],
        capture_output=False,
    )
    result.check_returncode()
    return dst


def _load_camera_calibration(count: int) -> list[dict]:
    """One image per calibration sample (first camera)."""
    manifest = json.loads((INPUTS_DIR / 'manifest.json').read_text())
    records = [r for r in manifest['samples'] if r['split'] == 'calibration'][:count]
    calib = []
    for r in records:
        with np.load(INPUTS_DIR / r['path']) as data:
            img = data['images'][0:1]   # (1,3,256,704) NCHW
        calib.append({'camera_input': img.transpose(0, 2, 3, 1)})  # NHWC
    print(f'[camera] using {len(calib)} real calibration samples', flush=True)
    return calib


def _load_bev_calibration(fp32_camera_net, count: int) -> list[dict]:
    """Generate real BEV pooled tensors by running the FP32 camera net forward."""
    from pool_reference import pool as bev_pool

    manifest = json.loads((INPUTS_DIR / 'manifest.json').read_text())
    records = [r for r in manifest['samples'] if r['split'] == 'calibration'][:count]
    calib = []
    for r in records:
        with np.load(INPUTS_DIR / r['path']) as data:
            images      = data['images']
            ranks_depth = data['ranks_depth']
            ranks_feat  = data['ranks_feat']
            ranks_bev   = data['ranks_bev']
        depths, features = [], []
        for i in range(6):
            out = fp32_camera_net.execute(
                inputs={'camera_input': images[i:i+1].transpose(0, 2, 3, 1)}
            )
            depths.append(out[0].transpose(0, 3, 1, 2))
            features.append(out[1].transpose(0, 3, 1, 2))
        pooled = bev_pool(np.concatenate(depths), np.concatenate(features),
                          ranks_depth, ranks_feat, ranks_bev)
        calib.append({'bev_input': pooled.transpose(0, 2, 3, 1)})  # NHWC
    print(f'[bev] using {len(calib)} real BEV calibration samples', flush=True)
    return calib


def compile_stage(stage: str, calib_count: int,
                  fp32_camera_net=None) -> tuple:
    """Returns (quantized_Model, fp32_LoadedNet)."""
    onnx_path = _strip_qdq(stage)
    out = OUT_DIR / stage
    out.mkdir(parents=True, exist_ok=True)

    print(f'[{stage}] loading {onnx_path}', flush=True)
    net = load_model(
        ImporterParams(format=ModelFormat.onnx, file_paths=[str(onnx_path)]),
        target=gen2_target,
    )
    if stage == 'camera':
        calib = _load_camera_calibration(calib_count)
    else:
        calib = _load_bev_calibration(fp32_camera_net, calib_count)
    print(f'[{stage}] quantizing (w8a8 PTQ, {len(calib)} calibration samples)', flush=True)
    quant = net.quantize(
        calibration_data=calib,
        quantization_config=default_quantization,
        model_name=f'flashocc_m0_{stage}',
        log_level=logging.INFO,
    )
    quant.save(model_name=f'flashocc_m0_{stage}', output_directory=str(out))
    print(f'[{stage}] compiling', flush=True)
    quant.compile(output_path=str(out), batch_size=1, log_level=logging.INFO)
    (out / 'compile_status.json').write_text(json.dumps(
        {'stage': stage, 'quantization': 'w8a8',
         'calibration': f'PTQ on QAT weights, {len(calib)} samples',
         'compiled': True}, indent=2))
    print(f'[{stage}] done -> {out}', flush=True)
    return quant, net  # (quantized Model, FP32 LoadedNet)


def validate(camera_model: Model, bev_model: Model) -> dict:
    from pool_reference import pool as bev_pool

    manifest = json.loads((INPUTS_DIR / 'manifest.json').read_text())
    records = [r for r in manifest['samples'] if r['split'] == 'validation']

    results = []
    masked_confusion = np.zeros((OCCUPANCY_CLASSES, OCCUPANCY_CLASSES), dtype=np.int64)
    all_confusion = np.zeros_like(masked_confusion)

    for r in records:
        with np.load(INPUTS_DIR / r['path']) as data:
            images      = data['images']
            ranks_depth = data['ranks_depth']
            ranks_feat  = data['ranks_feat']
            ranks_bev   = data['ranks_bev']
            semantics   = data['semantics'].astype(np.int64)
            mask_camera = data['mask_camera'].astype(bool)

        depths, features = [], []
        for i in range(6):
            img_nhwc = images[i:i+1].transpose(0, 2, 3, 1)
            out = camera_model.execute(inputs={'camera_input': img_nhwc}, use_jax=True)
            depths.append(out[0].transpose(0, 3, 1, 2))    # NHWC→NCHW
            features.append(out[1].transpose(0, 3, 1, 2))

        pooled = bev_pool(
            np.concatenate(depths), np.concatenate(features),
            ranks_depth, ranks_feat, ranks_bev,
        )
        bev_out = bev_model.execute(
            inputs={'bev_input': pooled.transpose(0, 2, 3, 1)}, use_jax=True
        )
        logits = bev_out[0]  # (1, 200, 200, 288) NHWC = (N,H,W,C)
        # AFE NHWC=(N,H,W,C); swap H↔W to get (N,W,H,C) matching semantics (W,H,depth)
        labels = (logits.transpose(0, 2, 1, 3)
                        .reshape(1, 200, 200, 16, 18)
                        .argmax(-1)
                        .astype(np.int64))[0]   # (200,200,16) = (W,H,depth)

        sample_masked = confusion_matrix(semantics[mask_camera], labels[mask_camera])
        sample_all    = confusion_matrix(semantics, labels)
        masked_confusion += sample_masked
        all_confusion    += sample_all

        masked_metrics = occupancy_metrics(sample_masked)
        rec = dict(token=r['token'], scene=r['scene'],
                   camera_masked_accuracy=masked_metrics['accuracy'],
                   camera_masked_miou=masked_metrics['miou_excluding_free'])
        results.append(rec)
        print(rec, flush=True)

    summary = dict(
        protocol='Occ3D camera mask; aggregate confusion matrix; mIoU classes 0-16',
        camera_masked=occupancy_metrics(masked_confusion),
        all_voxels_diagnostic=occupancy_metrics(all_confusion),
        samples=len(results),
    )
    return dict(summary=summary, per_sample=results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib-count', type=int, default=64,
                        help='Number of calibration samples per stage (default 64)')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()

    if args.validate_only:
        camera = Model.load('flashocc_m0_camera', str(OUT_DIR / 'camera'))
        bev    = Model.load('flashocc_m0_bev',    str(OUT_DIR / 'bev'))
    else:
        camera, fp32_camera = compile_stage('camera', args.calib_count)
        bev, _              = compile_stage('bev', args.calib_count,
                                            fp32_camera_net=fp32_camera)

    print('\n=== AFE PTQ validation (QAT weights) ===', flush=True)
    results = validate(camera, bev)
    out_path = OUT_DIR / 'accuracy.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + '\n')
    print('\nSummary:', flush=True)
    for k, v in results['summary'].items():
        if isinstance(v, dict):
            print(f'  {k}: {json.dumps(v, sort_keys=True)}', flush=True)
        elif k != 'samples':
            print(f'  {k}: {v}', flush=True)
    print(f'  samples: {results["summary"]["samples"]}', flush=True)
    print(f'\nResults -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
