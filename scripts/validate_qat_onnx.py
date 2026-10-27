#!/usr/bin/env python3
"""Validate the exported FlashOCC QAT ONNX pair with Occ3D metrics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from pool_reference import pool as bev_pool


ROOT = Path(__file__).resolve().parents[1]
OCCUPANCY_CLASSES = 18
FREE_CLASS = 17


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape:
        raise ValueError(
            f"Label and prediction shapes differ: {labels.shape} != {predictions.shape}"
        )
    if labels.size and (
        labels.min() < 0
        or labels.max() >= OCCUPANCY_CLASSES
        or predictions.min() < 0
        or predictions.max() >= OCCUPANCY_CLASSES
    ):
        raise ValueError("Occupancy classes must be in [0, 17]")
    bins = (labels * OCCUPANCY_CLASSES + predictions).reshape(-1)
    return np.bincount(
        bins, minlength=OCCUPANCY_CLASSES**2
    ).reshape(OCCUPANCY_CLASSES, OCCUPANCY_CLASSES)


def occupancy_metrics(confusion: np.ndarray) -> dict:
    diagonal = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - diagonal
    semantic_union = union[:FREE_CLASS]
    present = semantic_union > 0
    mean_iou = float(np.mean(diagonal[:FREE_CLASS][present] / semantic_union[present]))
    return {
        "accuracy": float(diagonal.sum() / max(confusion.sum(), 1)),
        "miou_excluding_free": mean_iou,
        "voxels": int(confusion.sum()),
        "per_class_iou": [
            float(diagonal[index] / union[index]) if union[index] else None
            for index in range(OCCUPANCY_CLASSES)
        ],
    }


def create_session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )


def validate_onnx_pair(
    camera_path: Path,
    bev_path: Path,
    data_root: Path,
    *,
    count: int | None = None,
    threads: int = 8,
    progress: bool = True,
) -> dict:
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    records = [record for record in manifest["samples"] if record["split"] == "validation"]
    if count is not None:
        records = records[:count]
    if not records:
        raise ValueError("No validation samples were selected")

    camera = create_session(camera_path, threads)
    bev = create_session(bev_path, threads)
    camera_input = camera.get_inputs()[0].name
    bev_input = bev.get_inputs()[0].name
    masked_confusion = np.zeros((OCCUPANCY_CLASSES, OCCUPANCY_CLASSES), dtype=np.int64)
    all_confusion = np.zeros_like(masked_confusion)
    per_sample = []
    started = time.monotonic()

    for index, record in enumerate(records, start=1):
        with np.load(data_root / record["path"]) as values:
            images = values["images"].astype(np.float32, copy=False)
            ranks_depth = values["ranks_depth"]
            ranks_feat = values["ranks_feat"]
            ranks_bev = values["ranks_bev"]
            semantics = values["semantics"].astype(np.int64)
            mask_camera = values["mask_camera"].astype(bool)

        depths = []
        features = []
        for image in images:
            depth, feature = camera.run(None, {camera_input: image[None]})
            depths.append(depth)
            features.append(feature)
        pooled = bev_pool(
            np.concatenate(depths),
            np.concatenate(features),
            ranks_depth,
            ranks_feat,
            ranks_bev,
        )
        logits = bev.run(None, {bev_input: pooled})[0]
        if logits.shape != (1, 288, 200, 200):
            raise ValueError(f"Unexpected BEV output shape: {logits.shape}")
        predictions = (
            logits.transpose(0, 3, 2, 1)
            .reshape(200, 200, 16, OCCUPANCY_CLASSES)
            .argmax(-1)
            .astype(np.int64)
        )

        sample_masked = confusion_matrix(
            semantics[mask_camera], predictions[mask_camera]
        )
        sample_all = confusion_matrix(semantics, predictions)
        masked_confusion += sample_masked
        all_confusion += sample_all
        per_sample.append(
            {
                "token": record["token"],
                "scene": record["scene"],
                "camera_masked": occupancy_metrics(sample_masked),
                "all_voxels_diagnostic": occupancy_metrics(sample_all),
            }
        )
        if progress and (index == 1 or index % 4 == 0 or index == len(records)):
            metrics = per_sample[-1]["camera_masked"]
            print(
                f"validated={index}/{len(records)} "
                f"masked_accuracy={metrics['accuracy']:.6f} "
                f"masked_miou={metrics['miou_excluding_free']:.6f}",
                flush=True,
            )

    return {
        "protocol": (
            "Occ3D camera mask; aggregate confusion matrix; mIoU over classes "
            "0-16 with free class 17 excluded"
        ),
        "camera_onnx": str(camera_path),
        "bev_onnx": str(bev_path),
        "samples": len(records),
        "seconds": time.monotonic() - started,
        "camera_masked": occupancy_metrics(masked_confusion),
        "all_voxels_diagnostic_do_not_use_for_model_accuracy": occupancy_metrics(
            all_confusion
        ),
        "per_sample": per_sample,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        type=Path,
        default=ROOT / "artifacts/qat-runs/m0-full/onnx/camera.qdq.onnx",
    )
    parser.add_argument(
        "--bev",
        type=Path,
        default=ROOT / "artifacts/qat-runs/m0-full/onnx/bev.qdq.onnx",
    )
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/qat-inputs")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/qat-runs/m0-full/onnx/validation_metrics.json",
    )
    parser.add_argument("--count", type=int)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count is not None and args.count < 1:
        raise ValueError("count must be positive")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    report = validate_onnx_pair(
        args.camera,
        args.bev,
        args.data,
        count=args.count,
        threads=args.threads,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = report["camera_masked"]
    print(
        f"camera_masked_accuracy={summary['accuracy']:.6f} "
        f"camera_masked_miou={summary['miou_excluding_free']:.6f}"
    )
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
