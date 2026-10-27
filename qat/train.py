#!/usr/bin/env python3
"""Jointly fine-tune FlashOCC camera and BEV stages with SiMa QAT."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from itertools import chain
from pathlib import Path, PosixPath

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn
from torch.ao.quantization import disable_observer
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_flashocc import M0, Stage  # noqa: E402
from flashocc_qat import (  # noqa: E402
    FlashOccTensorDataset,
    OCCUPANCY_CLASSES,
    collate_samples,
    pool_batch,
    restore_logits,
)
from sima_qat import (  # noqa: E402
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)
from validate_qat_onnx import validate_onnx_pair  # noqa: E402


CHECKPOINT_FORMAT = "sima-flashocc-m0-qat-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "artifacts/flashocc-r50-M0.pth")
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/qat-inputs")
    parser.add_argument("--output", type=Path, default=ROOT / "qat/runs/m0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--freeze-epoch", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--observer-warmup-batches", type=int, default=32)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def load_eager_model(weights: Path) -> M0:
    model = M0()
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def images_from_batch(samples, device: torch.device) -> Tensor:
    return torch.cat([sample["images"] for sample in samples]).to(
        device=device, non_blocking=True
    )


def occupancy_targets(samples, device: torch.device) -> tuple[Tensor, Tensor]:
    semantics = torch.stack([sample["semantics"] for sample in samples]).to(
        device=device, non_blocking=True
    )
    mask_camera = torch.stack([sample["mask_camera"] for sample in samples]).to(
        device=device, non_blocking=True
    )
    return semantics, mask_camera


def confusion_matrix(labels: Tensor, predictions: Tensor) -> Tensor:
    bins = OCCUPANCY_CLASSES * labels.to(torch.int64) + predictions.to(torch.int64)
    return torch.bincount(
        bins, minlength=OCCUPANCY_CLASSES**2
    ).reshape(OCCUPANCY_CLASSES, OCCUPANCY_CLASSES)


def occupancy_metrics(confusion: Tensor) -> tuple[float, float]:
    diagonal = confusion.diag().to(torch.float64)
    union = confusion.sum(0) + confusion.sum(1) - diagonal
    present_semantic_classes = union[: OCCUPANCY_CLASSES - 1] > 0
    iou = diagonal[: OCCUPANCY_CLASSES - 1] / union[: OCCUPANCY_CLASSES - 1].clamp_min(1)
    mean_iou = iou[present_semantic_classes].mean().item()
    accuracy = diagonal.sum().item() / confusion.sum().clamp_min(1).item()
    return accuracy, mean_iou


@contextmanager
def validation_mode(*models: nn.Module):
    modes = [model.training for model in models]
    observer_states = [
        (module, module.observer_enabled.clone())
        for model in models
        for module in model.modules()
        if isinstance(module, FakeQuantizeBase)
    ]
    try:
        for model in models:
            model.eval()
            model.apply(disable_observer)
        yield
    finally:
        for model, training in zip(models, modes):
            model.train(training)
        for module, enabled in observer_states:
            module.observer_enabled.copy_(enabled)


def forward_pair(camera, bev, samples, device):
    images = images_from_batch(samples, device)
    depth, features = camera(images)
    pooled = pool_batch(depth, features, samples)
    logits = restore_logits(bev(pooled))
    return depth, features, logits


def warm_observers(qat_camera, qat_bev, loader, device, maximum_batches):
    qat_camera.train(True)
    qat_bev.train(True)
    with torch.no_grad():
        for step, samples in enumerate(loader, start=1):
            forward_pair(qat_camera, qat_bev, samples, device)
            if step >= maximum_batches:
                break
    print(f"observer_warmup_batches={min(step, maximum_batches)}")


def evaluate(qat_camera, qat_bev, teacher_camera, teacher_bev, loader, device):
    totals = {
        "depth_mse": 0.0,
        "feature_relative_rmse": 0.0,
        "logit_relative_rmse": 0.0,
    }
    qat_confusion = torch.zeros(OCCUPANCY_CLASSES, OCCUPANCY_CLASSES, dtype=torch.int64)
    teacher_confusion = torch.zeros_like(qat_confusion)
    cross_entropy_sum = 0.0
    valid_voxels = 0
    label_agreement = 0
    occupancy_binary_agreement = 0
    batches = 0
    with validation_mode(qat_camera, qat_bev), torch.inference_mode():
        for samples in loader:
            qat_depth, qat_features, qat_logits = forward_pair(
                qat_camera, qat_bev, samples, device
            )
            ref_depth, ref_features, ref_logits = forward_pair(
                teacher_camera, teacher_bev, samples, device
            )
            qat_labels = qat_logits.argmax(-1)
            ref_labels = ref_logits.argmax(-1)
            semantics, mask_camera = occupancy_targets(samples, device)
            valid = mask_camera.bool()
            if not valid.any():
                raise RuntimeError("A validation batch has no camera-visible voxels")
            masked_semantics = semantics[valid]
            masked_qat_labels = qat_labels[valid]
            masked_ref_labels = ref_labels[valid]
            cross_entropy_sum += F.cross_entropy(
                qat_logits[valid], masked_semantics, reduction="sum"
            ).item()
            valid_voxels += masked_semantics.numel()
            qat_confusion += confusion_matrix(
                masked_semantics, masked_qat_labels
            ).cpu()
            teacher_confusion += confusion_matrix(
                masked_semantics, masked_ref_labels
            ).cpu()
            label_agreement += (masked_qat_labels == masked_ref_labels).sum().item()
            occupancy_binary_agreement += (
                (masked_qat_labels == 17) == (masked_ref_labels == 17)
            ).sum().item()
            totals["depth_mse"] += F.mse_loss(qat_depth, ref_depth).item()
            totals["feature_relative_rmse"] += (
                (qat_features-ref_features).square().mean().sqrt()
                / ref_features.square().mean().sqrt().clamp_min(1e-6)
            ).item()
            totals["logit_relative_rmse"] += (
                (qat_logits-ref_logits).square().mean().sqrt()
                / ref_logits.square().mean().sqrt().clamp_min(1e-6)
            ).item()
            batches += 1
    if not batches or not valid_voxels:
        raise RuntimeError("Validation produced no camera-visible occupancy labels")
    qat_accuracy, qat_miou = occupancy_metrics(qat_confusion)
    teacher_accuracy, teacher_miou = occupancy_metrics(teacher_confusion)
    return {
        "ground_truth_cross_entropy": cross_entropy_sum / valid_voxels,
        "ground_truth_accuracy": qat_accuracy,
        "ground_truth_miou": qat_miou,
        "teacher_ground_truth_accuracy": teacher_accuracy,
        "teacher_ground_truth_miou": teacher_miou,
        "label_agreement": label_agreement / valid_voxels,
        "occupancy_binary_agreement": occupancy_binary_agreement / valid_voxels,
        **{name: value / batches for name, value in totals.items()},
    }


def save_checkpoint(path, camera, bev, optimizer, epoch, frozen, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized_args = json.loads(json.dumps(vars(args), default=str))
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "camera": camera.state_dict(),
            "bev": bev.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "frozen": frozen,
            "args": serialized_args,
        },
        path,
    )


def inspect_onnx(path: Path, input_name: str, input_shape: tuple[int, ...]) -> dict:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    opset = next(item.version for item in model.opset_import if not item.domain)
    if opset != 17:
        raise RuntimeError(f"Expected ONNX opset 17 in {path}, found {opset}")
    counts = {"QuantizeLinear": 0, "DequantizeLinear": 0}
    for node in model.graph.node:
        if node.op_type in counts:
            counts[node.op_type] += 1
    if not all(counts.values()):
        raise RuntimeError(f"QDQ nodes missing from {path}: {counts}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    outputs = session.run(None, {input_name: np.zeros(input_shape, dtype=np.float32)})
    if not all(np.isfinite(output).all() for output in outputs):
        raise RuntimeError(f"ONNX Runtime produced non-finite values for {path}")
    return {
        "opset": opset,
        **counts,
        "output_shapes": [list(output.shape) for output in outputs],
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.observer_warmup_batches < 1:
        raise ValueError("epochs, batch-size, and observer-warmup-batches must be positive")
    if args.freeze_epoch < -1 or args.freeze_epoch >= args.epochs:
        raise ValueError("freeze-epoch must be -1 or an epoch within the run")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but is unavailable")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    train_data = FlashOccTensorDataset(args.data, "calibration")
    validation_data = FlashOccTensorDataset(args.data, "validation")
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=collate_samples,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=collate_samples,
    )

    eager = load_eager_model(args.weights)
    qat_camera = sima_prepare_qat_model(
        Stage(eager, "camera"), (torch.zeros(1, 3, 256, 704),), device
    )
    qat_bev = sima_prepare_qat_model(
        Stage(eager, "bev", conv_head=True), (torch.zeros(1, 64, 200, 200),), device
    )
    teacher_camera = Stage(eager, "camera").to(device).eval()
    teacher_bev = Stage(eager, "bev", conv_head=True).to(device).eval()
    for parameter in eager.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        chain(qat_camera.parameters(), qat_bev.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    frozen = False
    if args.resume:
        with torch.serialization.safe_globals([PosixPath]):
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError(f"Unsupported QAT checkpoint: {args.resume}")
        qat_camera.load_state_dict(checkpoint["camera"], strict=True)
        qat_bev.load_state_dict(checkpoint["bev"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        frozen = bool(checkpoint["frozen"])
    else:
        warm_observers(
            qat_camera,
            qat_bev,
            train_loader,
            device,
            args.observer_warmup_batches,
        )

    baseline = evaluate(
        qat_camera, qat_bev, teacher_camera, teacher_bev, validation_loader, device
    )
    print("prepared_validation=" + json.dumps(baseline, sort_keys=True))
    metrics_path = args.output / "metrics.json"
    if args.resume and metrics_path.is_file():
        history = json.loads(metrics_path.read_text(encoding="utf-8"))
        history.append({"stage": "resumed", "checkpoint": str(args.resume), **baseline})
    else:
        history = [{"stage": "prepared", **baseline}]
    metrics_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    for epoch in range(start_epoch, args.epochs):
        qat_camera.train(True)
        qat_bev.train(True)
        if args.freeze_epoch == epoch and not frozen:
            sima_freeze_qat(qat_camera)
            sima_freeze_qat(qat_bev)
            frozen = True
            print(f"Locked camera and BEV QAT grids at epoch {epoch}")
        started = time.monotonic()
        total_loss = 0.0
        for step, samples in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            _, _, qat_logits = forward_pair(
                qat_camera, qat_bev, samples, device
            )
            semantics, mask_camera = occupancy_targets(samples, device)
            valid = mask_camera.bool()
            if not valid.any():
                raise RuntimeError("A training batch has no camera-visible voxels")
            loss = F.cross_entropy(qat_logits[valid], semantics[valid])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                chain(qat_camera.parameters(), qat_bev.parameters()), args.gradient_clip
            )
            optimizer.step()
            total_loss += loss.detach().item()
            if step % 10 == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={loss.item():.6f} valid_voxels={valid.sum().item()}",
                    flush=True,
                )
        metrics = evaluate(
            qat_camera, qat_bev, teacher_camera, teacher_bev, validation_loader, device
        )
        epoch_record = {
            "stage": "epoch",
            "epoch": epoch,
            "frozen": frozen,
            "training_loss": total_loss / len(train_loader),
            "seconds": time.monotonic() - started,
            **metrics,
        }
        history.append(epoch_record)
        print("epoch_validation=" + json.dumps(epoch_record, sort_keys=True))
        save_checkpoint(
            args.output / "checkpoints" / f"epoch_{epoch:03d}.pt",
            qat_camera,
            qat_bev,
            optimizer,
            epoch,
            frozen,
            args,
        )
        metrics_path.write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )

    if args.no_export:
        return
    final_camera = sima_finalize_qat_model(qat_camera)
    final_bev = sima_finalize_qat_model(qat_bev)
    final_metrics = evaluate(
        final_camera, final_bev, teacher_camera, teacher_bev, validation_loader, device
    )
    history.append({"stage": "finalized", **final_metrics})
    export_directory = args.output / "onnx"
    export_directory.mkdir(parents=True, exist_ok=True)
    camera_path = export_directory / "camera.qdq.onnx"
    bev_path = export_directory / "bev.qdq.onnx"
    sima_export_onnx(
        final_camera,
        (torch.zeros(1, 3, 256, 704),),
        str(camera_path),
        input_names=["camera_input"],
        output_names=["depth", "features"],
    )
    sima_export_onnx(
        final_bev,
        (torch.zeros(1, 64, 200, 200),),
        str(bev_path),
        input_names=["bev_input"],
        output_names=["logits"],
    )
    report = {
        "method": "Occ3D camera-masked cross-entropy; FP32 fidelity is secondary",
        "camera": inspect_onnx(camera_path, "camera_input", (1, 3, 256, 704)),
        "bev": inspect_onnx(bev_path, "bev_input", (1, 64, 200, 200)),
        "finalized_validation": final_metrics,
        "onnx_runtime_validation": validate_onnx_pair(
            camera_path,
            bev_path,
            args.data,
            threads=4,
        ),
    }
    (args.output / "export_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    metrics_path.write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    print("export_validation=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

