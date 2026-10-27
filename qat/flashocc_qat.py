"""Shared FlashOCC M0 QAT data and tensor utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


CAMERAS_PER_SAMPLE = 6
DEPTH_BINS = 44
FEATURE_CHANNELS = 64
FEATURE_HEIGHT = 16
FEATURE_WIDTH = 44
BEV_HEIGHT = 200
BEV_WIDTH = 200
OCCUPANCY_HEIGHT = 16
OCCUPANCY_CLASSES = 18
OCCUPANCY_SHAPE = (BEV_WIDTH, BEV_HEIGHT, OCCUPANCY_HEIGHT)


class FlashOccTensorDataset(Dataset):
    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.camera_order = manifest["camera_order"]
        if len(self.camera_order) != CAMERAS_PER_SAMPLE:
            raise ValueError(f"Expected six cameras, found {self.camera_order}")
        self.records = [record for record in manifest["samples"] if record["split"] == split]
        if not self.records:
            raise ValueError(f"No {split!r} samples in {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        path = self.root / record["path"]
        with np.load(path) as values:
            images = torch.from_numpy(values["images"].copy())
            ranks_depth = torch.from_numpy(values["ranks_depth"].astype(np.int64, copy=True))
            ranks_feat = torch.from_numpy(values["ranks_feat"].astype(np.int64, copy=True))
            ranks_bev = torch.from_numpy(values["ranks_bev"].astype(np.int64, copy=True))
            semantics = torch.from_numpy(values["semantics"].astype(np.int64, copy=True))
            mask_camera = torch.from_numpy(values["mask_camera"].astype(bool, copy=True))
        if images.shape != (CAMERAS_PER_SAMPLE, 3, 256, 704):
            raise ValueError(f"Unexpected image tensor in {path}: {tuple(images.shape)}")
        if not len(ranks_depth) == len(ranks_feat) == len(ranks_bev):
            raise ValueError(f"Pooling rank lengths differ in {path}")
        if semantics.shape != OCCUPANCY_SHAPE or mask_camera.shape != OCCUPANCY_SHAPE:
            raise ValueError(f"Unexpected occupancy labels in {path}")
        if semantics.min() < 0 or semantics.max() >= OCCUPANCY_CLASSES:
            raise ValueError(f"Occupancy class outside [0, 17] in {path}")
        return {
            "token": record["token"],
            "images": images,
            "ranks_depth": ranks_depth,
            "ranks_feat": ranks_feat,
            "ranks_bev": ranks_bev,
            "semantics": semantics,
            "mask_camera": mask_camera,
        }


def collate_samples(samples: list[dict[str, Tensor | str]]) -> list[dict[str, Tensor | str]]:
    return samples


def differentiable_bev_pool(
    depth: Tensor,
    features: Tensor,
    ranks_depth: Tensor,
    ranks_feat: Tensor,
    ranks_bev: Tensor,
) -> Tensor:
    """PyTorch equivalent of BEVPoolv2 for one six-camera sample."""
    if depth.shape != (CAMERAS_PER_SAMPLE, DEPTH_BINS, FEATURE_HEIGHT, FEATURE_WIDTH):
        raise ValueError(f"Unexpected depth shape: {tuple(depth.shape)}")
    if features.shape != (
        CAMERAS_PER_SAMPLE,
        FEATURE_CHANNELS,
        FEATURE_HEIGHT,
        FEATURE_WIDTH,
    ):
        raise ValueError(f"Unexpected feature shape: {tuple(features.shape)}")
    device = depth.device
    ranks_depth = ranks_depth.to(device=device, non_blocking=True)
    ranks_feat = ranks_feat.to(device=device, non_blocking=True)
    ranks_bev = ranks_bev.to(device=device, non_blocking=True)
    flat_depth = depth.reshape(-1)
    flat_features = features.permute(0, 2, 3, 1).reshape(-1, FEATURE_CHANNELS)
    if ranks_depth.numel():
        if ranks_depth.min() < 0 or ranks_depth.max() >= flat_depth.shape[0]:
            raise ValueError("ranks_depth is outside the depth tensor")
        if ranks_feat.min() < 0 or ranks_feat.max() >= flat_features.shape[0]:
            raise ValueError("ranks_feat is outside the feature tensor")
        if ranks_bev.min() < 0 or ranks_bev.max() >= BEV_HEIGHT * BEV_WIDTH:
            raise ValueError("ranks_bev is outside the BEV tensor")
    weighted_features = flat_depth[ranks_depth, None] * flat_features[ranks_feat]
    output = features.new_zeros((BEV_HEIGHT * BEV_WIDTH, FEATURE_CHANNELS))
    output.index_add_(0, ranks_bev, weighted_features)
    return output.view(BEV_HEIGHT, BEV_WIDTH, FEATURE_CHANNELS).permute(2, 0, 1)


def pool_batch(
    depth: Tensor,
    features: Tensor,
    samples: list[dict[str, Tensor | str]],
) -> Tensor:
    if depth.shape[0] != len(samples) * CAMERAS_PER_SAMPLE:
        raise ValueError("Camera output batch does not match the sample count")
    pooled = []
    for index, sample in enumerate(samples):
        camera_slice = slice(index * CAMERAS_PER_SAMPLE, (index + 1) * CAMERAS_PER_SAMPLE)
        pooled.append(
            differentiable_bev_pool(
                depth[camera_slice],
                features[camera_slice],
                sample["ranks_depth"],
                sample["ranks_feat"],
                sample["ranks_bev"],
            )
        )
    return torch.stack(pooled)


def restore_logits(logits: Tensor) -> Tensor:
    expected_channels = OCCUPANCY_HEIGHT * OCCUPANCY_CLASSES
    if logits.ndim != 4 or logits.shape[1:] != (expected_channels, BEV_HEIGHT, BEV_WIDTH):
        raise ValueError(f"Unexpected BEV logits shape: {tuple(logits.shape)}")
    return logits.permute(0, 3, 2, 1).reshape(
        logits.shape[0], BEV_WIDTH, BEV_HEIGHT, OCCUPANCY_HEIGHT, OCCUPANCY_CLASSES
    )

