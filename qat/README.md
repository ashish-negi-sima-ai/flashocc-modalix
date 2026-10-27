# FlashOCC M0 SiMa QAT

This workflow applies SiMa QAT to both deployable FlashOCC M0 graphs:

- camera: `[1,3,256,704]` to depth and image features;
- BEV: `[B,64,200,200]` to `[B,288,200,200]` occupancy logits.

Differentiable PyTorch BEV pooling remains outside the two QAT graphs, matching
the deployment split. Recovery training uses FlashOCC M0's original
camera-masked occupancy cross-entropy against Occ3D labels. Validation reports
ground-truth accuracy and semantic mIoU, plus FP32 fidelity as a secondary
quantization-regression metric.

## Prepare

Extract and verify the checkpoint from the supplied bundle:

```bash
mkdir -p artifacts
unzip -j flashocc-onnx-export.zip \
  '*/artifacts/flashocc-r50-M0.pth' -d artifacts
echo 'cb776378659af93c2e6968d89767e4c03b6d3720d11fc2071e57b8d5af1d1cd5  artifacts/flashocc-r50-M0.pth' \
  | sha256sum --check
```

Download only the Occ3D trainval labels. The Hugging Face object below mirrors
the official `gts.tar.gz` byte size and avoids Google Drive quota failures:

```bash
mkdir -p artifacts/occ3d
curl -fL -C - \
  https://huggingface.co/datasets/RoyYao233/occ3d-nuscenes-transfer/resolve/main/gts.tar.gz \
  -o artifacts/occ3d/gts.tar.gz
test "$(stat -c %s artifacts/occ3d/gts.tar.gz)" = 2737816708
gzip -t artifacts/occ3d/gts.tar.gz
tar -xzf artifacts/occ3d/gts.tar.gz -C artifacts/occ3d
```

Create the environment and prepared QAT tensors:

```bash
qat/setup_env.sh
qat/.venv/bin/python scripts/prepare_nuscenes.py \
  --data /project/ml_datasets/public_datasets/nuscenes \
  --labels artifacts/occ3d/gts \
  --output artifacts/qat-inputs \
  --train-count 512 \
  --val-count 32
```

The trainval preprocessor uses the official nuScenes train and validation scene
split. Large nuScenes metadata tables are streamed, so only records needed by
the selected samples are retained in memory. Occ3D labels are expected as
`gts/scene-*/<sample-token>/labels.npz`; each file must contain `semantics`,
`mask_lidar`, and `mask_camera` arrays with shape `[200,200,16]`.
The example's 32 validation frames form a deterministic official-split subset;
use `--val-count 6019` to calculate the complete Occ3D validation benchmark.

## Train

Run a small capture/training smoke test first:

```bash
qat/run_slurm.sh \
  --epochs 2 \
  --freeze-epoch 1 \
  --batch-size 1
```

Then use a longer warm-up and recovery schedule, for example:

```bash
qat/run_slurm.sh \
  --epochs 8 \
  --freeze-epoch 4 \
  --batch-size 1 \
  --output qat/runs/m0-full
```

Each epoch saves resumable camera/BEV/optimizer state under `checkpoints/`.
Before the first epoch, 32 calibration batches warm the observers; adjust this
with `--observer-warmup-batches`. Observer state is restored instead when
resuming.

Resume by recreating both prepared graphs and restoring a checkpoint:

```bash
qat/run_slurm.sh --epochs 8 --freeze-epoch 4 \
  --resume qat/runs/m0-full/checkpoints/epoch_003.pt
```

Finalization writes `onnx/camera.qdq.onnx` and `onnx/bev.qdq.onnx`. The export
report validates opset 17, requires both `QuantizeLinear` and
`DequantizeLinear` nodes, and runs both graphs with ONNX Runtime.
`metrics.json` records ground-truth cross-entropy, camera-masked accuracy,
semantic mIoU (classes 0-16), and FP32 fidelity at each stage.

