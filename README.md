# FlashOCC on Modalix

The [host 3D viewer](viewer/README.md) provides live orbitable semantic voxels,
class and height filtering, and per-voxel inspection alongside the Insight video.

The complete C++ application runs the compiled camera and BEV networks on MLA
with threaded ARM NEON pooling and label selection. The
[Insight demo](app/INSIGHT.md) consumes six RTSP clips and displays live occupancy
projections and BEV grids. The [batch replay application](app/README.md) provides
reference comparisons and tensor dumps.

The [batch-six camera experiment](app/CAMERA_BATCH.md) compiles and benchmarks
all six views in one invocation. It lowers camera latency, but its current FP32
fidelity is worse, so the live application continues using batch one.

The preferred hardware-validated candidate uses a BF16 camera network and a BF16 BEV
network with INT8 Softplus. Both compiled for Modalix. Complete six-camera host
checks on two holdout samples gave 98.92% and 99.34% label agreement with FP32.
The complete C++ application also passed eight real samples on Modalix, with
98.92–99.40% FP32 label agreement and 141.3 ms median compute time. Ground-truth
accuracy remains unmeasured. See [application instructions](app/README.md).

Candidate archives, hashes, and validation summary: `artifacts/modalix-candidate.json`.
The earlier synthetic and fully INT8 artifacts are retained for comparison and
should not be mistaken for the preferred candidate.

Initial target: official FlashOCC M0, ResNet-50, one time step, six cameras,
256×704 images. Upstream source is in `FlashOCC/`, pinned during this work to
`449dc7947d6549ccb72a99d6f6d8c02f166ec0dd`.

Official checkpoint: https://drive.google.com/file/d/14my3jdqiIv6VIrkozQ6-ruEcBOPVlWGJ/view

Checkpoint SHA256: `cb776378659af93c2e6968d89767e4c03b6d3720d11fc2071e57b8d5af1d1cd5`.

## Deployment split

1. Camera network: `[1,3,256,704]` → depth `[1,44,16,44]` and features `[1,64,16,44]`.
   Run this stage once per camera, in upstream camera order.
2. BEV pooling outside the neural-network graphs: combine all six camera outputs
   using rank arrays generated from actual calibration and image transforms.
   Output `[1,64,200,200]`. The upstream implementation uses custom CUDA
   `mmdeploy::bev_pool_v2`; it cannot be assumed portable to Modalix.
3. BEV encoder and occupancy head: `[1,64,200,200]` → `[1,288,200,200]`.
4. Restore logits with `transpose(0,3,2,1).reshape(1,200,200,16,18)`, then
   argmax over the 18 semantic classes. Class 17 is free space.

The camera order is front-left, front, front-right, back-left, back, back-right.
Inputs must use the original image resizing/cropping, RGB normalization, and
correspondingly updated camera transforms. Arbitrary photographs are insufficient
for a calibrated 3D result.

## Export and checks

```bash
source /home/ashishnegi/.bashrc
activate-model-compiler
python scripts/export_flashocc.py --checkpoint artifacts/flashocc-r50-M0.pth --output artifacts/onnx-original
python scripts/export_flashocc.py --checkpoint artifacts/flashocc-r50-M0.pth --output artifacts/onnx-modalix --conv-head
python scripts/pool_reference.py
```

The exporter reuses upstream FPN, BEV backbone/neck, and head class bodies through
a narrowly scoped MMCV adapter. The image backbone uses torchvision ResNet-50
with the matching PyTorch stride convention. All 430 checkpoint entries load
strictly. This avoids installing the legacy CUDA training stack, but comparison
against an independently executed full upstream model on real data is still needed.

Original ONNX graphs were audited before the rewrite. The rewrite converts the
head's two pointwise Linear layers to equivalent 1×1 convolutions and leaves
height/layout restoration outside the compiled graph. It preserves logits after
restoring their layout; synthetic-input maximum absolute difference was
`2.6703e-5` (asserted with `rtol=1e-4`, `atol=1e-4`). ONNX Runtime vs PyTorch
maximum absolute differences were `1.9073e-6` for depth, `1.2517e-5` for camera
features, and `3.4332e-5` for rewritten BEV logits (`rtol=1e-3`, `atol=1e-3`).

After ONNX simplification, both stages have no unsupported or unknown operator
types under the bundled SiMa release 2.1 INT8 policy. BF16 audit flags Softplus in
the BEV stage. Operator-type audits do not prove all shape/attribute constraints
or hardware execution. Audit reports and validation results are in `artifacts/`.

## Compilation feasibility

Both stages compiled successfully with SiMa Model Compiler 2.1.3 for Modalix.
Deployment archives are `artifacts/compile-smoke/camera/flashocc_m0_camera_mpk.tar.gz`
and `artifacts/compile-smoke/bev/flashocc_m0_bev_mpk.tar.gz`. Each stage contains
one MLA segment plus EV74 support plugins; pooling is external to both archives.

```bash
export SIMA_MLA_SIM_PARALLEL=2
python scripts/compile_smoke.py --stage camera
python scripts/compile_smoke.py --stage bev
```

This uses the import API with explicit names, shapes, types, and NHWC calibration
tensors, including the 64-channel BEV feature input. Calibration is synthetic:
the resulting artifacts are **not production calibrated**. Camera INT8
quantization emitted saturation warnings, including weights being zeroed in a
layer. These artifacts must not be used as evidence of accuracy.

## CPU integration reference

```bash
python scripts/infer_reference.py --inputs /path/to/preprocessed-six-camera-inputs.npz
```

NPZ fields: float32 `images` `[6,3,256,704]`, and one-dimensional integer
`ranks_depth`, `ranks_feat`, `ranks_bev` produced by upstream
`voxel_pooling_prepare_v2`. The NumPy pooling reference passes the published
upstream pooling fixture. It is a correctness reference, not a performance kernel.
The complete split ONNX pipeline was exercised with synthetic images/ranks and
produced labels `[1,200,200,16]`; these labels have no scene accuracy meaning.
The final labels agreed exactly with the original exported BEV head for this
synthetic integration input.

## Remaining validation

Real mini images and calibration are now available. Occupancy ground truth is
still required to measure accuracy loss.
A Modalix device connection is required to implement/profile pooling and tensor
transfers, run the compiled stages, and measure full six-camera latency and memory.
No device inference or ground-truth accuracy measurement has been performed yet.

## Real nuScenes mini data

`artifacts/data/nuscenes-mini/` contains all 2,424 keyframe camera images and 13
metadata tables, streamed from the official archive. LiDAR and sweeps were
omitted. The download manifest records the source and scope. Image/geometry
preprocessing reuses the original class methods; the normalization reproduces
upstream OpenCV operations, including the channel reversal of PIL images.

```bash
python scripts/download_mini_subset.py --output artifacts/data/nuscenes-mini
python scripts/prepare_nuscenes.py --data artifacts/data/nuscenes-mini
python scripts/validate_real.py
python scripts/preview_real.py
```

Preparation selects four samples from each of the eight official mini training
scenes (32 six-camera calibration samples), and four from each of the two mini
validation scenes (eight holdout samples). The scene sets are disjoint.
Processed inputs and their sample/camera provenance are in
`artifacts/real-inputs/manifest.json`. Calibration tensors are in
`artifacts/real-calibration/{camera,bev}/{calibration,validation}/`.

All 40 samples passed ONNX vs adapted PyTorch numerical checks. Maximum absolute
errors: camera features `1.0491e-4`, depth `1.3053e-5`, BEV logits `1.4114e-4`.
The NumPy pooling output matched independent PyTorch accumulation exactly.
Minimum final-label agreement was `99.9996875%`; near-tied logits can switch
argmax because of FP32 rounding. This verifies the conversion, not prediction
accuracy or independent full upstream runtime parity.

Real-data quantization and compilation use:

```bash
export SIMA_MLA_SIM_PARALLEL=2
python scripts/compile_smoke.py --stage camera --precision bfloat16 --executor jax --calibration-dir artifacts/real-calibration/camera/calibration --validation-dir artifacts/real-calibration/camera/validation --verify-count 2 --build-dir artifacts/compile-real
python scripts/compile_smoke.py --stage bev --calibration-dir artifacts/real-calibration/bev/calibration --validation-dir artifacts/real-calibration/bev/validation --verify-count 8 --build-dir artifacts/compile-real
python scripts/validate_quantized_pipeline.py --count 2
```

The camera stage uses 192 actual image tensors. The BEV stage uses 32 pooled
feature tensors generated by the FP32 camera network. Full quantized pipeline
validation propagates quantized camera features through pooling into the
quantized BEV network. All SDK checks run on the host, not a Modalix device.

FP32 previews for the two holdout scenes are in `artifacts/real-predictions/`.
The official Occ3D `gts.tar.gz` download currently returns a Google Drive quota
error. Thus no mIoU or ground-truth occupancy accuracy is claimed. Mini provides
an initial calibration check; a larger diverse training subset is still desirable.

## Precision findings

Real-data INT8 camera artifacts are preserved in `artifacts/compile-real-int8/camera/`.
They still produced 12 weight-zeroing warnings and camera feature relative RMSE
of approximately 16–30% across 12 checked holdout camera images. They are not the
preferred camera artifacts.

[Layerwise PTQ analysis](app/PTQ_ANALYSIS.md) now separates local operator error
from accumulated error for both INT8 networks. The camera's zeroed filters had
negligible original FP32 weight energy, so those warnings alone do not explain
its prediction differences. The largest measured local errors occur at BEV
residual merges; the report includes ranked errors and reproduction commands.

The BF16 camera compiled successfully without weight-zeroing warnings and is
stored in `artifacts/compile-real/camera/`. Two real camera checks with the JAX
executor measured feature relative RMSE of approximately 1.47–2.10%. The first
check also completed with the normal executor, with essentially identical errors;
the remaining normal-executor checks were stopped because BF16 simulation was
slow. `verification-jax.json` records the completed checks.

The INT8 BEV stage in `artifacts/compile-real/bev/` compiled, but its labels agreed
with the FP32 stage on only approximately 68–81% of voxels across eight holdout
inputs. The complete BF16-camera + INT8-BEV pipeline gave 76.47% and 65.88% label
agreement on one sample from each holdout scene. These are conversion-agreement
measurements, not ground-truth accuracy scores.

The candidate BEV policy is BF16 except for INT8 `Softplus`. It uses the public
[per-node quantization API](https://docs.sima.ai/pages/model-sdk/quantize.html)
and gets its node name from the saved INT8 IR. The operator audit explicitly
checks BF16 support for all other operator types and INT8 support for Softplus.

```bash
python scripts/compile_smoke.py --stage bev --precision mixed-softplus --executor jax --calibration-dir artifacts/real-calibration/bev/calibration --validation-dir artifacts/real-calibration/bev/validation --verify-count 8 --build-dir artifacts/compile-real-mixed
python scripts/validate_quantized_pipeline.py --count 2 --bev-dir artifacts/compile-real-mixed/bev --output artifacts/compile-real-mixed/pipeline-validation.json
```

This keeps the original model math and two neural-network stages. Precision
conversions around Softplus are handled inside the compiled BEV graph.

The mixed BEV network compiled successfully in one MLA segment and completed all
eight holdout checks. Label agreement with FP32 was 99.46–99.61% with FP32 camera
features. With BF16 camera outputs propagated through pooling, complete pipeline
results were:

| Holdout scene | All-voxel label agreement | Occupied-union label agreement | Occupancy binary agreement |
| --- | --- | --- | --- |
| scene-0103 | 98.92% | 98.35% | 99.46% |
| scene-0916 | 99.34% | 98.72% | 99.66% |

These are host SDK JAX simulation comparisons to ONNX FP32 predictions, not
accuracy against labels or measurements of device behavior. Eight BEV-only tests
and two complete pipeline samples are an initial check, not a comprehensive
evaluation. Comparison figures are in `artifacts/real-predictions/*-precision-comparison.png`.

Preferred deployment archives:

- `artifacts/compile-real/camera/flashocc_m0_camera_mpk.tar.gz`
- `artifacts/compile-real-mixed/bev/flashocc_m0_bev_mpk.tar.gz`

The C++ CPU pooling implementation has now been validated on Modalix; see
[CPU pooling implementation and device results](cpp/README.md). Non-fused NEON
matched all eight real pooling references exactly and took approximately 3.2 ms
with sixteen threads. Neural-network hardware execution, tensor transfers,
six-camera scheduling, and full-pipeline latency/memory still need validation.

## Full upstream ONNX attempt

`artifacts/full-upstream/flashocc_m0_original.onnx` is a single M0 graph containing
the camera network, the original custom pooling operation, BEV network, original
Linear/Softplus head, and channel-to-height output reshape. It uses upstream
opset 11, the six original input names, dynamic rank/interval lengths, and one
`output_0` tensor `[1,200,200,16,18]`. No splitting, simplification, head rewrite,
or replacement of the exported pooling node was applied.

The exporter executes the original `BEVDetOCCTRT.forward_ori` and the original
`TRTBEVPoolv2.symbolic` class method. As the legacy CUDA/MMCV environment is not
installed, it uses the same strict-checkpoint CPU module adapter as earlier and
substitutes CPU pooling **only for tracing execution**. The resulting ONNX still
contains `mmdeploy::bev_pool_v2` with its seven inputs and original attributes.
This is not a byte-for-byte export from the untouched CUDA CLI/environment.

ONNX checker passed. The full CPU forward's labels matched the previous FP32
pipeline exactly on the selected real holdout sample. Full ONNX runtime numerical
verification is unavailable without a custom pooling runtime implementation.

Both INT8 and BF16 operator audits were saved before trying AFE. In addition to
the unknown custom pooling node, the unsimplified graph contains shape operations
not represented in the audit DB. INT8 audit flags Gather; BF16 flags Softplus.
These audit findings are not all compiler failures: the actual attempt stops at
the custom operation first.

```bash
source /home/ashishnegi/.bashrc
activate-model-compiler
python scripts/export_full_upstream.py
python /home/ashishnegi/.codex/skills/sima-model-surgery/scripts/model_surgery_guard.py audit-model --model artifacts/full-upstream/flashocc_m0_original.onnx --dtype int8 --json > artifacts/full-upstream/original.int8.audit.json
python /home/ashishnegi/.codex/skills/sima-model-surgery/scripts/model_surgery_guard.py audit-model --model artifacts/full-upstream/flashocc_m0_original.onnx --dtype bfloat16 --json > artifacts/full-upstream/original.bfloat16.audit.json
python scripts/try_compile_full_upstream.py
```

The direct import API binds representative shapes and float32/int32 types without
altering the original ONNX file. The import attempt is intentional despite audit
warnings, to answer whether the original custom operation is accepted by AFE.
AFE reports:

```text
The following operators are not supported for frontend ONNX: bev_pool_v2
Loading into TVM error.
```

Consequently, no quantization, accelerator partitioning or ELF compilation occurs.
The SDK terminates with SystemExit even though importing failed; the diagnostic
script catches this and records `imported=false`, `quantized=false`, and
`compiled=false`. Export details, audits, import traceback and compile result are
saved under `artifacts/full-upstream/`.

## Modalix application

The complete C++ replay application now runs the compiled camera and BEV models
with threaded ARM NEON pooling and label selection on Modalix. See
[app/README.md](app/README.md) for build/run commands, input/output contracts and
measured device results. Eight real six-camera samples passed two repeat runs;
independent tensor checks confirmed exact CPU pooling and label selection.

The [Insight demo](app/INSIGHT.md) consumes six real RTSP camera clips on Modalix
and displays live occupancy projections and BEV grids in the Insight viewer.
Its [execution settings](app/EXECUTION.md) select six sequential camera calls or
one spatially packed call, ARM or EV74 camera preprocessing, independent
camera/BEV pre/post placement, model archives and CPU worker counts. Use
`app/execution-packed.json` to run the packed camera model in the complete app.

