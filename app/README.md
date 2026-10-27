# Modalix replay application

`flashocc_replay` executes the camera and BEV compiled archives with threaded
ARM NEON pooling and occupancy argmax. It consumes already-normalized six-camera
`.frame` files for model accuracy and layout validation. The separate
[RTSP application](INSIGHT.md) adds decoding, synchronization and visualization.

## Contracts

- Six image views, normalized NHWC `[6,256,704,3]` FP32 in upstream camera order.
- Camera calls return depth `[16,44,44]` and features `[16,44,64]` per view;
  a leading singleton batch dimension is accepted.
- Pooling uses supplied calibrated `ranks_depth`, `ranks_feat` and `ranks_bev`.
- BEV input is `[200,200,64]`; output is `[200,200,288]` FP32 logits after
  dequantization. Restore the height/class axes and XY convention before argmax.
- Labels are `[200,200,16]` in XYZ order; 18 semantic classes, free class 17.

The binary checks tensor shapes, dtype, continuity and buffer sizes. Pooling
uses persistent ARM workers and non-fused arithmetic for reference agreement.
It selects camera output roles by shape, rather than assuming output order.

## Build

Inside the ARM64 Neat SDK, from the source root:

```bash
cmake -S app -B build/modalix \
  -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/modalix -j3
```

The application defaults to the current working directory as workspace. Pass
`--workspace` explicitly when launching over SSH. Model paths supplied through
CLI or JSON may be absolute or workspace-relative. The run controller and
comparison runner pass the workspace explicitly.

## Reproduce accuracy

Use the preserved 32 inputs and masks, exact model hashes and the comparison
runner described in the [root quickstart](../README.md). It regenerates original
FP32 predictions, packages replay frames and executes all three compiled pairs:

```bash
python3 scripts/reproduce_comparison.py accuracy \
  --asset-root /path/to/restored-assets \
  --results artifacts/comparison-repeat \
  --python /path/to/validation-environment/bin/python
```

The existing `.frame` format embeds original FP32 predictions as a secondary
numerical-fidelity reference. That field is not ground truth. The GT evaluator
separately loads `semantics` and `mask_camera` from the 32 pinned NPZ inputs,
aggregates the confusion matrix and computes classes 0–16 mIoU. Actual masked
accuracies/mIoUs are in [QAT_COMPARISON.md](QAT_COMPARISON.md).

For manual execution, run from the repository root on Modalix:

```bash
build/modalix/flashocc_replay --workspace "$PWD" \
  --input-dir artifacts/comparison-repeat/gt-inputs \
  --output-dir artifacts/replay \
  --warmup 2 --repeat 1 --check-finite-logits true
```

Default archives are the original BF16 camera and BF16/INT8-Softplus BEV pair.
Pass `--camera-model` and `--bev-model` for the small mixed or QAT pair. Use
`--inspect` to print boundary contracts without inference and `--dump-tensors`
for selected debugging runs. Large intermediate dumps are generated artifacts.

## Timers and validation

`compute_ms` includes pooling plan, input tensor setup, camera calls and copies,
pooling, BEV inference/copy and argmax. Validation/reference comparisons and
file I/O are separate. Model-call timing includes runtime and conversions;
it is not pure MLA hardware execution. Replay timing excludes RTSP and rendering.

Run the existing ARM64 correctness targets on Modalix:

```bash
build/modalix/pooling/pool_bench --self-test
build/modalix/pooling/test_labels
build/modalix/pooling/test_raycast
build/modalix/test_insight
```

Experimental camera-batch/transport/packing targets keep their CMake names;
their source and instructions are under [experiments](../experiments/README.md).
