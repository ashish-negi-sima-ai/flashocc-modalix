# Cleanup preservation baseline: FlashOCC comparison, 2026-10-27

The two tables in [the comparison report](../../app/QAT_COMPARISON.md) are
cleanup acceptance criteria. This directory retains exact values, raw numeric
evidence, asset identities and accuracy evaluation code. Cleanup must preserve
the ability to regenerate predictions and measure the complete application;
reprinting this saved baseline alone is not a fresh benchmark.

## Preserved evidence

- `baseline.json`: exact unrounded accuracy and average timing values, execution
  configurations and the Git revision from which this preservation bundle was
  created. This is not a claim that the original measurements used that Git SHA.
- `measurements.json`: the 256 per-frame records for each of five live runs,
  all four accuracy confusion matrices and per-sample accuracy results.
- `assets.json`: relative paths, sizes and SHA-256 hashes for the exact 32 NPZ
  input/label/mask files; FP32 ONNX networks; all compiled model pairs; the
  original and shape-adapted QAT BEV archives; reference ARM64 applications;
  six original and six 12 FPS videos; 40 geometry files; input provenance and
  execution configurations. These large files remain local for now.
- `predict_fp32.py`: original FP32 inference and optional replay-frame generation.
- `evaluate_gt.py`: all four prediction sets evaluated using the same GT masks.
- `check.py`: asset verification and recalculation of saved numeric evidence.

The preservation step kept the reference payloads in place. Subsequent moves,
archive restoration and independent reruns are recorded in the
[cleanup audit](../../reports/cleanup/README.md). A clean checkout needs the
external asset bundle and its checksum sidecar. Never replace the reference
snapshot with a rerun's output.

## Baseline conditions

Accuracy uses exactly 32 supplied pre-normalized six-camera NPZ inputs and
`mask_camera`, with 2,153,160 visible voxels. Accumulate one 18-class confusion
matrix; calculate mIoU over classes 0–16, excluding free class 17 and undefined
IoUs. Do not average per-frame mIoUs. Accuracy excludes RTSP/video conversion
and EV74 image normalization.

| Model pair | Camera-masked mIoU | Masked voxel accuracy |
| --- | ---: | ---: |
| Original FP32 | 34.58% | 89.34% |
| Original BF16-heavy | 34.65% | 89.36% |
| BF16 camera + small mixed BEV | 33.57% | 89.22% |
| QAT camera + QAT BEV | 31.87% | 86.00% |

Performance uses six sequential camera calls, 12 FPS RTSP inputs, occupancy
overlays, 1536×864 video, 2 Hz snapshots, scan disabled, 8 pooling/argmax workers
and 6 rendering workers. Each run processes 256 groups and excludes the first
64. Use mean milliseconds, not median latency or model-only timings. Snapshot
publication and other application work must remain included.

| Configuration | Mean application time | Completed updates/s |
| --- | ---: | ---: |
| Previous + ARM | 223.06 ms | 4.48 |
| Previous + EV74 | 153.33 ms | 6.50 |
| QAT + ARM | 198.42 ms | 5.03 |
| QAT + EV74 | 119.93 ms | 8.31 |

The complete per-component breakdown is stored in `baseline.json`. It includes
the extra QAT/EV74 scan-enabled control. Original measurements used Modalix at
`sima@10.42.0.102`, installed Neat 0.5.0 and A65 output conversion. Preserve the
matching compiler/runtime setup, model ELF contents and preprocessing semantics.
Version the execution environment when changing any of these.

## Verify before cleanup

From the repository root, no device is required:

```bash
python3 benchmarks/flashocc_20261027/check.py
# After restoring a bundle with the same relative directory structure:
python3 benchmarks/flashocc_20261027/check.py --asset-root /path/to/restored-assets
# Check only the small committed measurement records:
python3 benchmarks/flashocc_20261027/check.py --records-only
```

This recalculates timing means/FPS and masked accuracy from saved evidence.
The full check also verifies every retained asset, including GT hashes recorded
by the original evaluation. It does not run inference.

## Regenerate accuracy

The normal end-to-end entry point is `scripts/reproduce_comparison.py`, described
in the root quickstart. It verifies assets and regenerates predictions, replay
inputs and metrics in a new directory. The detailed commands below describe
those stages. Current source builds use `build/modalix`; reference binaries
under `artifacts/build-app` are preserved assets, not build output destinations.

Use an environment with NumPy and ONNX Runtime (original ORT version 1.21.1):

```bash
python benchmarks/flashocc_20261027/predict_fp32.py \
  --gt-dir gt \
  --results artifacts/comparison-repeat \
  --replay-input-dir artifacts/comparison-repeat/gt-inputs
```

This writes original FP32 predictions and generates all 32 `.frame` inputs with
the exact normalized images, rank arrays and FP32 reference labels. On Modalix,
run `build/modalix/flashocc_replay` three times, sequentially, with:

```text
--input-dir <shared-root>/artifacts/comparison-repeat/gt-inputs
--output-dir <shared-root>/artifacts/comparison-repeat/gt-replay/<original|mixed|qat>
--camera-model <camera archive from assets.json>
--bev-model <matching BEV archive from assets.json>
--camera-post-target A65 --bev-post-target A65
--threads 8 --warmup 2 --repeat 1 --check-finite-logits true
```

The original pair uses the BF16 camera and `compile-real-mixed/bev` archive;
the mixed pair uses that camera and `bev_small_mixed_mpk.tar.gz`; the QAT pair
uses the supplied QAT camera and compatible BEV archive. Absolute paths must
resolve on the shared workspace. Evaluate the four complete prediction sets:

```bash
python benchmarks/flashocc_20261027/evaluate_gt.py \
  --gt-dir gt --results artifacts/comparison-repeat
```

This writes `ground-truth.json` under the new results directory. Compare its
aggregate metrics with the reference to the displayed 0.01 percentage point.
Investigate differences instead of silently updating the reference.

## Regenerate performance

Use `scripts/benchmark_insight_fps.py` for each of `mixed-arm`, `mixed-ev74`,
`qat-arm` and `qat-ev74`, sequentially, while the accelerator is otherwise idle:

```bash
python3 scripts/benchmark_insight_fps.py \
  --device sima@10.42.0.102 \
  --api https://SDK_HOST:9900 \
  --rates 12 --frames 256 --warmup 64 --volume-fps 2 \
  --execution artifacts/qat-comparison-20261027/qat-ev74.json \
  --output-dir artifacts/comparison-repeat/live/qat-ev74
```

The four exact execution JSON files are listed in the asset manifest; their
settings are also preserved in `baseline.json`. Rewrite only host/path values
when restoring elsewhere. Preserve video pixels, frame order and geometry.
The existing harness verifies decoded-pixel hashes when retimestamping videos.
Record the actual device address and software versions for every new run.

The stage means are calculated after discarding 64 warmups. CPU preparation
combines `preparation_ms + camera_input_ms`; pooling combines
`geometry_plan_ms + pooling_ms`; BEV combines `bev_input_ms + bev_inference_ms`.
Other work is application time minus the explicitly timed components; combine
that residual with `volume_publish_ms` for the final table's snapshots/other row.
Do not double-count nested camera or compute timers.

Exact wall-clock timings vary. Fresh runs must use the same conditions and
report deviations from the recorded means/FPS; no numerical performance
tolerance has been agreed. Byte hashes and protocol identity are strict.

## Cleanup acceptance

Before removing a unique file or changing a path/API, restore the preserved
assets independently, regenerate all four accuracy rows and all four performance
columns, and retain the new raw evidence. Existing CPU correctness tests and
the Modalix build/run checks remain required for application changes. Reprinting
the snapshot or checking only Git-tracked files does not satisfy this gate.
