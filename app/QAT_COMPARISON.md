# QAT versus previous FlashOCC application approaches

The [cleanup preservation baseline](../benchmarks/flashocc_20261027/README.md)
retains the exact results, per-frame evidence, accuracy evaluation tools and
checksummed asset inventory required to reproduce these comparisons.

Measured on **2026-10-27**, on **`sima@10.42.0.102`**, using installed Neat
**0.5.0** and the existing ARM64 application. QAT with EV74 preprocessing reaches
**8.31 complete six-camera updates/s**, versus **6.50** for the previous mixed
models with EV74: **27.7% higher throughput**.

On the supplied **32 ground-truth samples**, camera-masked mIoU is **31.87%**
for QAT versus **33.57%** for the previous small mixed model pair. This is a
**1.70 percentage-point reduction** in exchange for the measured speed gain.
The original FP32 baseline scores **34.58%** on the same set.

## Models and test conditions

- **Previous mixed:** original BF16 camera plus
  `artifacts/compiled_with_better_quantization/bev_small_mixed_mpk.tar.gz`.
- **QAT:** both packages from `artifacts/compile_with_qat_better_accuracy/`.
  The archived compiler script describes W8A8 PTQ applied to QAT-trained weights
  after stripping Q/DQ. These are not the original checkpoint's weights.
- Both routes use six sequential batch-one camera calls, A65 output conversion,
  eight pooling workers, eight argmax workers and six rendering workers.
- Live tests use six 12 FPS RTSP streams, occupancy overlays, a 1536×864 Insight
  video and 2 Hz occupancy-volume snapshot export. Each test completes 256
  groups and excludes 64 warmups, leaving 192 measured groups. Tests run
  sequentially without other application workloads on the device.
- No compiler or runtime patch was applied. The QAT BEV package uses the
  separately generated shape-metadata compatibility copy described in
  [QAT application setup](QAT_APPLICATION.md). Its compiled ELF is unchanged.

## Full-application timing breakdown

Times below are **mean milliseconds per completed six-camera update**. The
logit validity scan is disabled in these four runs. Means include occasional
snapshot and preview writes; these costs are easily hidden by median timings.

| Component | Compute unit | Previous + ARM | Previous + EV74 | QAT + ARM | QAT + EV74 |
| --- | --- | ---: | ---: | ---: | ---: |
| Frame/geometry preparation and ARM normalization where selected | ARM | 75.73 | 3.36 | 76.10 | 3.57 |
| Camera tensor input setup/copies | ARM | 6.33 | 0.02 | 6.36 | 0.02 |
| Six camera calls, including output copies | MLA + runtime/conversions; EV74 preprocessing when selected | 49.79 | 60.75 | 27.79 | 31.45 |
| Pooling plan/rank preparation | ARM | 6.24 | 6.45 | 6.25 | 6.41 |
| BEV pooling | ARM NEON | 5.05 | 5.34 | 5.29 | 5.19 |
| BEV input tensor copy | ARM | 3.04 | 2.98 | 3.05 | 2.97 |
| BEV inference, including conversions and output copy | MLA + ARM runtime/conversions | 25.75 | 25.94 | 23.90 | 23.98 |
| Occupancy argmax | ARM NEON | 4.34 | 4.56 | 4.58 | 4.51 |
| Raycasting overlays and mosaic rendering | ARM | 32.34 | 32.50 | 31.67 | 31.86 |
| 3D snapshot publication, averaged over every update | ARM + shared filesystem | 9.67 | 6.75 | 8.72 | 5.11 |
| Other application bookkeeping/output handling | ARM | 4.78 | 4.70 | 4.70 | 4.85 |
| **Complete application processing** | | **223.06** | **153.33** | **198.42** | **119.93** |
| **Measured completed updates/s** | | **4.48** | **6.50** | **5.03** | **8.31** |

The camera-call row includes EV74 image preprocessing in the EV74 columns.
Compare it together with ARM preparation and input-copy work. ARM still performs
NV12-to-BGR conversion in background camera readers. VPU decoding and VPU video
encoding also run in the background; no isolated timings for those activities
were measured. Model-call rows include dispatch, synchronization, conversions
and copies, so they are not measurements of pure MLA hardware execution.

The complete application timer begins after six synchronized camera views are
available. It excludes waiting for inputs and network/browser delivery. FPS is
measured separately from completion timestamps. Snapshot export is capped at
2 Hz, so its per-update average varies with application FPS. Browser rendering
of interactive 3D voxels is outside these device timings.

The main QAT gain is the camera route: **60.75 → 31.45 ms** with EV74
preprocessing. The BEV route improves modestly, **25.94 → 23.98 ms**. Rendering
stays around 32 ms. With ARM preprocessing, the 76 ms preparation cost remains,
so the full-app gain is smaller: **4.48 → 5.03 FPS**.

### Effect of the logit validity scan

| QAT + EV74 | Mean scan | Mean application processing | Completed updates/s |
| --- | ---: | ---: | ---: |
| Check disabled | 0 ms | 119.93 ms | 8.31 |
| Check enabled | 33.95 ms | 156.32 ms | 6.38 |

Both settings retain the same prediction computation. The enabled scan checks
all FP32 output logits for NaN/infinity. Other timings fluctuate slightly
between the independent runs.

All five tests had zero invalid camera markers and no browser JavaScript
errors. The QAT/EV74 browser probe measured 8.295 total and presented frames/s
with zero dropped frames during its separate 20-second window. The QAT/ARM
probe counted six drops; its presented-frame estimate was 4.747 FPS versus
5.046 total-frame FPS. App FPS is independent of these browser counters.
At 12 FPS input, QAT/EV74 skips 31.0% of older synchronized input groups while
producing complete six-camera predictions. These are short steady-state tests.

## Ground-truth accuracy on the supplied 32-sample set

The supplied `gt/*.npz` files contain images, calibration/pooling ranks,
semantic labels and camera/lidar visibility masks. Every model pair uses all
32 identical image/rank inputs. They cover 2,153,160 camera-visible voxels.
The dataset manifest identifies 512 calibration and 32 validation samples from
nuScenes v1.0-trainval. This is the supplied 32-sample subset, not the complete
nuScenes validation benchmark.

The metric follows `FlashOCC/projects/mmdet3d_plugin/core/evaluation/occ_metrics.py`:
apply `mask_camera`, accumulate one 18×18 confusion matrix across all samples,
and average class IoUs for classes 0–16, excluding free class 17. Classes with
zero union are omitted, matching the reference's `nanmean`. Masked voxel
accuracy includes all 18 classes and is a separate metric from semantic mIoU.
Per-sample mIoUs are not averaged to obtain the aggregate score.

| Model pair | Execution | Camera-masked mIoU | Camera-masked voxel accuracy | mIoU difference from FP32 |
| --- | --- | ---: | ---: | ---: |
| Original FP32 ONNX | Host CPU | **34.58%** | 89.34% | Reference |
| Original BF16-heavy camera/BEV, with INT8 Softplus | Modalix | **34.65%** | 89.36% | +0.06 pp |
| BF16 camera + small mixed BEV | Modalix | **33.57%** | 89.22% | −1.01 pp |
| New QAT camera + QAT BEV | Modalix | **31.87%** | 86.00% | −2.71 pp |

All three compiled model pairs run on the new device with the same application
replay, eight pooling/argmax workers and an enabled finite-logit scan. The FP32
reference runs the existing original ONNX camera and BEV exports with FP32
pooling on the host. The shared previously validated sample reproduces its
saved original FP32 labels exactly.

QAT improves full-app throughput by 27.7% over small mixed precision and loses
**1.70 pp mIoU** / **3.22 pp masked voxel accuracy** on this supplied set. This
is the observed comparison of the deployed model pairs; it does not isolate
the effect of QAT training, PTQ re-quantization, export or compiler execution.
That isolation still requires the matching QAT FP32 exports and a reference
run with those same weights.

These accuracy tests use the supplied pre-normalized image tensors. RTSP
compression, NV12 conversion and EV74 image normalization are excluded, as in
model-level evaluation. Actual EV74/video-route mIoU was not measured on these
32 samples; the separate decoded-video fixture comparison below quantifies
its numerical difference from ARM preprocessing. The mIoU row must not be read
as a measured end-to-end EV74/video accuracy score.

The metric code, sample hashes, per-class IoUs, per-sample metrics and complete
confusion matrices are saved in `ground-truth.json`. The original FP32 and
three hardware prediction sets are under `gt-replay/`. All model outputs passed
finite/range checks. User-provided GT and report/manifest files were retained.

## Prediction agreement with the original checkpoint

For comparison with earlier investigations, all three compiled model pairs
were also rerun on the same eight lossless samples from scene-0103 and
scene-0916. These are **unmasked prediction-agreement percentages**, not mIoU
or ground-truth accuracy. References are the original FP32 model's outputs.

| Model pair | Mean voxel-label agreement | Per-sample range | Mean occupied-union semantic agreement |
| --- | ---: | ---: | ---: |
| Original BF16-heavy pair | 99.18% | 98.92–99.40% | 98.53% |
| BF16 camera + small mixed BEV | 95.66% | 94.07–96.90% | 92.65% |
| QAT camera + QAT BEV | 63.06% | 44.36–73.90% | 48.92% |

QAT predictions differ substantially from the old checkpoint, including in
regions outside the ground-truth camera visibility mask. This is why the
63.06% agreement score must not be interpreted as QAT ground-truth accuracy.
The masked mIoU table above is the appropriate task-accuracy comparison.
The occupied-union agreement excludes cells both models call free.

The QAT archive's validation script uses the same reshape, XY-axis swap and
free class ID 17 as the application. Class IDs are valid. Swapping XY again
worsens agreement with the old model to 31.89–51.47%, so it does not restore
those predictions. The cause of the prediction change has not been isolated.
`layout-check.json` records these diagnostics and class counts.

## Accuracy evidence: EV74 versus ARM preprocessing

These saved fixture tests compare identical decoded NV12 pixels with the
**same model pair on both sides**, on two six-camera frames. They are separate
from the eight-sample FP32 comparison above and were measured before this new
device run.

| Model pair | EV74/ARM voxel-label agreement | Logit relative RMSE |
| --- | ---: | ---: |
| Original BF16-heavy pair | 99.32–99.56% | See original fixture report |
| BF16 camera + small mixed BEV | 98.26–98.75% | 1.67–1.80% |
| QAT camera + QAT BEV | **97.42–97.92%** | **4.02–4.52%** |

EV74 introduces additional numerical differences with the QAT pair. These
numbers describe agreement, not lost ground-truth accuracy. They must not be
combined arithmetically with the FP32-agreement table to estimate accuracy.

## Earlier camera packing approach

The earlier two-camera packing test used three calls per update, ARM
normalization and the small mixed BEV. With the scan disabled, it achieved
**4.58 FPS** at 8 FPS input with overlays and 2 Hz snapshots. Its labels were
byte-identical to the sequential ARM/small-mixed reference on eight samples.
This is a historical measurement, not a fresh run on the new address. Packing
preserved that model pair's outputs, while the ARM preparation cost limited
full-app performance. The fresh QAT/EV74 route is the fastest measured route.

## Evidence and reproduction

Current evidence is in `artifacts/qat-comparison-20261027/`:

- `comparison.json`: complete statistics, archive hashes and execution settings.
- `ground-truth.json`: 32-sample camera-masked accuracy, per-class IoU and confusion
  matrices for original FP32 and all three compiled model pairs.
- `gt-replay/`, `predict_fp32.py`, `evaluate_gt.py`: predictions and accuracy
  generation/evaluation resources; `gt-inputs/` contains the packaged input frames.
- `live/<variant>/12/`: all five full-app logs, summaries, browser counters and
  screenshots. Variants are `mixed-arm`, `mixed-ev74`, `qat-arm`, `qat-ev74` and
  `qat-ev74-check`.
- `replay/{original,mixed,qat}/`: all three eight-sample hardware replays and
  output labels; `layout-check.json` records the additional checks.
- `methodology.json` and `summarize.py`: test conditions and aggregation.

To repeat one live route with the saved execution configuration:

```bash
python3 scripts/benchmark_insight_fps.py \
  --device sima@10.42.0.102 \
  --rates 12 --frames 256 --warmup 64 --volume-fps 2 \
  --execution artifacts/qat-comparison-20261027/qat-ev74.json \
  --output-dir artifacts/qat-comparison-repeat/qat-ev74
```

The harness stops its owned device process and six RTSP sources after each run.
The device had no running FlashOCC app before testing and none was left running.
The SDK runner failed `chmod` on the shared NFS executable, so device execution
used direct SSH. Original model archives, compiled ELFs and application binaries
were retained.

Earlier fixture evidence is in `artifacts/preprocess-validation/comparison.json`,
`artifacts/bev-small-mixed-ev74/fixtures/comparison.json` and
`artifacts/qat-app/fixtures/comparison.json`. Packing evidence is in
`artifacts/bev-small-mixed-packed2/results.json` and `replay-comparison.json`.
