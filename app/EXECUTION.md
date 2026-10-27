Configurable camera inference, preprocessing and compute placement
======================================================

The existing ARM preprocessing path remains the default. Choose an execution
preset when starting the application:

```bash
python3 scripts/run_insight_demo.py start \
  --config artifacts/fps-inputs/4/config.json \
  --execution app/execution-ev74-bgr.json
```

Use `app/execution-arm-fp32.json` for the original path or
`app/execution-ev74-bgr.json` to offload normalization while retaining ARM color
conversion. `app/execution-ev74-nv12.json` is experimental: the current EV74 NV12
color conversion has materially lower agreement with the ARM reference, as
measured below. Stop the running application before changing execution settings;
the controller does not reconfigure an existing process.

Settings can also live in the source config's `execution` object. An
`--execution FILE` supplies an object of overrides. Direct CLI overrides include
`--camera-mode`, `--camera-pack-size`, `--camera-guard-width`, `--camera-input`, `--camera-pre-target`, `--camera-post-target`,
`--bev-pre-target`, `--bev-post-target`, `--pool-threads`, `--argmax-threads`,
`--render-threads`, `--camera-model`, `--camera-model-order`, `--bev-model`,
`--timeout-ms`, `--check-finite-logits true|false`, and `--visualization true|false`. CLI options are processed in
order, with later values winning. Restart to apply changes.

| Setting | Default | Meaning |
| --- | --- | --- |
| `camera_mode` | `sequential` | Six batch-one calls, or `packed` for 6 / `camera_pack_size` spatially packed calls |
| `camera_pack_size` | 6 | Cameras per packed call: 2, 3 or 6; requires a matching compiled archive |
| `camera_guard_width` | 32 | Packed input gap in pixels; must match the compiled archive |
| `camera_input` | `arm-fp32` | `arm-fp32`, `ev74-bgr`, or `ev74-nv12` |
| `camera_model_order` | `BGR` | Input channel order of the selected camera archive: `BGR` or `RGB` |
| `camera_model` / `bev_model` | Existing validated archives | Shared-workspace model paths |
| `camera_pre_target` / `camera_post_target` | `AUTO` | Independent `AUTO`, `A65`, or `EV74` requests; packed input defaults to A65 |
| `bev_pre_target` / `bev_post_target` | `AUTO` | Independent `AUTO`, `A65`, or `EV74` requests |
| `pool_threads` | 8 | ARM NEON pooling workers, 1–256 |
| `argmax_threads` | 8 | ARM NEON label-selection workers, 1–256 |
| `render_threads` | 6 | ARM camera-overlay workers, 1–6 |
| `visualization` | true | Enable raycasting, mosaic and Insight video; false leaves inference/volume export running |
| `check_finite_logits` | true | Scan every BEV output score for NaN/infinity; false skips that scan |
| `mean` | `[123.675,116.28,103.53]` | Raw-pixel means in BGR channel order |
| `stddev` | `[58.395,57.12,57.375]` | Raw-pixel standard deviations in BGR order |
| `timeout_ms` | 20000 | Model call timeout, 1–120000 ms |

`volume_fps` is a top-level source-config setting, default 2. The existing
`--volume-fps 0..30` overrides it; zero disables snapshot export. RTSP URLs,
source FPS and Insight destination retain their existing source-config fields.
Unknown execution keys, invalid thread counts and incompatible image-preproc
targets are rejected before the pipeline starts. `execution.json`, each frame
record and the final summary record the requested settings and model paths.
`AUTO` is a runtime placement request, not proof that a stage executed on EV74.

The output-logit validity scan can be disabled in any execution preset by adding
`"check_finite_logits": false`, or overridden when launching the live app:

```bash
python3 scripts/run_insight_demo.py start \
  --config artifacts/fps-inputs/8/config.json \
  --execution app/execution-bev-small-mixed.json \
  --output-dir artifacts/bev-small-mixed/running \
  --check-finite-logits false
```

Stop an existing app before starting it with a changed setting. Pass `true` to
enable the check again. Both `flashocc_insight` and `flashocc_replay` accept the
same `--check-finite-logits true|false` flag. With no override, the check remains
enabled. The selected value is recorded in execution settings and frame logs;
`logit_finite_check_ms` records the scan duration, or exactly zero when disabled.
This duration is included in the existing `validation_ms` metric. Replay
reference comparisons continue to run independently of the finite-logit check.

For EV74 BGR preprocessing with the supplied small mixed-precision BEV archive,
use `app/execution-bev-small-mixed-ev74-bgr.json`. The camera archive stays the
original BF16 single-camera model, with six sequential calls per update. This
preset enables the logit check; its CLI override remains available:

```bash
python3 scripts/run_insight_demo.py stop \
  --config artifacts/fps-inputs/8/config.json \
  --output-dir artifacts/bev-small-mixed/running
python3 scripts/run_insight_demo.py start \
  --config artifacts/fps-inputs/8/config.json \
  --execution app/execution-bev-small-mixed-ev74-bgr.json \
  --output-dir artifacts/bev-small-mixed/running \
  --check-finite-logits false
```

Measured full-app throughput was 5.39 FPS with the check enabled and 6.68 FPS
with it disabled, using 8 FPS streams, overlays and 2 Hz 3D snapshots. ARM retains
NV12-to-BGR conversion; EV74 handles normalization and model-input preparation.
See [the benchmark report](MODEL_BENCHMARK.md) for preprocessing agreement and
browser drop counts.

QAT W8A8 camera and BEV
----------------------

`app/execution-qat-ev74-bgr.json` selects both supplied archives from
`artifacts/compile_with_qat_better_accuracy/`, with six sequential camera calls
and EV74 BGR preprocessing. First create the compatible BEV package using
`scripts/prepare_qat_runtime_archives.py`: the supplied BEV archive has a
singleton-dimension mismatch between MLA output and its dequantization stage
on Neat 0.5.0. The adapter changes only those shape fields in a separate archive;
the compiled ELF and quantization parameters remain unchanged.

Measured full-app throughput with 12 FPS inputs was 8.26 FPS with the logit scan
disabled and 6.43 FPS with it enabled, including overlays and 2 Hz snapshots.
The preset keeps the scan enabled. See [QAT application results](QAT_APPLICATION.md)
for package preparation, commands, the controlled comparison, and the numerical
difference between ARM and EV74 preprocessing.

Packed camera inference
-----------------------

The full RTSP-to-Insight application supports the validated packed camera
archive. Select it with the supplied preset; the BEV network, ARM pooling,
raycasting, video output and host 3D snapshots use the existing pipeline.

```bash
python3 scripts/run_insight_demo.py stop
python3 scripts/run_insight_demo.py start \
  --config artifacts/fps-inputs/4/config.json \
  --execution app/execution-packed.json
```

To switch back, stop and start with `app/execution-arm-fp32.json`, or use
`app/execution-ev74-bgr.json` for sequential calls with EV74 normalization.
With no mode specified the application retains sequential inference.

In packed mode ARM normalizes six BGR images, places them side by side with
32-pixel zero guards, and submits one 256×4384×3 tensor. The camera archive is
`artifacts/compile-packed/camera/flashocc_m0_camera_mpk.tar.gz`. Its outputs are
16×274×44 depth probabilities and 16×274×64 features. ARM removes the two output
guard columns between cameras, restoring the same six-camera ordering consumed
by the existing NEON pooler. This is spatial packing with network batch one.
See [camera packing validation](CAMERA_BATCH.md) for graph preparation and
compiler/hardware evidence.

The preset uses `camera_input=arm-fp32` and A65 camera pre/post conversion.
Packed mode currently requires normalization before inserting guards; combining
it with `ev74-bgr` or `ev74-nv12` is rejected. AUTO and EV74 remain explicit
conversion options, but both measured approximately 4.3 seconds for this wide
input on the installed runtime, compared with approximately 7 ms on A65.

`camera_model` can override the default archive. The application checks batch
size and output shapes at startup, rejecting an archive that does not match
the selected mode and guard width. Changing `camera_guard_width` requires a
matching re-exported and compiled model; it does not modify archive geometry.
The supplied archive uses 32 pixels. Runtime timing records include
`camera_calls` (one, two, three or six), `camera_pack_ms`, `camera_route_ms` and
`camera_unpack_ms`. `camera_route_ms` includes model input/output conversion,
MLA execution and runtime overhead; it is not pure accelerator time.

For the intermediate two-call approach, use `app/execution-packed3.json`:

```bash
python3 scripts/run_insight_demo.py stop
python3 scripts/run_insight_demo.py start \
  --config artifacts/fps-inputs/4/config.json \
  --execution app/execution-packed3.json
```

`camera_pack_size=3` packs cameras 0–2 for the first call and cameras 3–5 for
the second. Each input is 256×2176×3 with two 32-pixel guards. Each call returns
16×136×44 depth and 16×136×64 features; the app removes the guard columns and
restores the original six-camera ordering before pooling. Network batch remains
one. The default archive for this configuration is
`artifacts/compile-packed3/camera/flashocc_m0_camera_mpk.tar.gz`.
CLI overrides are `--camera-mode packed --camera-pack-size 3`. Selecting a
different pack size requires the corresponding compiled model; startup rejects
an incompatible output shape.

Validated on Modalix on 2026-10-20: all depth, feature and occupancy-label files
were byte-identical to the sequential BF16 pipeline on eight six-camera groups
(48 images). FP32 equivalence checks also passed bit for bit. Neat measured
23.376 ms per three-camera call, or 46.752 ms for two calls; see
[model comparisons](MODEL_BENCHMARK.md) for the single- and six-camera results.

With overlays, Insight video and 2 Hz 3D snapshots enabled, this preset sustained
3.32 updates/s at 4 FPS input and 3.30 updates/s at 8 FPS input. Each run processed
180 groups and excluded 60 warmups, with no invalid camera markers. Browser
checks delivered approximately 3.30 FPS with no JavaScript errors; their
20-second windows recorded one dropped video frame at 4 FPS input and none at
8 FPS input. This option works as an intermediate call grouping, but does not
show a meaningful full-application FPS improvement in these measurements.
Evidence is in `artifacts/packed3-app/equivalence.json` and
`artifacts/packed3-app/live/summary.json`.

For three sequential calls with two cameras per input, select
`app/execution-packed2.json` instead. It sets `camera_mode=packed` and
`camera_pack_size=2`, grouping camera indices 0–1, 2–3 and 4–5. Each input is
256×1440×3, including one 32-pixel guard; each output is 16×90×44 depth or
16×90×64 features. The archive defaults to
`artifacts/compile-packed2/camera/flashocc_m0_camera_mpk.tar.gz`.
Network batch size remains one, and the six outputs are restored to their
original order before pooling. The same stop/start commands apply with
`--execution app/execution-packed2.json`.

To use this packing configuration with the supplied small mixed-precision BEV
archive, select `app/execution-bev-small-mixed-packed2.json`. It uses the same
packed camera archive and ARM FP32 preprocessing; `--check-finite-logits false`
can override its enabled logit check. On Modalix it measured 3.96 FPS with the
check enabled and 4.58 FPS with it disabled. Sequential EV74 preprocessing with
the same BEV archive measured 5.39 and 6.68 FPS respectively, so it remains the
faster complete-app configuration. See [the benchmark report](MODEL_BENCHMARK.md)
for test settings and output-equivalence results.

Validated on Modalix on 2026-10-20: all 48 held-out camera images produced
byte-identical depth/features and occupancy labels to the original sequential
BF16 pipeline, with exactly three camera calls per group. FP32 equivalence
also passed bit for bit. Neat measured 14.028 ms per call, or 42.085 ms for all
six cameras, using BF16 and A65 input/output conversion. This is the lowest
camera-route latency among the tested pack sizes; see
[the four-mode comparison](MODEL_BENCHMARK.md).

With overlays, Insight video and 2 Hz 3D snapshots enabled, this preset sustained
3.36 updates/s at 4 FPS input and 3.33 updates/s at 8 FPS input. Each run
processed 180 groups and excluded 60 warmups, with no invalid camera markers.
Browser checks delivered approximately 3.35 and 3.30 FPS, respectively, with
no JavaScript errors; their 20-second measurement windows recorded zero and two
dropped video frames. The full-app gain is small because the camera route is
only part of the complete processing time. Evidence is in
`artifacts/packed2-app/equivalence.json` and
`artifacts/packed2-app/live/summary.json`.

The shared replay executable also accepts the new mode:

```bash
# On Modalix, with the live application stopped:
artifacts/build-app/flashocc_replay --camera-mode packed \
  --camera-post-target A65 --bev-post-target A65 --limit 8
```

The integrated replay produced byte-identical depth, features and occupancy
labels to the original six-call BF16 pipeline on eight six-camera groups.
Evidence is saved in `artifacts/packed-app/equivalence.json`. This checks
agreement with the existing implementation, not ground-truth accuracy.

To benchmark the complete application, including visualization and 2 Hz volume
publication, use:

```bash
python3 scripts/benchmark_insight_fps.py --rates 4 8 --frames 180 --warmup 60 \
  --volume-fps 2 --execution app/execution-packed.json \
  --output-dir artifacts/packed-app/live
```

This benchmark controls the six demo sources and must run with the live
application stopped. `--binary PATH` selects an isolated build if needed.

Measured on Modalix on 2026-10-16 with overlays, Insight video and 2 Hz volume
publication enabled; each run processed 180 groups and excluded 60 warmup groups:

| Camera mode | Input FPS per camera | Complete app updates/s | Median compute | Median application time |
| --- | ---: | ---: | ---: | ---: |
| Packed | 4 | 3.30 | 143.32 ms | 303.72 ms |
| Packed | 8 | 3.29 | 143.24 ms | 305.92 ms |
| Sequential ARM FP32 | 4 | 3.30 | 143.81 ms | 305.01 ms |

Both packed browser checks delivered approximately 3.30 FPS with no dropped
frames or JavaScript errors during their 20-second measurement windows. All
three runs had zero invalid camera markers. At 4 FPS input the packed app
skipped 23 synchronized groups between the first and last measured outputs;
at 8 FPS it skipped 171. Packing reduces camera invocation count, but does not
improve full application throughput in these measurements. Earlier application
measurements elsewhere in these docs used different runs/settings; use this
table for the controlled comparison. Evidence is in
`artifacts/packed-app/application-comparison.json`, `live/summary.json` and
`sequential-live/summary.json`.

The three paths
---------------

* `arm-fp32`: VPU decode → ARM NV12-to-BGR → ARM normalization → Neat
  cast/tessellation → MLA. This preserves existing behavior.
* `ev74-bgr`: VPU decode → ARM NV12-to-BGR → EV74 normalization/BF16 conversion
  and the model's required layout preparation → MLA. On this installed runtime,
  BF16 image preprocessing uses RGB transport. The BGR bytes are intentionally
  carried through its identity path without a channel swap, preserving the
  original model's numerical channel order.
* `ev74-nv12` (experimental): VPU decode → cropped NV12 tensor → EV74 color conversion and
  normalization/layout preparation → MLA using the equivalent RGB-input camera
  archive. Inference does not need an ARM color conversion. If visualization is
  enabled, the six selected images are converted to BGR on ARM afterward for
  overlays. Disabling visualization removes that conversion and rendering work.

Both normalization paths expose statistics in raw-pixel units; the EV74 adapter
divides them by 255 for the image-preproc API. RGB camera inputs also reverse
the channel statistics. `ev74-nv12` requires `camera_model_order=RGB` and the
matching RGB camera archive. Changing the order declaration alone does not
convert an existing archive.

These paths support the current 704×288 demo clips. Their model image occupies
the top 704×256 pixels. Source resize/crop is already baked into the clips;
the synchronization footer is stripped on ARM. The NV12 path copies the Y and
UV regions separately so the footer is never included in model input. It is
not a zero-copy decoder-to-MLA pipeline.

The older BF16 BEV archive's output conversion failed when forced to EV74.
The supplied small mixed-precision archive instead has INT8 output with an
INT8-to-FP32 dequantization stage; that stage now runs correctly on EV74, as does
the single-camera model's detessellation/cast. On identical decoded fixtures,
both choices produced byte-identical tensors and labels to A65 output conversion.
However, the current full app measured 6.64 FPS with A65 output conversion,
6.56 FPS with only camera output conversion on EV74, and 5.96 FPS with both camera
and BEV output conversion on EV74 (all with EV74 input preprocessing and the
logit check disabled). The presets retain A65 output conversion for performance.
Explicit overrides remain `--camera-post-target EV74` and `--bev-post-target EV74`,
or the matching `camera_post_target` / `bev_post_target` execution settings.
See [the benchmark report](MODEL_BENCHMARK.md) and
`artifacts/ev74-output-conversion/` for the comparison. Pooling, argmax and
raycasting remain ARM algorithms; their thread settings do not imply EV74
implementations.

RGB camera artifact
-------------------

The native NV12 path produces RGB channels. Its camera variant reverses the
first convolution's three input-channel weights. When its normalized input is
the original normalized tensor with channels reversed, the FP32 model is
mathematically equivalent. Only that first weight tensor changes; the BEV
network is unchanged. The preparation script checks ONNX consistency and
numerical equivalence on 12 held-out camera images before compilation.

```bash
activate-model-compiler
python scripts/prepare_rgb_camera.py
python /home/ashishnegi/.codex/skills/sima-model-surgery/scripts/model_surgery_guard.py \
  audit-model --model artifacts/onnx-rgb/camera.onnx --dtype bfloat16 --json \
  > artifacts/onnx-rgb/camera.bfloat16.audit.json
python scripts/compile_smoke.py --stage camera --model-path artifacts/onnx-rgb/camera.onnx \
  --precision bfloat16 --executor jax \
  --calibration-dir artifacts/real-calibration-rgb/camera/calibration \
  --validation-dir artifacts/real-calibration-rgb/camera/validation \
  --verify-count 2 --build-dir artifacts/compile-rgb
```

Validation
----------

The `check_preprocess` ARM64 executable compares each execution preset on the
same saved six-camera NV12 frames, using the matching geometry. It dumps
depth, features, pooled tensors, logits and labels on its first measured
iteration. It performs a warmup pass and then the requested repeats.

```bash
cmake -S app -B artifacts/build-app -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/build-app -j3
python scripts/prepare_preprocessing_fixtures.py
# Run sequentially on Modalix with the live application stopped:
artifacts/build-app/check_preprocess artifacts/preprocess-validation/fixtures.json \
  app/execution-arm-fp32.json artifacts/preprocess-validation/arm-fp32 10
artifacts/build-app/check_preprocess artifacts/preprocess-validation/fixtures.json \
  app/execution-ev74-bgr.json artifacts/preprocess-validation/ev74-bgr 10
artifacts/build-app/check_preprocess artifacts/preprocess-validation/fixtures.json \
  app/execution-ev74-nv12.json artifacts/preprocess-validation/ev74-nv12 10
# SDK host:
python scripts/check_preprocessing_results.py
```

Fixture results are numerical agreement with the existing application, not
ground-truth occupancy accuracy. Preparation timing includes fixture reads and
color conversion; camera timing includes EV74 preprocessing in the offloaded
paths. Compare their combined work rather than camera timing alone. Fixture
total time includes validation and occasional tensor dumps and is not live FPS.

Measured results, 2026-10-15
----------------------------

The two six-camera fixtures are scene-0103 frames 0 and 13. Each primary route
was warmed up on both fixtures, then measured for 20 groups (10 repeats).
Agreement is against the original ARM preprocessing + BF16 pipeline on the
same decoded pixels, not FP32 or ground truth.

| Route | Preparation median | Six-camera call median | Voxel-label agreement, two fixtures |
| --- | ---: | ---: | --- |
| ARM FP32 | 16.03 ms | 46.81 ms | Reference |
| EV74 BGR normalization | 7.97 ms | 56.08 ms | 99.32%, 99.56% |
| EV74 native NV12, RGB camera | 5.21 ms | 56.61 ms | 92.91%, 95.88% |

These timings move preprocessing work across measurement boundaries; they do
not establish a pure MLA speedup. The EV74 BGR path's camera feature relative
RMSE was 1.70–1.88% against ARM. Its full live run completed 160 groups at
approximately 4 FPS with no skipped groups or invalid camera markers. The
browser measured 4.15 FPS over 20 seconds, no frame drops during that window,
and no JavaScript errors. A separate window after 40 warmup groups measured
median application processing time of 230.55 ms.

The RGB-input camera archive was compiled as BF16 and checked in host JAX
execution against its RGB ONNX. With ARM preprocessing on Modalix it produced
100% voxel-label agreement with the original camera archive on both fixtures.
EV74 preprocessing from already converted RGB images reproduced the BGR path's
99.32–99.56% agreement. This isolates the larger remaining mismatch to native
NV12 color preprocessing rather than the first-convolution channel reversal.
A range-compensation experiment improved native-NV12 label agreement to
97.51–98.59%, which still does not match the BGR path. It is not applied in the
presets; the exact conversion mismatch remains unresolved.

Use the BGR preset for the current demo. Native NV12 remains an explicitly
selected experimental route. The generic default stays ARM for compatibility.
Detailed evidence is in `artifacts/preprocess-validation/comparison.json`,
`live-bgr/summary.json`, `live-bgr/browser.json`, and
`artifacts/onnx-rgb/equivalence.json`.

Device tests also covered NV12 plane layout, luma-marker decoding, synchronized
camera ordering, independent placement requests, invalid configuration rejection,
and preservation of the ARM default. A 12-group headless run used four pooling
workers and two argmax workers, with video and volume export disabled; it completed
with no invalid markers and no preview/snapshot publication. The live application
was then restored with the EV74 BGR preset and 2 Hz host-volume publication.
