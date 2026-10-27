Camera batch-size-six experiment
================================

This experiment recompiles the existing BF16 camera model with requested batch
size six. It reuses the saved quantized model from `artifacts/compile-real/camera`;
there is no retraining, requantization, RGB conversion, or new calibration.
The original batch-one archive remains in place. The BEV model stays at batch
one because it consumes one pooled representation for the six cameras.

The intended tensor interface is normalized FP32 NHWC `[6,256,704,3]`, yielding
depth `[6,16,44,44]` and features `[6,16,44,64]`. Cameras remain in the original
six-camera order. This is the camera dimension flattened into a normal batch,
not a 3D convolution over cameras.

Compilation and validation
--------------------------

```bash
activate-model-compiler
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 SIMA_MLA_SIM_PARALLEL=4 \
  python scripts/compile_camera_batch.py --batch 6
python scripts/prepare_batch_camera_reference.py

cmake -S app -B artifacts/build-app-batch -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/build-app-batch --target check_camera_batch -j3

# Build the isolated runtime workaround described below (SDK host).
python3 scripts/build_batch_neat.py

# SDK host: pause only the owned live app, run device checks, then restore its command.
python3 scripts/run_camera_batch_checks.py --samples 300 --dump-boundaries \
  --runtime-library-dir artifacts/build-neat-batch
python scripts/check_batch_boundaries.py
python scripts/report_camera_batch.py
```

The compiler's requested and actual batch sizes must be checked in the output
archive. The benchmark also checks Neat's `compiled_batch_size()` so a smaller
resolved batch cannot silently be reported as batch six.

Accuracy is evaluated on eight saved nuScenes-mini six-camera holdout groups,
48 images total. All normalized replay inputs are checked against the inputs
used to generate the FP32 camera references. Original batch-one ONNX inference
provides independent per-camera depth and feature references. The hardware
camera outputs then go through ARM NEON pooling and the existing BF16/INT8-
Softplus BEV model. Final labels are compared with saved full FP32 pipeline
labels. They are also compared directly between batch-one and batch-six
hardware executions. These measure numerical conversion fidelity, not
ground-truth occupancy accuracy.

Measured results on Modalix
---------------------------

Compilation produced an actual batch of six, confirmed by the MPK manifest and
Neat. Hardware execution completed on all eight holdout groups. **This batch-six
candidate is not accepted for the live application because numerical fidelity
degrades substantially.** The camera weights and activation configuration are
the saved BF16 configuration; the experiment did not apply INT8 quantization.

| Measurement | Batch 1, six sequential calls | Batch 6, one call |
|---|---:|---:|
| Neat latency per call | 7.13 ms | 32.34 ms |
| Comparable latency for six cameras | 42.79 ms | 32.34 ms |
| Asynchronous logical image throughput | 214.82 images/s | 249.14 images/s |
| Asynchronous six-camera equivalent throughput | 35.80 groups/s | 41.52 groups/s |
| Camera depth relative RMSE vs FP32 | 1.46–2.15% | 17.26–19.57% |
| Camera feature relative RMSE vs FP32 | 1.70–2.05% | 20.56–22.71% |
| Final occupancy-label agreement vs full FP32 | 98.92–99.40% | 92.28–95.39% |
| Agreement on union of occupied voxels | 97.57–99.08% | 85.16–93.15% |

The six-camera latency improves **1.32×** (24.4% lower), and asynchronous image
throughput improves **16.0%**. Ground-truth accuracy and full-application batch-six
FPS have not been measured. The occupancy comparison includes the same ARM pooler
and batch-one BEV model in both variants.

Runtime compatibility
---------------------

The installed Neat runtime fails before batch-six output conversion, reporting
`required_bytes=2433024 actual_bytes=946176` for the first logical output.
The fused `detesscast` route places the full batched transport size into a
per-frame descriptor, which the kernel multiplies by six again. The isolated
[patch](../../app/patches/neat-detesscast-batch.patch) uses per-frame kernel descriptors
while retaining complete batch sizes and offsets in routing/publication metadata.

Both benchmark columns above use the same patched Neat core library, selected
only for the test processes through `LD_LIBRARY_PATH`. The installed Neat core
and runtime plugins are unchanged. The live app is restarted without this override.
`artifacts/compile-batch6/test-runtime.json` records the tested library hash;
`stock-runtime/` preserves the original failure and stock batch-one measurements.

The public-API C++ checker is also the execution regression: the stock runtime
fails to build this batch-six route, while the isolated library completes the
48-image check. Boundary captures and the independent NumPy decoder check that
the output conversion matches the raw MLA BF16 tiles bit for bit.
The captured input transport prefixes also match at the MPK's physical batch
stride. Public Tensor mappings omit some tile-padding bytes, so this input
capture is explicitly marked incomplete in the boundary report. A wider Neat
regression suite and upstream submission of this runtime patch are outside this
experiment. Follow-up tests below reproduce the degradation in compiled MLA
execution. Source inspection below identifies a pooling-padding rewrite in the
compiler's batch segmentation pass; this is not quantization error.

Follow-up correctness diagnosis
-------------------------------

The direct-MLA follow-up uses the installed, unpatched Neat library, an
independent NumPy FP32-to-BF16 input packer, `model.inference()` alone, and an
independent output decoder. Both Neat input and output CVU adapters are bypassed.
On the first six-camera holdout group, both raw batch-six output heads match the
patched full camera route bit for bit. The batch-one camera-zero control also
matches. Therefore the isolated `detesscast` patch does not cause the discrepancy
for this group.

Additional controls:

* Forcing ARM/A65 input conversion instead of EV74 gives bit-identical depth,
  features and occupancy labels on all eight groups for both batch sizes.
* Zeroing cameras 1–5 leaves camera 0's two outputs unchanged.
* The five identical zero-input cameras produce identical outputs.
* Regenerating the batch-six ELF produces the same SHA256 as the tested archive's
  ELF: `8988a311954e8aa9c3dc42a310891962695d213c6e6a204b557b66aa02da54ab`.

The simulator check uses this exact ELF, independently packed real inputs, and
checks 811,008 defined output bytes against direct hardware output. It covers
all feature bytes and the first 32 depth channels, excluding channel padding.
The simulator completed with exit code zero: all 811,008 checked bytes match the
hardware result exactly. It simulated 2,560,737 instructions and 23,825,962 clock
cycles in 428.2 seconds. This reproduces the discrepancy in compiled MLA execution
with Neat input/output adapters and the isolated patch excluded. It also validates
the frame-major to tensor-major unpacking on the checked bytes.

The result is recorded in `artifacts/compile-batch6/diagnostics/diagnosis.json`.
The subsequent source inspection below identifies the incorrect operation.

Follow-up tools are `scripts/prepare_camera_transport.py`,
`app/check_camera_transport.cpp`, and `scripts/prepare_batch_msim.py`.
`artifacts/compile-batch6/diagnostics/` preserves the controls and simulator log.

Compiler source diagnosis
-------------------------

Inspection of `/workspace/BitBucket/sima_src/n2a_compiler` (HEAD `17f16e4b`)
found that `LargeTensorLayer.process_batch()` constructs full spatial `FMSlice`
objects whose padding defaults to zero. `LargeTensorPool._create_vertex()` then
recomputes pooling padding using those zero values, even though only the batch
dimension is being split. The spatial pooling operation must remain unchanged.

Instrumenting the **installed compiler**, without modifying its files, confirms
this happens to `MLA_0/max_pool2d_1` in all six FlashOCC camera segments:

| Property | Original graph | Batch-six rewrite |
|---|---|---|
| Pool input H×W×C | 128×352×64 | 128×352×64 |
| Kernel / stride | 3×3 / 2×2 | 3×3 / 2×2 |
| Top, bottom, left, right padding | 1, 1, 1, 1 | 0, 1, 0, 1 |

The output dimensions remain the same, but pooling windows shift by one feature
pixel vertically and horizontally. This changes the features entering the rest
of the camera network for every camera.

An independent causality check changes **only** the original FP32 ONNX MaxPool's
padding to the compiler's observed values. Across all eight groups / 48 images,
the existing batch-six hardware outputs agree with this altered FP32 network at
the ordinary BF16 error level:

| Hardware comparison, relative RMSE | Depth | Features |
|---|---:|---:|
| Against original FP32 network | 17.26–19.57% | 20.56–22.71% |
| Against FP32 with shifted pooling | 1.40–2.09% | 1.67–2.08% |

This identifies the pooling-padding rewrite as the cause of the large error.
The required compiler fix is to preserve original pooling parameters for a pure
batch split, while retaining spatial-padding recomputation for spatial splits.
A corrected compiler ELF has not yet been built or validated on hardware.
The compiler checkout, installed compiler and live application's models remain
unchanged.

Reproduce the inspection and causality check in the compiler environment:

```bash
python scripts/inspect_batch_compiler.py --batch 6 \
  --output artifacts/compile-batch6/diagnostics/compiler-source
python scripts/check_batch_pool_padding.py
```

The first command stops after graph planning, before MLA code generation.
Reports, graph snapshots and logs are under `diagnostics/compiler-source*`; the FP32
causality measurements are in `compiler-source/padding-causality.json`.

Benchmark interpretation
------------------------

`pyneat.Model.benchmark()` runs 50 warmup batches before each measurement
window. Each measured window uses 300 batches by default. Latency is the mean
single-flight model-route latency for one batch. Throughput is independently
measured with asynchronous submission and reports logical images per second;
divide by six to obtain batch-six groups per second.

For sequential batch-one camera execution, the comparable six-camera latency
is six times the measured batch-one latency. Neither measurement includes ARM
image normalization, CPU pooling, BEV inference, raycasting or browser delivery.
They include the required model-boundary casts, tessellation and output
conversion. The synthetic benchmark does not measure full-application FPS.

Alternative: guarded spatial camera packing
-------------------------------------------

The supplied `artifacts/model_prepared.onnx` uses a different six-camera strategy.
Its input is `[1,6,3,396,704]`, interpreted as scene, camera, channel, height,
width. It crops rows 140:396, transposes and reshapes to expose the cameras,
then concatenates them along image width with five 32-pixel zero guards.
The convolutional network receives `[1,3,256,4384]`: its batch stays one.

Eighteen `spatial_guard_*` multiplications reset the guard regions to zero
throughout the network. This prevents intermediate activations from filling
the gaps and changing neighboring camera boundary features. The raw head is
`[1,123,16,274]`, with 44 valid columns per camera and two guard columns between
adjacent cameras. Camera `i` occupies columns `46*i : 46*i+44`.
The published output is NHWC `[1,16,274,123]` and retains those guards.

`scripts/inspect_prepared_camera.py` derives a single-camera FP32 reference from
the supplied model using the same weights and operations, removes the spatial
guards, and adjusts its fixed FPN resize width. It compares the raw head before
the final source mask and sigmoid. Across all eight saved groups / 48 camera
images, valid packed output regions are bit-identical to separate per-camera
execution (maximum absolute error and relative RMSE both zero).
The 256-row normalized test images are placed in input rows 140:396, with the
cropped-away rows set to zero. This is a graph equivalence check; the colleague's
training preprocessing and ground-truth accuracy have not been established.

The supplied model is a different checkpoint: none of its 57 convolution weight
tensors equal ours. Its head has 59 sigmoid depth channels plus 64 feature
channels, whereas ours has 44 softmax depth channels plus 64 feature channels.
Its final `source_mask` additionally zeros raw outputs at the first spatial
pixel, before depth sigmoid. It cannot directly replace our camera model in
the existing application.

This strategy should avoid the two identified batch-six-specific paths by
keeping batch one, but compiler lowering and hardware behavior remain untested.
All operator types pass the local INT8 and BF16 support audits; shape constraints
still need importer/compiler validation. No MLA compilation, quantization,
device test or performance measurement was performed for this supplied model.
Adapting the strategy to our checkpoint must retain our weights, normalization,
44-channel depth softmax, camera ordering and pooler output contract.

Reproduction and reports:

```bash
activate-model-compiler
python scripts/inspect_prepared_camera.py
```

Results and operator audits are in `artifacts/model-prepared-analysis/`.

Guarded packing applied to our checkpoint
----------------------------------------

`scripts/prepare_packed_camera.py` now applies the technique directly to our
existing `artifacts/onnx-modalix/camera.onnx`. It produces two variants:

| ONNX file under `artifacts/onnx-packed/` | FP32 input contract |
|---|---|
| `camera.onnx` | `[1,6,3,256,704]`: scene, camera, channel, height, width |
| `camera_spatial.onnx` | `[1,3,256,4384]`: one already-packed NCHW image |

Both use network batch one, five 32-pixel input guards, and 18 intermediate
guard masks. All 57 original convolution weight tensors, bias tensors and
convolution parameters are checked byte for byte / structurally against the
original model. The original 44-channel depth softmax is retained; no colleague
weights, sigmoid, source mask or extra image crop is copied. Inputs use exactly
the same normalization and channel order as our original 256×704 camera input.
Only the static FPN resize width changes, from 44 to 274.

Both models retain the two original output names:

| Output | Packed ONNX NCHW | After unpacking NCHW | After unpacking SDK NHWC |
|---|---|---|---|
| `depth` | `[1,44,16,274]` | `[6,44,16,44]` | `[6,16,44,44]` |
| `features` | `[1,64,16,274]` | `[6,64,16,44]` | `[6,16,44,64]` |

Valid output column ranges are `[0:44]`, `[46:90]`, `[92:136]`, `[138:182]`,
`[184:228]`, and `[230:274]`. The two columns between cameras must be discarded;
the published gap outputs need not be zero. `unpack_output()` in the exporter
handles either NCHW or NHWC and returns contiguous per-camera tensors in the
original camera order. `pack_images()` prepares the 4D variant's input with
zero guards. The five-dimensional variant inserts the guards internally.

```python
from prepare_packed_camera import pack_images, unpack_output

# images: existing normalized float32 [6,3,256,704] camera tensors
scene_input = images[None]                # camera.onnx
spatial_input = pack_images(images)      # camera_spatial.onnx
# After ONNX inference, remove output guards before BEV pooling:
depth = unpack_output(packed_depth)      # [6,44,16,44]
features = unpack_output(packed_features)  # [6,64,16,44]
# For SDK NHWC buffers, use unpack_output(value, layout='NHWC').
```

`scripts/validate_packed_camera.py` ran both variants on eight saved holdout
groups / 48 images and compared both outputs with six separate runs of our
original FP32 model. **All depth and feature outputs match bit for bit** after
unpacking. The two packed variants also match each other bit for bit. The check
verifies depth normalization, NCHW/NHWC unpacking, and that zeroing cameras 1–5
does not change camera 0. An additional index-pattern check verifies contiguous
NCHW/NHWC buffers and correct camera ranges for both channel counts.

Reproduce in the model-compiler environment:

```bash
python scripts/prepare_packed_camera.py
python scripts/validate_packed_camera.py
```

`artifacts/onnx-packed/packing.json` records hashes, shapes, camera ranges and
mask sites; `validation.json` records all numerical checks. Both ONNX files pass
ONNX full checking, strict shape inference, and local INT8/BF16 operator audits.
The original models, compiled archives and live application remain unchanged.
Compile these packed exports with **batch size one**. The Modalix validation
workflow below measures accuracy and performance before application integration.

Modalix validation workflow for the packed model
------------------------------------------------

The device experiment uses `camera_spatial.onnx`, with ARM packing of normalized
FP32 NHWC camera tensors. Its zero input guards are in normalized tensor space;
packing black uint8 images before mean/std normalization would give nonzero
guards and would change the computation. The five-dimensional wrapper is covered
by the FP32 equivalence checks; this device workflow tests the 4D CNN variant.

The precision policy remains BF16 weights and activations. Calibration groups
the original 192 calibration images into 32 six-view tensors; the 48 validation
images are kept separate. The checker unpacks both outputs, runs the existing
ARM NEON pooler and batch-one BEV model, and compares occupancy labels with the
full FP32 pipeline. Neither labels nor numerical fidelity constitute a measured
ground-truth occupancy accuracy.

```bash
activate-model-compiler
python scripts/prepare_packed_calibration.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 SIMA_MLA_SIM_PARALLEL=2 \
  python scripts/compile_smoke.py --stage camera --precision bfloat16 \
  --model-path artifacts/onnx-packed/camera_spatial.onnx \
  --calibration-dir artifacts/real-calibration-packed/camera/calibration \
  --build-dir artifacts/compile-packed

cmake -S app -B artifacts/build-app-packed \
  -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/build-app-packed --target check_camera_batch -j3
python3 scripts/run_camera_batch_checks.py \
  --packed-model artifacts/compile-packed/camera/flashocc_m0_camera_mpk.tar.gz \
  --checker artifacts/build-app-packed/check_camera_batch \
  --results-dir artifacts/compile-packed/device --variants batch1 --samples 300
python3 scripts/run_camera_batch_checks.py \
  --packed-model artifacts/compile-packed/camera/flashocc_m0_camera_mpk.tar.gz \
  --checker artifacts/build-app-packed/check_camera_batch \
  --results-dir artifacts/compile-packed/device --variants packed --accuracy-only \
  --execution app/execution-packed-check.json
ssh sima@10.42.0.73 \
  '/home/sima/pyneat/bin/python /workspace/GitHub/occupancy-prediction/scripts/benchmark_models.py camera \
  --model /workspace/GitHub/occupancy-prediction/artifacts/compile-packed/camera/flashocc_m0_camera_mpk.tar.gz \
  --input-width 4384 --cameras-per-input 6 --expected-batch 1 --pre-target A65 \
  --samples 300 --output-dir /workspace/GitHub/occupancy-prediction/artifacts/compile-packed/device/packed/benchmark'
python scripts/report_packed_camera.py
```

The comparison uses the installed Neat runtime without the earlier batch-six
patch. It saves a fresh batch-one baseline, raw camera outputs, occupancy labels,
and Neat benchmark reports under `artifacts/compile-packed/device/`.
`comparison.json` summarizes the results; `provenance.json` records model and
runtime identity. The runner restores the owned live application if it was active
before testing. This experiment does not deploy a new live application route.

Measured packed-model results on Modalix
---------------------------------------

The 4D packed BF16 model compiled successfully as one MLA stage, with compiled
batch one. Compiler spatial/channel splitting produces 407 internal layers, all
batch one. It ran successfully with the original installed Neat core
(`4b1496cbd8bc6a30592b67e1d49abc2e2ed02c281d2bb690cd87a48a7dbf88e1`),
without the isolated batch-six runtime patch or any compiler source changes.
The archive SHA256 is
`dac89da796ba559a3877fe102fc25e2c9b51d2b08bb8e77cce4b997c97b29a94`.

On all eight holdout groups / 48 images, **unpacked hardware depth, features,
and final occupancy labels are bit-identical to the original batch-one
hardware pipeline**. This holds for both AUTO and forced-A65 input conversion.
The FP32 fidelity therefore remains exactly the original baseline's:

| Measurement | Six separate BF16 calls | Packed BF16, A65 input conversion |
|---|---:|---:|
| Depth relative RMSE vs FP32 | 1.46–2.15% | 1.46–2.15% |
| Feature relative RMSE vs FP32 | 1.70–2.05% | 1.70–2.05% |
| Occupancy-label agreement vs full FP32 | 98.92–99.40% | 98.92–99.40% |
| Neat latency per six-camera group | 42.91 ms | 45.20 ms |
| Neat asynchronous six-camera throughput | 36.01 groups/s | 27.78 groups/s |

The Neat measurements use 50 warmups and 300 samples per window. Batch-one
six-camera latency is six times its 7.152 ms single-call latency; its asynchronous
216.07 images/s is divided by six for the group-equivalent throughput. Packed
batch one contains all six camera views, so its 27.78 logical inputs/s directly
represents six-camera groups/s. These are camera-route measurements, not complete
application FPS. They include tensor conversion/tessellation and output
detessellation/casting, and exclude ARM camera packing/output unpacking, pooling,
BEV, RTSP and visualization. The replay checker separately records input
preparation and output-unpacking costs.

**Force `camera_pre_target=A65` for this packed candidate.** The AUTO full route
took approximately 4,315 ms per group. Public-API fragment profiling locates
approximately 4,276 ms in input conversion/tessellation alone; adding MLA gives
approximately 4,313 ms. Forcing A65 reduces the complete route to approximately
45 ms without changing any output bytes. The profiling tool applies placement
both to Model routes and explicitly to the standalone preprocess Graph.
The AUTO 300-sample benchmark was stopped after the eight-image-group accuracy
run exposed this cost. The completed 300-sample packed benchmark uses A65.
The exact cause of AUTO's slow conversion path has not been established.

A subsequent control explicitly forced **EV74** input conversion, with output
conversion left AUTO. The public-API stage profiler used one warmup and two
measured calls per fragment on the same real six-camera input:

| Requested input-conversion target | Input conversion/tessellation | Full camera route |
|---|---:|---:|
| AUTO | 4,276.21 ms | 4,314.29 ms |
| EV74 | 4,276.31 ms | 4,314.20 ms |
| A65 | 6.94 ms | 45.49 ms |

Thus explicitly requesting EV74 reproduces the slow behavior seen with AUTO;
the original comparison was not sufficient to establish EV74 performance by
itself. This measures normalized FP32-to-tiled-BF16 conversion for a
`256×4384×3` tensor. The underlying reason for EV74's poor performance on this
shape remains uninvestigated. Reports are `profile/auto.json`,
`profile/ev74.json` and `profile/a65.json`. These short stage measurements are
separate from the completed 300-sample A65 Neat benchmark above.

This packing strategy avoids the two batch-six-specific failures and preserves
baseline numerical fidelity, but this implementation is about **5.3% slower**
in six-camera route latency and about **22.9% lower** in asynchronous throughput.
It does not establish a full-application speed improvement. The live application
was inactive before testing and remains inactive; its model/configuration files
were not replaced.

Additional artifacts:

* `artifacts/compile-packed/device/packed-auto/`: original slow AUTO accuracy outputs.
* `artifacts/compile-packed/profile/`: separate preprocess, preprocess+MLA and full-route timings.
* `artifacts/compile-packed/compiler-diagnostics/`: planned compiler graph, ELF, MLC and simulator check inputs.
* `app/profile_packed_camera.cpp`: public-API stage profiler; accepts archive, output JSON, pre target and post target.

Artifacts
---------

* `artifacts/compile-batch6/camera`: archive, batch metadata, SHA256 provenance and MLA cycle statistics.
* `artifacts/compile-batch6/reference`: FP32 depth/features for the holdout groups.
* `artifacts/compile-batch6/batch1` and `batch6`: hardware accuracy JSON, labels and Neat benchmark reports/logs.
* `artifacts/compile-batch6/comparison.json`: consolidated accuracy and performance comparison.
* `artifacts/compile-batch6/boundaries-comparison.json`: independent boundary verification.
* `artifacts/compile-batch6/live-before`: original live command and logs preserved before testing.

The live application is restored to its original command after testing. This
experiment does not automatically switch its six camera calls to a batched
application route.
