Neat model benchmarks on Modalix
================================

The [2026-10-27 controlled QAT comparison](QAT_COMPARISON.md) includes fresh
full-application timing breakdowns on `sima@10.42.0.102` and the accuracy-evidence
comparison. QAT/EV74 measured 8.31 FPS versus 6.50 FPS for the previous models;
on the supplied 32 labeled samples QAT scored 31.87% camera-masked mIoU versus
33.57% for small mixed precision and 34.58% for the original FP32 baseline.
QAT-versus-own-FP32 fidelity remains unmeasured.

For the supplied QAT-derived W8A8 camera and BEV packages tested on 2026-10-26,
see [QAT full-application results](QAT_APPLICATION.md). Their complete
RTSP-to-Insight application measured 8.26 FPS with EV74 preprocessing and the
logit scan disabled, versus a freshly rerun 6.51 FPS previous-model control.
These are application measurements, separate from the synthetic Neat model
benchmarks below.

BEV small mixed precision, 2026-10-23
------------------------------------

The supplied `artifacts/compiled_with_better_quantization/bev_small_mixed_mpk.tar.gz`
ran successfully in both the complete inference replay and the six-RTSP-to-Insight
application. It has the existing float32 boundary contracts: input
`[200,200,64]`, output `[200,200,288]`, compiled batch one. No C++ changes or
recompilation were needed. Its archive SHA256 is
`36c4a5fc3cd02834fa7574c44506c7d9794db9202ee3524e932aa9310225a88e`.

Both archives below were measured on the same Modalix with the app and source
streams stopped, using `pyneat.Model.benchmark(300, True)`, AUTO input conversion
and A65 output conversion. Latency and async throughput are separate windows.
These values include the model route's boundary conversions and runtime overhead,
and exclude application tensor copies, camera inference, pooling, argmax,
video and visualization. They are not pure MLA timings or full-app FPS.

| BEV archive | Mean model-route latency | Async pooled inputs/s |
| --- | ---: | ---: |
| Previous BF16 / INT8-Softplus archive | 61.327 ms | 34.66 |
| New `bev_small_mixed_mpk.tar.gz` | 16.677 ms | 155.68 |

The new model reduces route latency by 72.8% and has 4.49x higher asynchronous
throughput in this comparison. Plugin latency tracing was unavailable; zero
power telemetry was also unavailable data.

The full app used the existing single-camera BF16 archive six times per group,
ARM FP32 image preparation, eight ARM NEON pooling workers, A65 model-output
conversion, occupancy overlays, H.264 Insight delivery and 2 Hz 3D snapshots.
At 8 FPS per input stream, 180 groups were processed and the first 60 excluded:

| Full-application measurement | Result |
| --- | ---: |
| Sustained six-camera updates/s | 3.898 |
| Browser delivery, separate 20-second window | 3.897 FPS |
| Median inference compute | 99.63 ms |
| Median application processing | 257.55 ms |
| Median rendering | 31.57 ms |
| Invalid camera markers | 0 across all six cameras |
| Browser dropped frames during measurement | 0 |

The app skips older synchronized groups when input arrives faster than processing;
approximately 51.4% of input groups were skipped in the measured window. All six
views are processed for every output update. Browser JavaScript errors were absent.

A follow-up with `check_finite_logits=false` sustained **4.536 updates/s**
(4.546 browser FPS) using the same models, 8 FPS streams, ARM preprocessing,
overlays and 2 Hz snapshots. This run processed 160 groups, excluded 40 warmups,
and measured 120 groups. The scan timer was exactly zero; browser dropped frames,
JavaScript errors and invalid camera markers were zero during measurement.
Separate on/off replays produced byte-identical occupancy labels on eight
six-camera samples, with the enabled scan taking a median 33.29 ms. The check
remains enabled by default. Follow-up evidence is in
`artifacts/finite-logit-check/`, including `replay-comparison.json` and
`live-off/summary.json`; see [execution settings](EXECUTION.md) for the toggle.

EV74 BGR preprocessing was then tested with the same BF16 camera archive and
small mixed BEV archive. Both EV74 tests processed 180 six-camera groups,
excluded 60 warmups, and retained overlays and 2 Hz 3D snapshots at 8 FPS input:

| Camera route and preprocessing | Logit check enabled: full-app FPS | Logit check disabled: full-app FPS |
| --- | ---: | ---: |
| ARM FP32, earlier runs | 3.898 | 4.536 |
| EV74 BGR, follow-up runs | 5.386 | 6.683 |
| Packed two, ARM FP32, subsequent runs | 3.956 | 4.582 |

The new runs offload normalization and model-input preparation to EV74; ARM
still converts decoded NV12 images to BGR. Median live image/geometry preparation
fell from approximately 74 ms with ARM to approximately 3 ms with EV74 BGR.
Camera-route time includes the EV74 preprocessing work, so compare complete
application throughput rather than treating camera inference timing alone as
a measure of the optimization. The ARM rows reuse earlier measurements rather
than a new simultaneous baseline.

Both EV74 runs had zero invalid markers and JavaScript errors. The browser
quality counters reported one dropped video frame over 20 seconds with the
check enabled and five with it disabled. Browser total-frame rates were 5.397
and 6.696 FPS respectively; these counters include dropped frames. Subtracting
the drop counts gives estimated presented-frame rates of 5.347 and 6.446 FPS.
The table reports completed occupancy groups on Modalix.

On two identical decoded six-camera fixtures, three repeats each, the EV74 BGR
route's labels agreed with the ARM route on **98.261% and 98.751%** of voxels.
Occupied-union semantic agreement was 97.463% and 98.121%; camera feature
relative RMSE was 1.878% and 1.700%. These results compare preprocessing routes
using the same models; they are not agreement against FP32 or ground-truth
accuracy. The models were not recompiled for this experiment.

Use `app/execution-bev-small-mixed-ev74-bgr.json` for this combination. Its logit
check is enabled; override it with `--check-finite-logits false` after stopping
the existing app. Evidence, raw timing summaries, fixture dumps and browser
screenshots are in `artifacts/bev-small-mixed-ev74/`; the combined report is
`results.json`. The application was left running with this EV74 preset and the
logit check enabled.

The lowest-latency existing camera packing configuration (two cameras per call,
three calls per six-camera update) was subsequently tested with the small mixed
BEV archive. Network batch stays one, with a `[1,3,256,1440]` input and a 32-pixel
normalized-zero guard. Its camera archive SHA256 remains
`ca2d3cdab67dd2003f49fa02e034408d48a88dfb48426290d413e35c698e64da`.
The current packed route requires ARM normalization before inserting guards.

Both packed full-app runs processed 180 groups and excluded 60 warmups at
8 FPS input, with overlays and 2 Hz snapshots enabled. With the logit check
disabled, median camera-route time was 44.80 ms versus 59.12 ms for sequential
EV74 preprocessing, but preparation took 74.75 ms versus 3.20 ms. Complete app
throughput was consequently lower: 4.582 versus 6.683 updates/s. With the check
enabled, the corresponding rates were 3.956 versus 5.386 updates/s.

Packed labels were byte-identical to the sequential ARM/small-BEV reference on
eight six-camera samples, two repeats each. Both live tests had zero invalid
camera markers and JavaScript errors. Browser measurements counted zero dropped
frames with the check enabled and three with it disabled over their separate
20-second windows. These are model-equivalence and integration checks, not
ground-truth accuracy measurements.

The tested packed preset is `app/execution-bev-small-mixed-packed2.json`.
Sequential EV74 was restored as the faster full-app route, with the logit check
enabled. Evidence is under `artifacts/bev-small-mixed-packed2/`, including
`results.json`, `replay-comparison.json`, and `live-{on,off}/summary.json`.

In a separate lossless replay of eight six-camera samples, two repeats each,
the median BEV stage including its float32 output copy was 24.98 ms and median
complete inference compute was 96.69 ms. Agreement with the saved FP32 occupancy
labels was **94.07–96.90%** across those eight samples; occupied-union semantic
agreement was **90.71–94.95%**. These are numerical agreement measurements,
not ground-truth accuracy. The speed results do not establish equivalent accuracy.

Output conversion on EV74 was also tested independently for the camera and
small mixed BEV archives. Both routes are supported with these artifacts. The
camera output stage combines BF16 detessellation and FP32 conversion; the small
BEV archive's output stage dequantizes INT8 scores to FP32. This is different
from the older BF16 BEV archive's detessellation/cast path that previously failed
when forced to EV74.

On two identical decoded six-camera fixtures, three measured repeats after a
warmup pass, all four configurations (A65/A65, EV74/A65, A65/EV74, EV74/EV74)
produced byte-identical depth, features, pooled tensors, logits and voxel labels.
Profiles confirmed asynchronous EV74 dispatch for the forced EV74 conversions.
Indicative fixture-profile output stage means were about 0.56 ms per camera
call on A65 versus 0.92 ms on EV74, and about 6.1 ms per BEV call on A65 versus
16.0 ms on EV74. These stage samples include warmup; use the warmed full-app
comparison below for throughput conclusions. Output-stage times include runtime
work and are not hardware-only kernel times.

| Camera output target | BEV output target | Full-app updates/s |
| --- | --- | ---: |
| A65 | A65 | 6.637 |
| EV74 | A65 | 6.560 |
| EV74 | EV74 | 5.955 |

All three fresh live runs used EV74 BGR input preprocessing, sequential camera
calls, the same small mixed BEV archive, 8 FPS streams, overlays, 2 Hz snapshots
and a disabled logit check. Each processed 180 groups and excluded 60 warmups.
All had zero invalid camera markers and JavaScript errors. Separate 20-second
browser windows recorded two, zero and one dropped video frames respectively.
The camera-only difference is small; neither tested EV74 output configuration
improved throughput. Conversion to FP32 still leaves output mapping/copy work
before ARM pooling and class selection, so these choices do not remove all
ARM output handling.

The faster A65/A65 output settings were restored, with EV74 input preprocessing
and the default enabled logit check. Existing `camera_post_target` and
`bev_post_target` settings already expose both choices; no application code or
model recompilation was required. Raw configuration files, logs, fixture dumps,
`fixture-comparison.json` and `live-{baseline,camera-ev74,both-ev74}/summary.json`
are under `artifacts/ev74-output-conversion/`.

Select the new archive explicitly with `app/execution-bev-small-mixed.json`:

```bash
python3 scripts/run_insight_demo.py start \
  --config artifacts/fps-inputs/8/config.json \
  --execution app/execution-bev-small-mixed.json \
  --output-dir artifacts/bev-small-mixed/running

python3 scripts/run_insight_demo.py stop \
  --config artifacts/fps-inputs/8/config.json \
  --output-dir artifacts/bev-small-mixed/running
```

To repeat the isolated model benchmark on Modalix with the live app stopped:

```bash
~/pyneat/bin/python scripts/benchmark_models.py bev --samples 300 \
  --model artifacts/compiled_with_better_quantization/bev_small_mixed_mpk.tar.gz \
  --expected-batch 1 --pre-target AUTO --post-target A65 \
  --output-dir artifacts/bev-small-mixed/benchmark/small
```

Evidence is under `artifacts/bev-small-mixed/`: `benchmark/{small,baseline}/`,
`replay/`, `live/8/` and `live/summary.json`. The browser screenshot is
`live/8/browser.png`. The supplied large and recalibrated archives were not part
of this experiment.

Camera packing comparison, 2026-10-20
-------------------------------------

The app supports one, two or three spatially packed calls per six-camera update,
as well as six single-camera calls. All four variants below were measured using the
installed, unmodified Neat runtime, BF16 archives, A65 input/output conversion,
50 warmups and 300 measured samples per latency and throughput window.

| Mode | Cameras per call | Calls per six-camera update | Mean latency per call | Six-camera route latency | Async six-camera equivalent groups/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single camera | 1 | 6 | 7.691 ms | 46.148 ms | 35.31 |
| Packed two | 2 | 3 | 14.028 ms | 42.085 ms | 34.80 |
| Packed three | 3 | 2 | 23.376 ms | 46.752 ms | 30.11 |
| Packed six | 6 | 1 | 45.218 ms | 45.218 ms | 27.78 |

Six-camera latency is derived as call latency multiplied by call count.
Throughput comes from a separate asynchronous measurement and is normalized
by cameras per input. These are model-route measurements, including required
conversion and runtime overhead; they exclude application packing/unpacking,
pooling, BEV inference, RTSP and visualization. They are not pure MLA time or
full application FPS.

The two-camera variant has the lowest measured six-camera route latency:
approximately 8.8% below six single-camera calls and 6.9% below one six-camera
packed call. Its asynchronous throughput is close to the single-camera model's
six-camera equivalent throughput. These results use the same BF16/A65 settings;
application performance is measured separately.

The three-camera option preserves network batch one. Its 256×2176 input has
two 32-pixel zero guards, with masks inside the model to keep cameras
independent. The shared C++ pipeline restores all six outputs before pooling.
On eight groups (48 cameras), its depth, features and occupancy labels were
byte-identical to the original sequential BF16 pipeline. The FP32 exports also
matched the original FP32 model bit for bit on those images. This is numerical
agreement, not a ground-truth accuracy measurement.

The two-camera variant also preserves network batch one. Its 256×1440 input
contains one 32-pixel guard. Three calls process cameras 0–1, 2–3 and 4–5.
Its FP32 outputs and its hardware depth/features/occupancy labels matched the
respective sequential references bit for bit across all 48 validation images.
The existing BEV network and pooler are shared by every mode.

Use `app/execution-packed3.json`; see [execution settings](EXECUTION.md) for
startup commands. Exact reports and hashes are in
`artifacts/compile-packed3/benchmark/comparison.json`, with hardware evidence in
`artifacts/packed3-app/equivalence.json`.

Use `app/execution-packed2.json` for three calls with two cameras per input.
The combined four-variant report is
`artifacts/compile-packed2/benchmark/comparison.json`, with accuracy evidence in
`artifacts/packed2-app/equivalence.json`. To reproduce its Neat benchmark, change
the command below to `compile-packed2`, `--input-width 1440` and
`--cameras-per-input 2`.

To benchmark the three-camera archive on Modalix with the live app stopped:

```bash
~/pyneat/bin/python scripts/benchmark_models.py camera --samples 300 \
  --model artifacts/compile-packed3/camera/flashocc_m0_camera_mpk.tar.gz \
  --input-width 2176 --cameras-per-input 3 --expected-batch 1 \
  --pre-target A65 --post-target A65 \
  --output-dir artifacts/compile-packed3/benchmark/packed3
```

Original model measurements, 2026-10-15
---------------------------------------

Measured with the deployed camera BF16 and BEV BF16/INT8-Softplus packages,
with the live application stopped to avoid accelerator contention.

| Network | Mean model-route latency | Async throughput |
| --- | ---: | ---: |
| Camera, one image | 7.104 ms | 213.91 images/s |
| BEV/head, one pooled input | 61.389 ms | 34.55 inputs/s |

Each model used `pyneat.Model.benchmark(300, True)`: 50 warmup samples
before each of two separate measurement windows, single-flight latency and
asynchronous throughput. Synthetic tensor inputs use the application's
normalization/resize/color-conversion settings (disabled), with CVU AUTO.
Required casts, tessellation and output conversion are included. Application
input construction, CPU output copies, pooling, argmax, video and visualization
are excluded. Throughput is independently measured; it is not reciprocal latency
or full-application FPS. Six sequential camera calls plus one BEV call give
approximately 104 ms of model-route latency, before application work.

Plugin latency tracing reported `lttng/unavailable`. Power readings were zero
and should be treated as unavailable, not zero device consumption.

Run sequentially on the device while the application is stopped:

```bash
cd /workspace/GitHub/occupancy-prediction
~/pyneat/bin/python scripts/benchmark_models.py camera --samples 300
~/pyneat/bin/python scripts/benchmark_models.py bev --samples 300
```

Reports and full runtime logs are in `artifacts/model-benchmark/`.
The original live application was restarted after both measurements.

Camera model comparison, 2026-10-16
-----------------------------------

Reran the single-image and spatially packed BF16 camera archives using the
installed, unmodified Neat runtime, with the live app and its six sources
stopped. Both models used **A65 input and output conversion**, compiled batch
one, 50 warmups and 300 measured samples in each of the separate latency and
asynchronous throughput windows.

| Measurement | Single-camera model | Packed six-camera model |
| --- | ---: | ---: |
| Camera views per call | 1 | 6 |
| Image height × width per call | 256 × 704 | 256 × 4384 |
| Mean latency per call | 7.805 ms | 45.195 ms |
| Latency for six cameras | 46.827 ms, derived as six calls | 45.195 ms, one measured call |
| Async model calls/s | 212.70 | 27.79 |
| Equivalent camera images/s | 212.70 | 166.71 |
| Equivalent six-camera groups/s | 35.45 | 27.79 |

The packed model has approximately 3.5% lower six-camera route latency with
these conversion settings, and approximately 21.6% lower asynchronous
throughput. This is a synthetic model-route comparison: it includes required
casts, tessellation, output conversion and runtime overhead, and excludes
image normalization, application packing/unpacking, pooling, BEV inference,
RTSP decoding and visualization. It does not measure pure MLA execution or
complete application FPS. Throughput uses pipelined execution and should not
be calculated as reciprocal single-flight latency.

The earlier single-camera runs used AUTO conversion; this comparison explicitly
uses A65 for both models. Plugin tracing remained unavailable, and zero power
readings mean telemetry was unavailable. The packed live app was restored with
its previous execution settings after testing.

Exact JSON results, archive hashes and runtime logs are under
`artifacts/camera-model-benchmark/20261016T111524Z/`.

To reproduce, run on Modalix with the live application stopped:

```bash
~/pyneat/bin/python scripts/benchmark_models.py camera --samples 300 \
  --expected-batch 1 --pre-target A65 --post-target A65 \
  --output-dir artifacts/camera-model-benchmark/single
~/pyneat/bin/python scripts/benchmark_models.py camera --samples 300 \
  --model artifacts/compile-packed/camera/flashocc_m0_camera_mpk.tar.gz \
  --input-width 4384 --cameras-per-input 6 --expected-batch 1 \
  --pre-target A65 --post-target A65 \
  --output-dir artifacts/camera-model-benchmark/packed
```
