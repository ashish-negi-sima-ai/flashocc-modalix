Neat model benchmarks on Modalix
================================

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
