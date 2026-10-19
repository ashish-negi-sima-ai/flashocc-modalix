Neat model benchmarks on Modalix, 2026-10-15
==========================================

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
