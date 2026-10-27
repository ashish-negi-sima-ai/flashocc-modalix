# FlashOCC QAT application on Modalix

For the newer controlled comparison on `sima@10.42.0.102`, including all four
ARM/EV74 model combinations and the eight-sample FP32 prediction comparison,
see [QAT versus previous approaches](QAT_COMPARISON.md). Ground-truth evaluation
on the supplied 32 samples measured 31.87% camera-masked mIoU for QAT versus
33.57% for small mixed precision. The preprocessing-agreement numbers below
are from the earlier test and are separate from those ground-truth metrics.

Tested on 2026-10-26 using `sima@10.42.0.73`, the existing ARM64 application
and installed Neat 0.5.0 runtime. Both networks come from
`artifacts/compile_with_qat_better_accuracy/`:

| Network | Supplied archive | SHA-256 |
| --- | --- | --- |
| Camera | `flashocc_m0_camera_mpk.tar.gz` | `a66706506cccb06385c7a9700675dec7d74d3d4a256627cc159d20f7ea050ad0` |
| BEV | `flashocc_m0_bev_mpk.tar.gz` | `f18aec7eda8eb1835f54fd2db9915784f734b5a559dc8fa1fc005ac69f63d7d4` |

The archived compilation script describes W8A8 PTQ applied to QAT-trained
weights after stripping Q/DQ operators. Each network contains one compiled
batch-one MLA stage. These tests use the supplied compiled networks; no weights
were changed or models recompiled.

## Runtime compatibility

The original BEV package fails graph startup on this runtime with
`gstreamer.caps_incompatible`. Its MLA output advertises INT8 `[200,200,288]`,
but the following dequantization stage expects `[1,200,200,288]`.

`scripts/prepare_qat_runtime_archives.py` creates a separate compatible BEV MPK
by removing that leading singleton dimension from the dequantization stage's
`input_shapes` and `output_shapes`. The element count, quantization parameters,
buffer sizes and every other payload are preserved. It verifies the archived
payloads after writing, including the byte-identical ELF. The original package
is retained.

```bash
python3 scripts/prepare_qat_runtime_archives.py \
  --source artifacts/compile_with_qat_better_accuracy/flashocc_m0_bev_mpk.tar.gz \
  --output artifacts/qat-app/compatible/flashocc_m0_bev_mpk.tar.gz
```

The corresponding `.tar.compatibility.json` records both archive hashes, the
two changed fields and the unchanged ELF hash. The camera package works without
this adaptation. `app/execution-qat-ev74-bgr.json` selects the original camera
and compatible BEV packages.

## Processing and accuracy scope

The EV74 configuration uses six sequential camera calls per synchronized
update. VPU decodes the six RTSP streams; ARM converts NV12 to BGR; EV74 prepares
the camera inputs; MLA runs the camera and BEV networks. Output dequantization,
pooling, argmax, raycasting and mosaic rendering use ARM. Model outputs are
converted on A65, following the previous application configuration.

Both ARM-normalized and EV74-preprocessed routes passed full-pipeline execution
on the same two saved six-camera groups, with finite output tensors. Each route
used a warmup on both groups and three measured repeats. EV74 and ARM voxel-label
agreement was **97.419% and 97.923%**; logit relative RMSE was **4.52% and 4.02%**.
These compare preprocessing routes using the same QAT networks. They do not
measure ground-truth occupancy accuracy or confirm the training accuracy claim.
The larger camera-feature differences are recorded in the fixture report; use
the ARM route when evaluating the compiled networks with the existing ARM
preprocessing reference.

Evidence: `artifacts/qat-app/fixtures/comparison.json`, both fixture summaries
and logs, `inspect/contracts.json`, and the compatibility sidecar.

## Full-application FPS

Every output update processes all six cameras. Tests retain occupancy overlays,
1536×864 H.264 output to Insight and 2 Hz volume snapshots. Each run processes
256 groups, excludes 64 warmups, and measures the remaining 192. FPS is computed
from completed updates and wall time, rather than reciprocal median latency.
The input clips are retimestamped copies with identical decoded pixels.

| Networks and preprocessing | Input FPS per camera | Logit scan | Completed updates/s | Insight browser FPS |
| --- | ---: | --- | ---: | ---: |
| Previous BF16 camera + small mixed BEV, EV74 BGR | 12 | Off | 6.515 | 6.495 |
| New QAT camera + QAT BEV, EV74 BGR | 8 | Off | 8.243 | 8.192 |
| New QAT camera + QAT BEV, EV74 BGR | 12 | Off | **8.264** | 8.295 |
| New QAT camera + QAT BEV, EV74 BGR | 12 | On | **6.431** | 6.445 |
| New QAT camera + QAT BEV, ARM FP32 preprocessing | 12 | Off | **5.045** | 5.047 |

The previous-model control was rerun on the same device with the same binary,
12 FPS inputs, visualization and disabled scan. The QAT route improves full-app
throughput by **26.8%** against that control. At 12 FPS input the faster app still
skips 86 older synchronized groups between the first and last measured outputs
(31.0% of incoming groups); each completed prediction uses six matched views.
The 8 FPS run skipped none in its measurement window.

All these runs had zero invalid camera markers. Each browser probe measured a
separate 20-second window, with zero dropped frames and no JavaScript errors.
These are short steady-state measurements, not long-duration stability tests.

| Median stage duration, 12 FPS inputs and scan off | Previous models | QAT models |
| --- | ---: | ---: |
| Six camera calls, including EV74 preprocessing and output copies | 60.43 ms | 31.34 ms |
| BEV call, including runtime, conversions and output copy | 25.78 ms | 23.90 ms |
| Inference compute, including pooling and argmax | 105.61 ms | 75.15 ms |
| Rendering | 32.03 ms | 31.68 ms |
| Complete application processing | 144.22 ms | 113.10 ms |

These are application timers, not pure MLA execution time. Compute includes
the two network rows and other inference work; the table is not additive.
Complete application time includes occasional snapshot publication. The QAT
scan-on run has median application time 146.23 ms and mean logit-scan time
33.76 ms. Mean application time is 120.56 ms without the scan and 155.15 ms with
it, which better reflects the occasional snapshot work than the median.
Background decode, video delivery and browser latency are outside that timer.

Evidence is under `artifacts/qat-app/live-off/{8,12}/`, `live-on/12/`,
`live-baseline-off/12/` and `live-arm-off/12/`: each contains `summary.json`, `app.log`, execution
settings, browser counters and a screenshot. Network package hashes and
compatibility changes are recorded separately as described above.
`artifacts/qat-app/application-comparison.json` combines all five runs and
records both mean and median stage timings.

## Reproduce

Run benchmarks sequentially with other accelerator workloads stopped:

```bash
python3 scripts/benchmark_insight_fps.py \
  --rates 12 --frames 256 --warmup 64 --volume-fps 2 \
  --execution app/execution-qat-ev74-bgr.json \
  --output-dir artifacts/qat-app/live-on
```

The preset enables the full-logit validity scan. For the disabled-scan benchmark,
copy the preset into `artifacts/qat-app/execution-off.json`, set
`check_finite_logits` to `false`, and supply that file with a separate output
directory. For the ARM reference, also set `camera_input` to `arm-fp32` and
`camera_pre_target` to `AUTO`.

To start the live application, after stopping an existing instance:

```bash
python3 scripts/run_insight_demo.py start \
  --config artifacts/fps-inputs/12/config.json \
  --execution app/execution-qat-ev74-bgr.json \
  --output-dir artifacts/qat-app/running \
  --check-finite-logits false
```

Use the same config and output directory with `stop` to stop that instance and
its six owned sources. Pass `--check-finite-logits true` when starting to enable
the scan. Generic application defaults are unchanged.

After benchmarking, the live QAT demo was started using **8 FPS inputs**, the
EV74 preset and `--check-finite-logits false`, with output under
`artifacts/qat-app/running/`. It uses the matching 8 FPS source config, rather
than the 12 FPS config in the capacity-test example above:

```bash
python3 scripts/run_insight_demo.py status \
  --config artifacts/fps-inputs/8/config.json \
  --output-dir artifacts/qat-app/running
python3 scripts/run_insight_demo.py stop \
  --config artifacts/fps-inputs/8/config.json \
  --output-dir artifacts/qat-app/running
```
