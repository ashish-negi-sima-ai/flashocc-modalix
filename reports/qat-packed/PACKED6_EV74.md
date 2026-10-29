# Packed 6 with EV74 preprocessing

Tested on Modalix `sima@10.42.0.102` on 2026-10-28. The supplied QAT packed-6
camera archive works with the existing per-camera EV74 preprocessing path and
the compatible QAT BEV archive. **Packed 6 + EV74 measured 11.70 FPS with
visualization disabled.** Sequential EV74 remained fastest in this comparison.

## Fresh comparison

All three runs used the same six 20 FPS RTSP clips, unchanged application binary,
Neat 0.5.0 runtime and QAT BEV archive. Each processed 256 six-camera updates;
statistics exclude the first 64 and use the remaining 192. Occupancy snapshots
remained enabled at 2 Hz. Visualization, video output, JPEG preview saving and
the full-logit validity scan were disabled. Clip preparation verified identical
decoded pixels after retimestamping.

| Mode | Camera MLA calls per update | Mean app processing | Measured FPS |
| --- | ---: | ---: | ---: |
| Sequential + EV74 | 6 | **82.23 ms** | **12.08** |
| Packed 3 + EV74 | 2 | 83.74 ms | 11.79 |
| Packed 6 + EV74 | 1 | 84.74 ms | 11.70 |

FPS uses completion intervals within the measured window, rather than the
reciprocal of mean processing time. Processing starts after acquisition of a
synchronized six-camera group; these are not RTSP-to-output latency measurements.
The packed-3 control completed its entire timing window but required termination
during shutdown, as described below. Sequential and packed-6 exited normally.

Average milliseconds per six-camera update:

| Component | Sequential EV74 | Packed-3 EV74 | Packed-6 EV74 |
| --- | ---: | ---: | ---: |
| ARM geometry loading and image tensor preparation | 3.58 | 3.56 | 3.61 |
| Camera input preparation | 0.02 | 16.98 | 17.41 |
| Camera routes and output copies/unpacking | 31.25 | 15.52 | 16.20 |
| ARM pooling plan and NEON pooling | 12.15 | 12.42 | 12.13 |
| BEV input copy and inference route | 27.02 | 26.90 | 27.18 |
| ARM argmax | 4.56 | 4.60 | 4.59 |
| Snapshots and other application work | 3.64 | 3.75 | 3.61 |
| **Total processing** | **82.23** | **83.74** | **84.74** |

Camera input preparation in packed-6 comprises approximately **15.10 ms** for
the six EV74 normalization/quantization requests and runtime handoffs,
**1.13 ms** for ARM packing/gap filling, and **1.17 ms** for input tensor creation,
copies and other preparation. These are subsets of the 17.41 ms input timer.
Sequential preprocessing is inside its camera routes, rather than its input
preparation timer.

The complete camera stage therefore takes **31.27 ms**, **32.50 ms** and
**33.61 ms** for sequential, packed-3 and packed-6 respectively. Packing all six
views reduces the call count, but its wider camera route does not run faster
than two packed-3 calls in this test. None of these route timers isolates pure
MLA execution; see the separate [ELF-only profile](MLA_ONLY.md) for the earlier
sequential/packed-3 hardware comparison.

## Configuration

New preset:
[`configs/experiments/qat-packed6-ev74-bgr.json`](../../configs/experiments/qat-packed6-ev74-bgr.json).
It selects `camera_mode=packed`, `camera_pack_size=6`, `camera_input=ev74-bgr`,
EV74 camera preprocessing and A65 output conversion. The preset retains
visualization and finite-logit checking enabled; the benchmark override disables
them explicitly. Existing presets and application code were not changed.

Six 704×256 images are normalized and quantized individually on EV74. ARM packs
them into one natural INT8 tensor of shape **256×4384×3**, with five 32-pixel
gaps containing quantized zero (**−16**). The camera model executes once, then
its depth/features are unpacked back into the original six-camera order before
ARM pooling. Network batch size remains one.

The generic implementation added for packed-3 already supports this grouping;
no C++ changes, recompilation, archive modifications or runtime patches were
needed. Binary and source hashes match the earlier validated implementation.

## Numerical validation

The hardware pixel checker passed for all six camera positions:

- 3,244,032 camera values checked; maximum error versus CPU reference: one INT8 level.
- All 122,880 gap values were exactly −16.
- Camera ordering and the single packed-input count were checked.

Two decoded six-camera fixtures were also evaluated through camera inference,
ARM pooling, BEV inference and argmax, with finite-logit checking enabled.
Compared with packed-6 ARM preprocessing using the same archives:

| Fixture token prefix | Voxel labels equal | Feature relative RMSE | Logit relative RMSE |
| --- | ---: | ---: | ---: |
| `3e8750` | 97.49% | 14.75% | 4.33% |
| `de7593` | 97.95% | 13.60% | 3.86% |

Mean voxel-label agreement is **97.72%**. EV74 and ARM preprocessing differ in
rounding, so their predictions are not numerically identical. This measures
prediction agreement, not ground-truth accuracy or mIoU. Ground-truth accuracy
of this preprocessing route was not evaluated here.

The live packed-6 run produced a valid `FOCC3D01` snapshot containing 640,000
labels in the range 0–17. All three live runs had zero invalid camera markers.

## Shutdown observations

One packed-6 fixture-validation process produced all tensors and its final
summary, then hung during shutdown and was killed by the 180-second timeout.
A diagnostic retry using the same application binary exited successfully and
produced byte-identical saved outputs. Stop tracing and a temporary signal-stack
helper were enabled only for that retry; timed app runs used neither.

The fresh packed-3 live control also produced all 256 updates and its final
summary, then hung during shutdown. The verified benchmark PID was stopped with
SIGTERM followed by SIGKILL. Its timing records above describe the completed
steady-state window; shutdown time is outside that window. The normal benchmark
helper consequently returned an error for this control. Its metrics were
recomputed from the completed summary using the same reporting function.

**The new packed-6 live run and the sequential live run both exited cleanly.**
These observations do not establish a shutdown root cause or a packing-specific
bug. The intermittent shutdown issue remains unresolved. All test processes and
the six RTSP sources were stopped afterward; AppComplex remained active.

## Evidence and reproduction

[Settings, identities and results](packed6-ev74-20261028.json) ·
[All timing records, including warmup](packed6-ev74-20261028-measurements.json).
Local fixtures, logs and raw summaries are under
`artifacts/qat-packed6-ev74-20261028/`. Model, binary and fixture hashes are
retained in the report. Earlier packed-3 and frozen baseline reports are unchanged.

Run the hardware input checker on Modalix from the shared repository root:

```bash
build/modalix/check_packed_preprocess \
  configs/experiments/qat-packed6-ev74-bgr.json artifacts/packed6-input-check

build/modalix/check_preprocess \
  artifacts/preprocess-validation/fixtures.json \
  configs/experiments/qat-packed6-ev74-bgr.json artifacts/packed6-fixture-check 1
```

Regenerate decoded fixtures with `scripts/prepare_preprocessing_fixtures.py`
if needed. To reproduce the live timing from the SDK host, configure
`FLASHOCC_DEVICE` and `FLASHOCC_INSIGHT_API` as in the main README, then use a
fresh output directory:

```bash
mkdir -p artifacts/packed6-repeat
python3 - <<'PY'
import json
from pathlib import Path
p = Path('configs/experiments/qat-packed6-ev74-bgr.json')
settings = json.loads(p.read_text())
settings.update(visualization=False, check_finite_logits=False)
Path('artifacts/packed6-repeat/execution.json').write_text(json.dumps(settings, indent=2))
PY
env -u FLASHOCC_BROWSER_PYTHON python3 scripts/benchmark_insight_fps.py \
  --device "$FLASHOCC_DEVICE" --api "$FLASHOCC_INSIGHT_API" \
  --base artifacts/insight-demo/config.json --rates 20 --frames 256 --warmup 64 \
  --volume-fps 2 --binary build/modalix/flashocc_insight \
  --execution artifacts/packed6-repeat/execution.json \
  --inputs-dir artifacts/packed6-repeat/inputs \
  --output-dir artifacts/packed6-repeat/live
```
