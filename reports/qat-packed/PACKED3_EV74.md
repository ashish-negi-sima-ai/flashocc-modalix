# Packed 3 with EV74 preprocessing

Tested on Modalix on 2026-10-28 using the supplied packed-3 camera archive and
the existing compatible QAT BEV archive. The opt-in implementation works with
unchanged model archives and an unchanged Neat 0.5.0 runtime.

With visualization disabled, **Packed 3 + EV74 measured 11.79 FPS**, versus
6.50 FPS for Packed 3 + ARM and 12.13 FPS for sequential EV74. It substantially
improves on packed ARM, but does not beat sequential EV74 in this test.

## Actual measurements without visualization

Each mode processed 256 six-camera updates from the same six 20 FPS RTSP streams.
The first 64 updates were warmup; statistics use the remaining 192. Occupancy
snapshots remained enabled at 2 Hz, and the full-logit scan was disabled.
Overlay creation, video output and JPEG preview saving were disabled in the app.
These are measured runs, rather than FPS estimates made by subtracting a timer.

| Mode | Camera MLA calls per update | Preparation + compute | App processing without visualization | Measured FPS |
| --- | ---: | ---: | ---: | ---: |
| Sequential + EV74 | 6 | **78.13 ms** | **82.04 ms** | **12.13** |
| Packed 3 + ARM | 2 | 145.29 ms | 152.11 ms | 6.50 |
| Packed 3 + EV74 | 2 | 79.60 ms | 83.65 ms | 11.79 |

`Preparation + compute` includes image preparation and the entire model pipeline
through argmax. App processing also includes snapshot publishing and other loop
work. Measured FPS uses completion intervals, so input waits and work between
timed sections can make it lower than `1000 / mean application_ms`.

Useful average timers for the two packed routes:

| Timer | Packed ARM | Packed EV74 |
| --- | ---: | ---: |
| ARM geometry loading and image preparation | 75.15 ms | 3.47 ms |
| Camera input preparation, including packing | 7.67 ms | 16.66 ms |
| Of which EV74 normalization/quantization and runtime handoffs | 0 | 14.39 ms |
| Of which packing copies and gap filling | 3.66 ms | 1.04 ms |
| Two camera calls and output unpacking | 18.84 ms | 15.55 ms |
| Complete pipeline `compute_ms` | 70.15 ms | 76.13 ms |

The normalization and packing rows are subsets of camera input preparation.
Do not add them to that timer again. `compute_ms` excludes the ARM image
normalization performed before `Pipeline::run`, but includes the new EV74
preprocessing requests inside that function. This explains why packed ARM can
show a smaller on-screen compute number despite processing fewer updates per
second. Compare preparation plus compute when comparing those routes.

Sequential EV74 fuses preprocessing into its six camera routes. Packed EV74
uses six separate preprocessing requests followed by two camera routes. The
extra preprocessing handoffs offset the savings from fewer camera calls.
None of these timers isolates pure EV74 or MLA kernel execution.

## Full Insight check

A separate run used the original 12 FPS source rate with visualization enabled:
Packed 3 + EV74 measured **8.22 FPS**, with mean on-screen compute of 75.85 ms.
The Insight browser measured 8.25 FPS, with zero dropped browser frames or page
errors. Its output was visually inspected. This is approximately the same
throughput as the earlier sequential EV74 run, 8.21 FPS; a single run does not
establish a meaningful difference between them.

All four runs completed 256 updates with zero invalid camera markers. The app
and its six source streams were stopped after testing.

## What changed

The new preset is
[`configs/experiments/qat-packed3-ev74-bgr.json`](../../configs/experiments/qat-packed3-ev74-bgr.json).
It sets `camera_mode=packed`, `camera_pack_size=3`, `camera_input=ev74-bgr` and
`camera_pre_target=EV74`.

1. The existing decoder path produces six cropped 704×256 BGR images.
2. The app queues all six images through an EV74 preprocessing fragment. This
   normalizes and quantizes at single-camera width using the packed archive's
   quantization parameters.
3. ARM packs the natural INT8 outputs into two 256×2176×3 tensors. The 32-pixel
   gaps contain INT8 **−16**, the archive's quantized representation of zero.
4. The packed camera inference and postprocessing fragments run twice. Their
   outputs are unpacked into the original six-camera order before pooling.

The preprocessing-only model object loads the same archive with an explicit
704×256 preprocessing geometry. Its MLA fragment is never executed at that
geometry. The separate camera inference object retains the original packed
geometry. This uses public `Model::preprocess()`, `inference()` and
`postprocess()` APIs; no ELF or archive metadata was edited.

The implementation requires natural INT8 ingress and identical preprocessing
and inference quantization parameters. Packed NV12 and BF16-input archives
are not supported by this new route. The original presets and enabled logit
check defaults are retained.

An initial direct 2176-pixel-wide preprocessing probe failed its pixel-value
sanity check. A manual standalone `Preproc` node also crashed during Neat graph
setup. The selected implementation avoids both paths: it uses model-managed
preprocessing at width 704. Their underlying causes were not established in
this experiment; diagnostics remain under the local results directory.

## Numerical checks and limits

The hardware input checker tested all six camera positions with synthetic
patterns spanning all 256 pixel intensities per channel. All 3,244,032 camera
values were within one INT8 quantization level of the CPU reference, and all
98,304 gap values were exactly the required zero point. FIFO camera order was
checked with distinct patterns for each camera.

Two decoded six-camera NV12 fixtures were evaluated through the complete
pipeline with finite-logit checking enabled. Compared with packed ARM using
the same archives:

| Fixture token prefix | Voxel labels equal | Feature relative RMSE | Logit relative RMSE |
| --- | ---: | ---: | ---: |
| `3e8750` | 97.40% | 14.28% | 3.89% |
| `de7593` | 97.71% | 13.42% | 3.86% |

Mean label agreement is **97.56%**. EV74 and ARM preprocessing are not
numerically identical; the bounded input rounding difference propagates through
the quantized networks. These are prediction-agreement checks, not GT accuracy
or mIoU measurements. Ground-truth accuracy of this new path remains unmeasured.
The queued and synchronous EV74 versions produced byte-identical saved tensors
and labels on both fixtures.
After the change, eight existing packed ARM replay groups also produced
byte-identical labels to the earlier hardware run, with finite-logit checks
enabled.

## Evidence and reproduction

[Settings, hashes and validation results](packed3-ev74-20261028.json) ·
[Every measured and warmup timing record](packed3-ev74-20261028-measurements.json).
Local logs and selected output tensors are under
`artifacts/qat-packed-ev74-20261028/`.

Build using the main README. The new hardware checker can then be run on
Modalix from the shared repository root:

```bash
build/modalix/test_insight
build/modalix/check_packed_preprocess \
  configs/experiments/qat-packed3-ev74-bgr.json artifacts/packed-ev74-check
```

The two decoded fixtures can be regenerated on the SDK host with
`python3 scripts/prepare_preprocessing_fixtures.py`. On Modalix, compare
`build/modalix/check_preprocess` with the ARM and EV74 packed-3 presets as
described in [execution settings](../../app/EXECUTION.md). Keep fixture hashes
with the comparison; the JSON report pins the exact fixtures used here.

To repeat the application benchmarks, configure `FLASHOCC_DEVICE`,
`FLASHOCC_INSIGHT_API` and `FLASHOCC_BROWSER_PYTHON` as in the main README, then
run this from the SDK repository root. It uses a fresh results directory:

```bash
python3 - <<'PY'
import json, os, subprocess, sys
from pathlib import Path

report = json.loads(Path('reports/qat-packed/packed3-ev74-20261028.json').read_text())
output = Path('artifacts/packed3-ev74-repeat').resolve()
output.mkdir(exist_ok=False)
order = ['packed3-ev74-full', 'sequential-ev74-no-render',
         'packed3-arm-no-render', 'packed3-ev74-no-render']
for name in order:
    variant = report['variants'][name]
    settings = output / (name + '.json')
    settings.write_text(json.dumps(variant['execution'], indent=2))
    command = [sys.executable, 'scripts/benchmark_insight_fps.py',
        '--device', os.environ['FLASHOCC_DEVICE'],
        '--api', os.environ['FLASHOCC_INSIGHT_API'],
        '--base', 'artifacts/insight-demo/config.json',
        '--rates', str(variant['input_fps']), '--frames', '256', '--warmup', '64',
        '--volume-fps', '2', '--binary', 'build/modalix/flashocc_insight',
        '--execution', str(settings), '--inputs-dir', str(output / 'inputs'),
        '--output-dir', str(output / 'live' / name)]
    environment = os.environ.copy()
    environment.pop('FLASHOCC_BROWSER_PYTHON', None)
    if variant['visualization']:
        command += ['--browser-python', os.environ['FLASHOCC_BROWSER_PYTHON']]
    subprocess.run(command, check=True, env=environment)
PY
```

For an interactive Insight run, select the new preset with
`scripts/run_insight_demo.py start --execution configs/experiments/qat-packed3-ev74-bgr.json`.
Pass the source config and output directory as in the main quickstart, and
disable the logit check explicitly when reproducing these performance numbers.
