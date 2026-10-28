# QAT packed-camera full-app benchmark

Measured on Modalix on 2026-10-28. **Sequential QAT + EV74 remains the fastest
supported app route: 8.21 FPS.** Among the new packed models, three cameras per
call gave the fastest observed result, 5.29 FPS. The packed results are close;
this single run does not establish a reliable difference between 5.29 and
5.25 FPS.

| Camera mode | Cameras per call | Calls per update | Image normalization | Camera runtime / update | Mean app processing | Measured app FPS |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Sequential QAT | 1 | 6 | EV74 | 31.84 ms | 121.52 ms | **8.21** |
| Sequential QAT | 1 | 6 | ARM FP32 | 27.68 ms | 197.09 ms | 5.06 |
| QAT packed 2 | 2 | 3 | ARM FP32 | 20.09 ms | 190.87 ms | 5.23 |
| QAT packed 3 | 3 | 2 | ARM FP32 | **18.70 ms** | **188.63 ms** | **5.29** |
| QAT packed 6 | 6 | 1 | ARM FP32 | 18.82 ms | 190.01 ms | 5.25 |

Every update processes all six cameras. Packed graphs have model batch size
one and 32-pixel guards between images; this is spatial packing. The supplied
archives are under
`artifacts/compile_with_qat_better_accuracy/compile-qat-packed{2,3,6}/`.

## Why packing does not win overall

Packing reduces camera-call overhead: compared with sequential ARM, the camera
stage drops from 27.68 ms to about 19 ms. However, the current packed app route
normalizes images on ARM. Image preparation and model-input copies take about
83 ms, compared with 5 ms for the sequential EV74 route. This dominates the
camera-stage savings. Sequential EV74 delivers 55% more updates per second than
the fastest observed packed route.

Average milliseconds per six-camera update:

| Component | Sequential EV74 | Sequential ARM | Packed 2 | Packed 3 | Packed 6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image preparation + camera input copies/packing | 5.05 | 81.68 | 83.22 | 83.04 | 84.01 |
| Camera calls, including output unpacking | 31.84 | 27.68 | 20.09 | 18.70 | 18.82 |
| ARM pooling plan + NEON pooling | 11.43 | 11.48 | 11.56 | 11.42 | 11.47 |
| BEV input copy + inference | 26.73 | 26.66 | 27.03 | 26.63 | 26.83 |
| ARM argmax | 4.59 | 4.43 | 4.49 | 4.50 | 4.46 |
| Overlay rendering | 32.04 | 31.94 | 31.78 | 31.74 | 31.82 |
| Snapshots + other application work | 9.83 | 13.22 | 12.69 | 12.59 | 12.60 |
| **Total processing** | **121.52** | **197.09** | **190.87** | **188.63** | **190.01** |

Camera and BEV timings include Neat runtime work and tensor conversion; they are
not pure MLA execution times. EV74 camera normalization is included in camera
runtime, while ARM normalization is included in image preparation. Nested
packing and unpacking timers are not counted twice. These measurements do not
measure capture-to-display latency.

## Protocol and checks

- Same six 12 FPS RTSP streams: 40-frame `scene-0103` replay, with pixel hashes
  checked after retimestamping.
- Same compatible QAT BEV archive, 8 pooling threads, 8 argmax threads and
  6 rendering threads.
- Overlays enabled, occupancy snapshots at 2 Hz, full-logit validity scan
  disabled for timing.
- 256 updates per mode; discard 64 warmup updates, average the remaining 192.
  App FPS is 191 completion intervals divided by their elapsed wall time.
- One run per mode, in order: sequential EV74, packed 2, packed 3, packed 6,
  sequential ARM. Installed Neat 0.5.0; exact runtime and executable identities
  are in the JSON report.
- All five runs completed successfully with zero invalid camera markers.
  The live Insight browser probes passed with zero dropped browser frames.
- Before timing, all three packed models and sequential ARM passed eight
  normalized replay groups with finite BEV logits and valid semantic labels.
  Packed-versus-sequential QAT voxel-label agreement averaged 95.60%, 96.05%
  and 95.66% for packs of 2, 3 and 6. These are agreement checks, not GT accuracy.
  The earlier simulator mIoU results are separate; hardware GT accuracy was not
  re-evaluated here.

The packed routes in this initial comparison use A65 input/output conversion.
The app then rejected packed EV74 image preprocessing. A later
[packed-3 EV74 follow-up](PACKED3_EV74.md) adds and measures a separate per-camera
normalization/quantization route. The SDK `dk` runner encountered an NFS executable
`chmod` permission error, so the tests used direct SSH to the same shared
workspace. No compiler or Neat runtime patch was applied.

## Evidence and reproduction

[Summary, settings and model hashes](modalix-app-20261028.json) ·
[All 256 timing records per mode](modalix-app-20261028-measurements.json).

Local app logs, label outputs, previews and browser checks are under
`artifacts/qat-packed-modalix-20261028/`. These generated files and model
archives are ignored by Git. The three supplied packed archives are additional
assets; they are not included in the older `flashocc_20261027-assets.tar.gz`
bundle. Their SHA-256 identities are pinned in the JSON report.

To repeat, restore the reference input assets and install the supplied packed
archives at the paths recorded in the report. Build the app using the root
README. Configure `FLASHOCC_DEVICE`, `FLASHOCC_INSIGHT_API` and
`FLASHOCC_BROWSER_PYTHON` (an interpreter with Playwright/Chromium). From the
repository root, use a fresh output directory:

```bash
python3 - <<'PY'
import json, os, subprocess, sys
from pathlib import Path

report = json.loads(Path('reports/qat-packed/modalix-app-20261028.json').read_text())
output = Path('artifacts/qat-packed-repeat').resolve()
output.mkdir(exist_ok=False)
for name in report['protocol']['run_order']:
    settings = output / (name + '.json')
    settings.write_text(json.dumps(report['variants'][name]['execution'], indent=2))
    subprocess.run([
        sys.executable, 'scripts/benchmark_insight_fps.py',
        '--device', os.environ['FLASHOCC_DEVICE'],
        '--api', os.environ['FLASHOCC_INSIGHT_API'],
        '--browser-python', os.environ['FLASHOCC_BROWSER_PYTHON'],
        '--base', 'artifacts/insight-demo/config.json',
        '--rates', '12', '--frames', '256', '--warmup', '64', '--volume-fps', '2',
        '--binary', 'build/modalix/flashocc_insight',
        '--execution', str(settings), '--inputs-dir', str(output / 'inputs'),
        '--output-dir', str(output / 'live' / name),
    ], check=True)
PY
```

The benchmark controller starts and stops the six FlashOCC RTSP sources for
each run; the finite app process and browser probe exit automatically. The
test left the app stopped. Existing execution defaults, including the enabled
logit check in the presets, are unchanged.
