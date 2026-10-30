# Ready-to-run FlashOCC demo

This directory contains the selected application assets in ordinary Git: the
sequential QAT camera model, compatible QAT BEV model, six 20 FPS H.264 clips,
40 matching geometry files, and the prebuilt ARM64 application. No separate
model or dataset download, compiler, Python ML environment or submodule checkout
is needed to run this demo. Total payload size is approximately 150 MiB.

`manifest.json` pins every payload's SHA-256 and size, runtime versions and data
provenance. The prebuilt executable is unchanged from the validated benchmark;
its application source hashes are recorded in the manifest. Rebuilding is
optional. Models are PTQ on QAT-trained weights; the BEV package includes the
previously validated shape-metadata adapter for Neat 0.5.0.

## Prerequisites

- Modalix with compatible Neat runtime/plugins, OpenCV 4.6 shared libraries and
  running AppComplex. Validated with Neat 0.5.0 and MLA at 1 GHz; exact versions
  and the Neat runtime hash are in the manifest. These system libraries are not
  bundled with the application.
- A Neat SDK container with running Insight, Python 3.11+ and `curl`.
- Passwordless SSH to the DevKit, and a shared workspace visible to the SDK and
  DevKit at the same absolute path. Clone anywhere within that shared workspace;
  there is no required repository folder name.
- Six unused Insight source slots, indices 1–6. The helper also accepts existing
  assignments to these same demo filenames, and rejects other assignments.

## Run from a fresh clone

Run these commands inside the SDK container. Replace the device and SDK host
addresses for your network. The SDK host must be reachable from Modalix.

```bash
git clone https://github.com/ashish-negi-sima-ai/flashocc-modalix.git
cd flashocc-modalix
python3 scripts/verify_demo.py

export FLASHOCC_DEVICE=sima@10.42.0.102
export FLASHOCC_INSIGHT_API=https://10.42.0.1:9900

# Check Insight and its published port mappings.
neat --json

# Use rtsp.tcp hostPortStart and videoUDP hostPortStart from that output.
python3 scripts/run_insight_demo.py start \
  --rtsp-port 8554 --video-port 9000 \
  --output-dir artifacts/live

tail -f artifacts/live/live.log
```

This selects six sequential MLA camera calls, EV74 BGR normalization, QAT BEV,
eight ARM pooling/argmax threads, 2 Hz occupancy snapshots, rendering off and
the logit validity scan off. The fastest comparable run measured approximately
12.1 six-camera updates per second. Timing varies with the runtime and device.
The 40 frames replay at 20 FPS and loop every two seconds; this is accelerated
replay of a scene originally captured at 2 Hz.

The launcher reads `inputs/config.json`, resolves geometry relative to that
file, resolves model paths relative to the repository, and substitutes the
Insight hostname and mapped ports. It writes the resulting absolute-path
configuration to `artifacts/live/launch-config.json`. Tracked inputs remain
unchanged. Pass `--insight-host HOST` if Modalix needs a different destination
than the API hostname. Existing configurations with absolute paths and explicit
URLs continue to work.

## Enable Insight visualization

Stop the existing run, then enable the mosaic:

```bash
python3 scripts/run_insight_demo.py stop --output-dir artifacts/live
python3 scripts/run_insight_demo.py start \
  --rtsp-port 8554 --video-port 9000 \
  --visualization true --output-dir artifacts/live
```

The source-control helper prints the actual viewer URL returned by Insight.
Visualization measured approximately 8.3 FPS in the earlier 12 FPS-input test;
it adds rendering and video delivery work. `--visualization false` selects the
fastest mode again, and `--check-finite-logits true` enables the optional validity
scan. All modes retain configurable execution overrides via `--execution JSON`.

The bundled workflow was smoke-tested on 2026-10-30 from a checkout at a
different workspace path, using only tracked files: 64 updates without rendering
and 32 with the Insight mosaic, both with zero invalid markers and normal exit.
Occupancy snapshots were validated and the generated mosaic inspected. These
short runs measured 11.05 and 8.01 FPS respectively; they are not the longer
benchmark's 12.1 FPS result. Both applications and all six sources were stopped
after testing. [Validation evidence](../reports/demo/validation-20261030.json).

For interactive 3D, in a native host terminal outside Docker:

```bash
cd /path/to/the/same/shared/flashocc-modalix
python3 scripts/run_host_3d_viewer.py start --source artifacts/live/latest.occ
```

The host viewer prints its URL. It can display the 2 Hz snapshots even when
the Insight mosaic is disabled. Browser rendering is separate from inference.

## Status and stop

From the SDK container, with the same environment variables:

```bash
python3 scripts/run_insight_demo.py status --output-dir artifacts/live
python3 scripts/run_insight_demo.py stop --output-dir artifacts/live
```

If started, stop the host viewer separately in the native host terminal:

```bash
python3 scripts/run_host_3d_viewer.py stop
```

## Scope and rebuilding

The source data is nuScenes mini `scene-0103`. The clips contain resized/cropped
camera images plus synchronization marker footers, which are excluded from
model input. `geometry/*.geom` contains matched depth-to-BEV ranks and camera
rays in FOCCGEO2 format. Outputs are predictions, not ground truth. Upstream
FlashOCC and nuScenes provenance remain recorded; this directory does not
relicense upstream models or dataset content.

This is the runnable demo, not the full 32-input labeled accuracy comparison.
That separate comparison still uses the external preserved asset bundle.

To rebuild the executable in the ARM64 SDK:

```bash
cmake -S app -B build/modalix \
  -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/modalix --target flashocc_insight -j3
python3 scripts/run_insight_demo.py start \
  --binary build/modalix/flashocc_insight \
  --rtsp-port 8554 --video-port 9000 --output-dir artifacts/live
```
