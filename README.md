# FlashOCC on Modalix

Six calibrated camera views produce a dense `200 × 200 × 16` semantic occupancy
volume. The C++ app runs the camera and BEV networks on MLA, with ARM NEON pooling,
argmax and occupancy rendering. EV74 camera preprocessing is configurable.
Six RTSP streams feed the [Insight mosaic](app/INSIGHT.md); the
[host 3D viewer](viewer/README.md) displays interactive semantic voxels.

## Demo clips

**Insight — six camera overlays and BEV occupancy slices (10 seconds).**
Click the animated preview to open the MP4.

[![Insight: six camera occupancy overlays and two BEV slices](docs/media/insight.gif)](docs/media/insight.mp4)

**Host 3D grid — live semantic voxels with an orbiting view (10 seconds).**
The viewer uses physical height scale and 0.4 m cells.

[![Host viewer: live 3D semantic occupancy grid](docs/media/host-3d-grid.gif)](docs/media/host-3d-grid.mp4)

**15-second Full HD MP4s (1920×1080):**
[Insight](docs/media/hd-15s/insight.mp4) ·
[Host 3D grid with camera orbit](docs/media/hd-15s/host-3d-grid.mp4).

Both recordings use QAT models with EV74 preprocessing on Modalix and replay
`scene-0103` at 8 FPS. The host receives occupancy snapshots at 2 Hz; browser
rendering speed is separate from application throughput.
[Recording details and reproduction](docs/media/README.md).

## Validated comparison

The preserved 2026-10-27 test uses the same 32 labeled inputs and Occ3D camera
visibility masks for each model pair. Accuracy is model-level evaluation of
normalized inputs, excluding RTSP compression and EV74 image normalization.

| Model pair | Camera-masked mIoU | Masked voxel accuracy |
| --- | ---: | ---: |
| Original FP32 | 34.58% | 89.34% |
| Original BF16-heavy | 34.65% | 89.36% |
| BF16 camera + small mixed BEV | 33.57% | 89.22% |
| QAT camera + QAT BEV | 31.87% | 86.00% |

Full-app performance uses six 12 FPS RTSP streams, overlays, 2 Hz snapshots and
an explicitly disabled logit validity scan. Every output processes all six views.

| Networks / image preprocessing | Mean app processing | App FPS |
| --- | ---: | ---: |
| Previous mixed / ARM | 223.1 ms | 4.48 |
| Previous mixed / EV74 | 153.3 ms | 6.50 |
| QAT / ARM | 198.4 ms | 5.03 |
| QAT / EV74 | 119.9 ms | 8.31 |

QAT + EV74 is the fastest measured route; among the compiled pairs compared
here, the original BF16-heavy pair has the highest measured accuracy. Choose a model pair according
to the application requirement. These are subset results, not a full nuScenes
validation benchmark. [Detailed timings and scope](app/QAT_COMPARISON.md) ·
[Frozen evidence and reproducibility contract](benchmarks/flashocc_20261027/README.md).

[Cleanup audit and independent reproduction](reports/cleanup/README.md).

## Setup

Clone with the pinned upstream FlashOCC submodule:

```bash
git submodule update --init --recursive
cp configs/runtime.example.json configs/runtime.local.json
```

Set `device`, `insight_api` and optionally `insight_url` in that local file.
Alternatively export `FLASHOCC_DEVICE`, `FLASHOCC_INSIGHT_API` and
`FLASHOCC_INSIGHT_URL`. CLI arguments override these defaults. No developer
home directory or fixed device address is required. The host and Modalix must
see the shared workspace at the same path.

Use the SiMa SDK/compiler environment for export/validation and the ARM64 Neat
SDK for application builds. The original comparison used installed Neat 0.5.0;
precompiled model identities are pinned by SHA-256 in [models/manifest.json](models/manifest.json).
Model and dataset payloads belong in external asset storage, not source Git.

Restore the preserved asset bundle obtained from your model storage:

```bash
python3 scripts/comparison_assets.py restore \
  --archive /path/to/flashocc_20261027-assets.tar.gz \
  --asset-root /path/to/restored-assets
python3 scripts/comparison_assets.py verify --asset-root /path/to/restored-assets
```

Keep the adjacent `.tar.gz.json` inventory/checksum file with the bundle.
The local cleanup archive is stored outside the repository under
`/workspace/benchmark-archives/`; a public release download URL has not been
published. A source checkout alone does not contain images, GT or model weights.

## Build and run

Inside the SDK container, from the repository root:

```bash
cmake -S app -B build/modalix \
  -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/modalix -j3

# With reference assets restored at the repository root:
python3 scripts/run_insight_demo.py start \
  --config artifacts/fps-inputs/12/config.json \
  --execution configs/execution/qat-ev74-bgr.json \
  --output-dir artifacts/live \
  --check-finite-logits false

python3 scripts/run_insight_demo.py status \
  --config artifacts/fps-inputs/12/config.json --output-dir artifacts/live
python3 scripts/run_insight_demo.py stop \
  --config artifacts/fps-inputs/12/config.json --output-dir artifacts/live
```

Execution presets use workspace-relative model paths. For an asset bundle
restored elsewhere, use the comparison runner below or generate execution/source
configs with the corresponding absolute asset paths. Preset defaults retain the
logit scan; disable it explicitly when reproducing the FPS table.
[All execution settings](app/EXECUTION.md).

For interactive 3D, in a host terminal outside Docker:

```bash
python3 scripts/run_host_3d_viewer.py start --source artifacts/live/latest.occ
python3 scripts/run_host_3d_viewer.py stop
```

The service prints its URL; pass `--host` for access from another machine.
WebGL rendering runs in the browser, separately from Modalix inference.

## Reproduce both comparisons

Use a fresh results directory. The asset root may be a separately restored
bundle. `--python` must identify an environment with NumPy and ONNX Runtime.
The device/API values can come from the local runtime configuration.

```bash
python3 scripts/reproduce_comparison.py accuracy \
  --asset-root /path/to/restored-assets \
  --results artifacts/comparison-repeat \
  --python /path/to/validation-environment/bin/python

python3 scripts/reproduce_comparison.py performance \
  --asset-root /path/to/restored-assets \
  --results artifacts/comparison-repeat \
  --browser-python /path/to/playwright-environment/bin/python

python3 scripts/reproduce_comparison.py summarize --results artifacts/comparison-repeat
python3 benchmarks/flashocc_20261027/check.py --records-only
```

Accuracy regenerates original FP32 labels and runs all three compiled model
pairs on the same 32 inputs. Performance runs all four ARM/EV74 configurations
sequentially with 256 groups and 64 warmups. It uses a separate retimestamped
input directory and preserves the reference assets. The output includes raw
records and `comparison-report.json`. The optional Playwright interpreter can
also be set with `FLASHOCC_BROWSER_PYTHON`; it was enabled for the reference
viewer-quality measurements. Timing variations must be reported rather
than overwriting the frozen baseline.

## Development and model preparation

- [Application contracts and replay](app/README.md)
- [Execution and preprocessing settings](app/EXECUTION.md)
- [QAT training and dataset preparation](qat/README.md)
- [Model distribution and identities](models/README.md)
- [Reports and metric definitions](reports/README.md)
- [ARM pooling implementation](cpp/README.md)
- [Experiments and preserved investigations](experiments/README.md)

The deployment boundary remains camera → calibrated ARM pooling → BEV/head →
argmax. Camera order is front-left, front, front-right, back-left, back,
back-right. Model output class 17 is free space. Preserve upstream image
transforms and calibrated geometry; independent photographs are insufficient.
The QAT BEV compatibility copy changes shape metadata only, retaining its ELF.

Run the existing host tests with an appropriate validation environment:

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest viewer.test_server -v
python3 -m unittest discover -s qat/tests -v
```

ARM pooling/raycast/stream tests and device inference must also run on Modalix
when application behavior changes. Historical measurements and compiler/runtime
investigations remain in [docs/experiments](docs/experiments/FLASHOCC_HISTORY.md).
