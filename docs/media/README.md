# Visualization recordings

These are live application predictions captured on 2026-10-27. Each H.264 MP4
contains exactly 10 seconds, at 1280 px width. The looping GIFs provide inline
README playback; clicking them opens the corresponding MP4.

| Clip | Contents | MP4 size |
| --- | --- | ---: |
| [Insight](insight.mp4) | Six calibrated camera overlays, two BEV slices and semantic legend | 2.2 MB |
| [Host 3D grid](host-3d-grid.mp4) | Live voxel updates and camera orbit in the native host viewer | 1.5 MB |

Additional recordings captured on **2026-10-29** provide exactly **15 seconds**
at **1920×1080**, H.264/yuv420p, encoded at 20 FPS with fast-start MP4 headers:

| Full HD clip | Contents | MP4 size |
| --- | --- | ---: |
| [Insight, 15 seconds](hd-15s/insight.mp4) | Live six-camera occupancy mosaic and two BEV slices | 8.1 MB |
| [Host 3D grid, 15 seconds](hd-15s/host-3d-grid.mp4) | Live volume updates during a camera orbit | 3.8 MB |

The Insight video is scaled from the live 1536×864 stream to 1920×1080. The
3D viewer is captured at a native 1920×1080 browser viewport. Both show the same
QAT/EV74 configuration, 8 FPS replay and 2 Hz snapshots described below.
Capture ran in native host Chrome using SwiftShader software WebGL, at physical
height scale and all 0.4 m cells within 30 m. Encoding does not change replay
speed; repeated encoded frames do not imply additional inference or rendering.
See [the Full HD recording metadata](hd-15s/recording.json) for renderer details,
live-update/orbit checks, model hashes, video probes and SHA-256 hashes. Both
files decoded without errors; the application, sources and host viewer were
stopped after capture.

The app uses `configs/execution/qat-ev74-bgr.json`: QAT camera and compatible QAT
BEV archives, six sequential camera calls, EV74 BGR preprocessing, ARM NEON
pooling, overlays enabled, logit validity scan disabled. Model hashes are recorded
in [recording.json](recording.json) and [the model manifest](../../models/manifest.json).

Input is the 40-frame nuScenes `scene-0103` replay at 8 FPS, looping every five
seconds. This accelerates the original 2 Hz keyframes fourfold. Occupancy snapshots
are published at 2 Hz. The 3D viewer shows all 0.4 m cells within a 30 m radius,
with exposed surfaces and physical height scale (`1×`). It renders in Chromium
on the host, separately from the networks running on Modalix.

Both clips retain real capture time. Encoding uses 20 FPS MP4 and 8 FPS GIF;
repeated video frames do not represent additional inference or 3D updates.
The headless capture uses software WebGL, so the viewer's render FPS can be low.
Use the [preserved comparison](../../app/QAT_COMPARISON.md) for performance numbers.

To record again, first start the app and native host viewer following the
[main quickstart](../../README.md#build-and-run). Use a Python environment with
Playwright and its Chromium browser installed, plus `ffmpeg` on `PATH`:

```bash
python -m pip install playwright
python -m playwright install chromium

python scripts/record_visualizations.py \
  --insight-url "$FLASHOCC_INSIGHT_URL" \
  --grid-url http://localhost:8090/ \
  --output docs/media \
  --work-dir artifacts/demo-capture/recording \
  --seconds 10
```

Use URLs reachable from the recording environment. The script waits for live
video and occupancy updates, records the Insight video track, drags the 3D view
to orbit it, and extracts the final 10 seconds of the viewer capture. Software
rendering can make the full recording session take several minutes. It writes
SHA-256 hashes and viewer checks to `recording.json`; raw captures stay under the
ignored work directory. Stop the app and host viewer after recording using the
quickstart stop commands.

For 15-second Full HD MP4-only output, use the same live URLs with:

```bash
python scripts/record_visualizations.py \
  --insight-url "$FLASHOCC_INSIGHT_URL" \
  --grid-url http://localhost:8090/ \
  --output docs/media/hd-15s \
  --work-dir artifacts/hd-capture/recording \
  --seconds 15 --width 1920 --height 1080 --mp4-only --crf 20
```

`--browser-executable /path/to/google-chrome` optionally uses an installed
Chrome browser. The capture checks that the WebGL context is live and voxel
geometry is rendered before recording. Orbit motion follows elapsed wall time
on rendered frames, avoiding a recording stretched by slow automation calls.
