# Visualization recordings

These are live application predictions captured on 2026-10-27. Each H.264 MP4
contains exactly 10 seconds, at 1280 px width. The looping GIFs provide inline
README playback; clicking them opens the corresponding MP4.

| Clip | Contents | MP4 size |
| --- | --- | ---: |
| [Insight](insight.mp4) | Six calibrated camera overlays, two BEV slices and semantic legend | 2.2 MB |
| [Host 3D grid](host-3d-grid.mp4) | Live voxel updates and camera orbit in the native host viewer | 1.5 MB |

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
