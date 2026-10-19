# Host 3D occupancy viewer

Open **http://10.42.0.1:8090/** to orbit and inspect the live FlashOCC volume.
The Python HTTP service runs on the SDK host, outside Docker. Three.js renders
in the browser using WebGL; the host's browser/GPU performs visualization.
FlashOCC inference remains on Modalix. The existing Insight mosaic is available
through the viewer's Open Insight link.

The default view uses actual metre coordinates and 0.4 m cells, with an ego
reference, directional axes, 5 m floor grid and volume boundary. It displays
exposed occupied voxels from the full 200 × 200 × 16 prediction. No voxel
sampling is applied by default. Interior cells remain available through the
exposed-voxel switch; height cuts and class filters expose internal surfaces.
Tall structures are clipped by the prediction's upper boundary at z=5.4 m;
their flat tops are volume cuts, not reconstructed building roofs.

Controls:

- Drag to orbit, scroll to zoom, right-drag to pan; Perspective/Top/Street presets.
- Scene/Objects/Ground class presets and individual class toggles.
- Height range, radius and optional XY sampling to reduce visible geometry.
- Vertical scale defaults to physical 1×; exaggeration is explicitly labeled.
- Pause/resume, voxel class/coordinate picking, PNG and full volume download.

The ego vehicle is an illustrative reference at the coordinate origin; it is
not a predicted object. Class counts in the sidebar refer to the full volume,
while the large voxel count is the number currently rendered after filtering.

## Setup

Inside the SDK container, from the repository:

```bash
python3 scripts/setup_3d_viewer.py
python3 scripts/run_insight_demo.py stop --config artifacts/fps-inputs/4/config.json
cmake --build artifacts/build-app --parallel 4
python3 scripts/run_insight_demo.py start --config artifacts/fps-inputs/4/config.json --volume-fps 2
```

Stop the application before replacing its executable. If it is already stopped,
build and then start it. `--volume-fps 0` disables 3D export; 2 is the default.
The ordinary default demo config remains at 2 FPS; the prepared faster replay
config runs at 4 FPS. The 3D export rate is independent of the video input rate.

In a **host terminal outside Docker**:

```bash
cd /home/ashishnegi/workspace/GitHub/occupancy-prediction
python3 scripts/run_host_3d_viewer.py start
python3 scripts/run_host_3d_viewer.py status
python3 scripts/run_host_3d_viewer.py stop
```

The host service needs only Python 3's standard library. Vendored browser assets
are pinned to Three.js 0.180.0, verified against the package's SHA-512 integrity
value and served locally; the browser does not require a CDN. Its MIT license
is retained in `vendor/LICENSE-three.txt`. Assets can be regenerated with the
setup script. See the [Three.js documentation](https://threejs.org/docs/) for
the browser rendering APIs.

For foreground operation, run `python3 viewer/server.py --port 8090`. Override
`--source` to view another compatible `.occ` snapshot. Use the appropriate host
address if accessing the service outside the current 10.42.0.x network.

## Data flow and frame consistency

```text
Modalix inference → uint8 XYZ labels + frame metadata
  → atomic shared-workspace snapshot → host HTTP service
  → gzip-compressed binary response → browser instanced 3D voxels
```

The producer writes and closes a temporary file before renaming it to
`artifacts/insight-device/latest.occ`. A reader sees one complete frame, never a
mix of header and voxel data from different frames. Browser polling uses ETags
to avoid retransmitting unchanged volumes. It displays data age, group/frame
IDs and stale/paused status. This is a latest-volume view: snapshots may skip
intermediate inference groups and are not frame-synchronized with WebRTC video.

The binary format is little-endian:

| Offset | Content |
|---|---|
| 0 | Eight bytes `FOCC3D01` |
| 8 | uint64 group sequence |
| 16 | uint64 replay frame index |
| 24 | uint64 device Unix timestamp, milliseconds |
| 32 | 32 ASCII bytes: sample token |
| 64 | 640000 uint8 class IDs in contiguous `[x,y,z]` order |

Voxel centre coordinates are `(-39.8 + 0.4*x, -39.8 + 0.4*y, -0.8 + 0.4*z)`.
The browser rotates this right-handed ego frame to Three.js Y-up as
`(worldX,worldY,worldZ)=(-egoY,egoZ,-egoX)`. Free class 17 is not rendered.

The server validates size/header/classes and compresses each new snapshot once.
A raw snapshot is 640064 bytes; a measured example compressed to about 49 KB.
The default export is capped at 2 snapshots/s. The browser renders camera
movement independently of the data update rate. Rendering performance depends
on the browser's WebGL acceleration and visible voxel count; Objects, radius,
height cuts and sampling controls can reduce the workload.

## Validation

```bash
python3 -m unittest viewer.test_server -v
ssh sima@10.42.0.73 \
  /workspace/GitHub/occupancy-prediction/artifacts/build-app/test_insight
```

Tests cover invalid/truncated snapshots, atomic replacement, frame metadata,
gzip delivery, conditional requests and startup waiting. Device tests verify
that replacing a snapshot preserves the previous frame for an already-open
reader. Browser checks exercise live updates, pause/resume, class filtering,
height cuts, vertical scale, orbit/top view and voxel picking. Evidence and
screenshots are under `artifacts/host-3d-viewer/`.

With default 2 Hz 3D export enabled, a 240-group steady window retained about
4.00 six-camera updates/s with no skipped groups. Snapshot publication took
about 20 ms on frames that published. The original FPS benchmark disables
volume export by default; use its `--volume-fps` option to measure both together.

The viewer displays model predictions, not ground truth. It improves geometric
inspection; it does not improve model accuracy.
