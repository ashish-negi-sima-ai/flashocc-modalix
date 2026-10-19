# FlashOCC in Insight

For an orbitable live 3D voxel view on the host, see the
[host 3D viewer](../viewer/README.md), available at http://10.42.0.1:8090/.
The Insight video mosaic remains available separately.

The [visual application walkthrough](../artifacts/application-flow/index.html)
shows actual camera images, measured intermediate tensors, component operations
and compute-unit placement in parallel input/component/output columns:

- [RTSP and preprocessing](../artifacts/application-flow/01-rtsp-and-preprocessing.png)
- [Inference and pooling](../artifacts/application-flow/02-inference-and-pooling.png)
- [Rendering and Insight delivery](../artifacts/application-flow/03-rendering-and-insight.png)

Generate these local illustrations from the existing device tensor dumps and
raycast comparison artifacts with
`/sdk-extensions/model-compiler/bin/python scripts/visualize_application_flow.py`.
Heatmaps summarize tensor magnitudes or scores; they are not additional semantic
predictions. Packet bytes and NV12 layout are explicitly schematic. The 3D
point view samples real occupied voxel centres with vertical scale exaggerated.

The live application consumes six RTSP feeds on Modalix, executes the compiled
FlashOCC camera and BEV models with ARM NEON pooling, and sends an annotated
H.264 mosaic to Insight channel 0. Inference uses the decoded stream pixels;
occupancy predictions are produced on the device for each matched frame.

Open the viewer:

<https://10.42.0.1:8081/static/viewer.html?mode=light&src=0&max_channels=4>

The console is <https://10.42.0.1:9900>. Both addresses come from this SDK's
published port map. The viewer shows six camera panels with depth-aware 3D
occupancy overlays, two semantic BEV slices (0.0 m and 1.2 m), a legend, frame
index and measured compute time. Camera overlays raycast the complete
200 × 200 × 16 volume and show the nearest occupied voxel along each camera ray.
Free voxels are transparent; farther occupied voxels are hidden by nearer ones.
The renderer uses 176 × 64 rays per camera, enlarged onto the camera panels and
blended at 35% opacity. The BEV grid covers 80 × 80 m;
up is vehicle +x, left is +y, and the black cross marks the ego origin.

## Run and stop

From the shared workspace on the development host:

```bash
python3 scripts/run_insight_demo.py start
python3 scripts/run_insight_demo.py status
python3 scripts/run_insight_demo.py stop
```

`start` uploads the six owned clips, assigns Insight source slots 1–6, starts
the publishers together and launches `flashocc_insight` through SSH. It preserves
other source assignments and refuses to replace occupied slots. If this
application is already running, `start` reports its status. Startup/model loading
and opening the six decoder graphs take roughly 20 seconds. `stop` verifies the
PID still belongs to this binary, sends SIGINT, waits for resource cleanup and
stops only these six owned source slots. Logs and saved outputs live in
`artifacts/insight-device/`.

For a finite device test with sources already running:

```bash
mkdir -p artifacts/insight-device-test
chmod 777 artifacts/insight-device-test
ssh sima@10.42.0.73 \
  '/workspace/GitHub/occupancy-prediction/artifacts/build-app/flashocc_insight --frames 40 --output-dir /workspace/GitHub/occupancy-prediction/artifacts/insight-device-test'
```

Do not run a second instance while the background application is active. The
`--config`, `--output-dir` and `--frames` flags are also available directly on
the device executable. A zero frame limit runs continuously.

## Prepared streams

The six 20-second H.264 clips contain the 40 consecutive nuScenes mini
`scene-0103` keyframes at 2 Hz. Each is 704 × 288: the top 256 rows contain the
upstream PIL resize/crop, and the bottom 32 rows contain a binary frame marker.
The marker encodes frame index, camera identity, complement and checksum. It is
removed before normalization, inference and display. Device normalization uses
the original model's BGR channel ordering, mean and standard deviation.

| Camera | Device RTSP URL |
|---|---|
| Front left | `rtsp://10.42.0.1:8554/src1` |
| Front | `rtsp://10.42.0.1:8554/src2` |
| Front right | `rtsp://10.42.0.1:8554/src3` |
| Back left | `rtsp://10.42.0.1:8554/src4` |
| Back | `rtsp://10.42.0.1:8554/src5` |
| Back right | `rtsp://10.42.0.1:8554/src6` |

Readers use bounded queues and retain at most 12 recent images each. A frame
enters inference only when all six validated markers agree. Clips repeat the
same images and geometry, so readers joining different loop generations can
still pair the same frame index. Stalled cameras or missing matching groups
produce an error after 20 seconds. This synchronization is specific to the
prepared replay clips; physical cameras need hardware timestamps and their own
synchronization policy.

Matching groups are drained in frame order. Hardware decoders can release a
burst of frames; choosing only the newest group would skip valid clip positions.
The device synchronization test covers burst ordering, different loop
generations, camera pairing, and corrupted/wrong-camera markers.

Pooling ranks and camera rays are prepared per keyframe using the real
camera intrinsics, camera/ego poses and upstream image transforms. The matching
geometry is selected by the decoded marker. H.264 CRF 16 introduces transport
compression relative to the original JPEG-derived model inputs.

The versioned `FOCCGEO2` geometry file contains rank count, the three int32 rank
arrays, six float32 camera origins and six sets of 176 × 64 float32 XYZ ray
directions. Rays undo the exact image post-transform and intrinsics, then rotate
into the same key-ego frame as the occupancy volume. Roundtrip reprojection of
all rays over all 40 frames has maximum error below 0.00013 pixels. A 1 m near
plane matches the model's depth range. DDA voxel traversal determines visibility
and uses the upstream XYZ label order.

The earlier camera renderer drew only the 1.2 m slice without occlusion; that
produced misleading horizontal bands. The full-volume renderer replaces it.
These are projected occupancy predictions with 0.4 m voxels, so their boundaries
remain coarser than image pixels.

### FP32 camera-overlay comparison

Four frames from this scene were compared using the FP32 ONNX reference and
previously recorded Modalix hardware outputs. Their preprocessed images, pooling
ranks and camera calibration were checked for exact equality, and both branches
used the same C++ ray traversal as the live renderer. Live H.264 compression was
excluded from both branches for this comparison.

| Frame | 3D voxel class agreement | Visible camera-ray class agreement |
|---|---:|---:|
| 0 | 98.92% | 99.88% |
| 13 | 99.40% | 98.42% |
| 26 | 99.29% | 98.88% |
| 39 | 99.29% | 99.32% |

Side-by-side overlays show the same broad shapes and boundary errors in FP32.
Deployment precision therefore does not explain most of the observed visual
errors on these frames. This is agreement between two predictions, not accuracy:
shared model/preprocessing/projection problems can affect both. In particular,
roundtrip calibration tests establish internal consistency but cannot prove
alignment to real scene surfaces. Occupancy ground truth and an untouched
upstream evaluation are still needed to attribute the remaining errors.

The FP32 reference uses our strict-checkpoint CPU/export adapter described in
the root README. Comparison figures (FP32, Modalix, visible-class changes) and
metrics are saved under `artifacts/fp32-raycast-comparison/`.

```bash
cmake --build artifacts/build-pool-native --target raycast_volume
/sdk-extensions/model-compiler/bin/python scripts/compare_fp32_raycast.py
```

## Faster replay and throughput measurement

The application reads integer `fps` (1–30) from its config and applies it to all
six decoder contracts, video input caps and the Insight encoder. Each processed
frame group still contains six synchronized camera views.

```bash
python3 scripts/run_insight_demo.py stop
python3 scripts/benchmark_insight_fps.py --rates 2 4 6 8 --frames 180 --warmup 60
```

Run only one application instance. The benchmark retimestamps each existing
clip, updates H.264 timing metadata, and verifies identical decoded pixels using
SHA-256. It uploads each rate to the same six owned Insight source slots, runs a
finite device test, and stops those publishers before the next rate. Faster
replay speeds up the recorded scene; it does not create new camera observations.

Results are saved under `artifacts/fps-runs/`, with per-rate summaries, browser
screenshots, ingest/egress statistics and clip verification. Sustained app FPS is
the number of completed groups divided by their wall-clock completion interval,
after excluding the initial 60 groups. Sequence gaps measure groups skipped
before inference; these are separate from browser video drops. The browser probe
measures a 20-second playback window and requires the optional Playwright Python
environment selected by `--browser-python`.

To leave the 4 FPS replay running after a benchmark:

```bash
python3 scripts/run_insight_demo.py start \
  --config artifacts/fps-inputs/4/config.json
```

Use the same `--config` with `stop` or `status`. The ordinary demo assets remain
at 2 FPS. `completed_ms` in `frames.jsonl` is a monotonic application timestamp;
its differences include input waiting, application work and logging between
completions. `application_ms` covers work for one group and is not, by itself, a
measurement of sustained FPS.

Measured on Modalix on 2026-10-15, with the full six-camera decode, MLA inference,
ARM NEON pooling/label selection, 3D occupancy overlays and Insight encoding:

| Input FPS per camera | Sustained six-camera updates/s | Groups skipped before inference |
|---|---:|---:|
| 2 | 2.00 | 0% |
| 4 | 4.01 | 0% |
| 6 | 4.40 | 26.5% |
| 8 | 4.36 | 44.9% |

Each run processed 180 groups; the last 120 form the steady measurement window.
All six cameras reported zero invalid markers at every rate. Independent
20-second browser windows measured approximately 2.00, 4.10, 4.35 and 4.35 FPS,
respectively, with no browser video drops or JavaScript errors. The browser
windows differ from the app measurement windows and include normal arrival
bursts. Detailed data is in `artifacts/fps-runs/summary.json`.

Use **4 FPS** for the current visualization app. Raising input to 6 or 8 FPS
does not improve output beyond roughly 4.4 FPS and causes synchronized groups
to be skipped. This is measured full-application throughput, including normal
input handling and output submission; it is distinct from the roughly 7 FPS
inference-compute estimate.

All graphics are drawn into the outgoing video on Modalix. Insight's ordinary
WebRTC viewer displays the result without a custom frontend or occupancy
metadata schema. Metadata counters are consequently zero. The output consumes
one viewer channel, which fits this SDK's four-channel UDP port mapping; the
six inputs remain separate RTSP sources.

## Rebuild the assets

```bash
/sdk-extensions/model-compiler/bin/python scripts/prepare_nuscenes.py \
  --data artifacts/data/nuscenes-mini --output artifacts/insight-inputs \
  --train-count 0 --val-count 40 --val-scenes scene-0103
/sdk-extensions/model-compiler/bin/python scripts/prepare_insight_demo.py
cmake -S app -B artifacts/build-app -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/app/toolchain.cmake"
cmake --build artifacts/build-app --parallel 4
```

The generator's `--host`, `--rtsp-port` and `--video-port` options support a
different SDK endpoint. Check `neat --json` before changing them. Device/Insight
control addresses can be overridden with the controller's `--device` and `--api`
flags. The geometry paths assume the workspace is shared at the identical path.

## Validation and troubleshooting

Device tests exercised six hardware decoders, MLA inference, NEON pooling/label
selection, calibrated rendering and hardware H.264 encoding. The initial
12-frame test had zero invalid markers. Independent host decoding verified all
40 markers in every clip. After the shared pipeline refactor, the original
replay sample retained its 99.2923% FP32 label agreement.

Browser validation confirmed 1536 × 864 video, advancing decoded frames and no
JavaScript errors. Evidence is in `artifacts/insight-browser-validation.json`
and `artifacts/insight-browser-live.png`. These are predictions; no ground-truth
accuracy is measured. At 2 Hz, each scene update is a genuine keyframe; motion
between keyframes is not interpolated.

The full-volume renderer passed more than 80 consecutive frames, covering all
40 positions with no sequence gaps. Compute takes about 144 ms; camera rendering
takes about 31 ms, and application work including preparation and rendering
takes about 225 ms. These timings
exclude waiting for the next RTSP group and WebRTC/browser latency. The input
scene updates at 2 Hz. The final report is
`artifacts/insight-demo-validation.json`. With a browser validation report present,
repeat the continuity check after at least two clip loops using:

```bash
python3 scripts/verify_insight_demo.py
ssh sima@10.42.0.73 \
  /workspace/GitHub/occupancy-prediction/artifacts/build-app/test_insight
ssh sima@10.42.0.73 \
  /workspace/GitHub/occupancy-prediction/artifacts/build-app/pooling/test_raycast
```

`frames.jsonl` records compute, geometry/normalization preparation, rendering and
application time separately. `latest.jpg` and `latest.labels.u8` update every 20
processed frames. On a clean stop, `summary.json` records total frames, marker
counts, camera receive/invalid-marker counts and up to 256 recent timing records.
The JSONL file retains the complete timing history.

```bash
tail -f artifacts/insight-device/live.log
curl -k https://10.42.0.1:9900/api/health
curl -k https://10.42.0.1:9900/api/ingest/stats
curl -k https://10.42.0.1:9900/api/egress/stats
```

For channel 0, rising RTP packets plus SPS/PPS and IDR counts confirm video
ingest. With the viewer open, check advancing browser `frames_decoded` in egress
stats. A stopped viewer can leave historical peer records; use recent active
peer reports. If the application exits, inspect `live.log`, check all six source
states, then restart with the controller.
# Execution placement and preprocessing

Camera inference can use six sequential calls or one spatially packed call.
Select `app/execution-packed.json` when starting the controller to use the packed
archive throughout the full RTSP-to-Insight app. Camera preprocessing can run
on ARM or EV74 in sequential mode; packed mode uses ARM normalization and A65
tensor conversion. See [execution configuration](EXECUTION.md) for presets,
channel-order handling, validation, performance evidence and commands.
