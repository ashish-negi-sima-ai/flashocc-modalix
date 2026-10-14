# FlashOCC in Insight

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
