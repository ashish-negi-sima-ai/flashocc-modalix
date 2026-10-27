# RTSP occupancy visualization

The app synchronizes six cropped RTSP camera clips by their encoded camera/frame
markers. VPU decodes; ARM performs NV12-to-BGR conversion; the selected ARM/EV74
preprocessing prepares inputs. MLA runs six camera calls and one BEV call with
ARM NEON pooling between them. ARM argmax, raycasting and mosaic construction
produce the Insight video. Occupancy snapshots feed the separate host 3D viewer.

## Start/stop

Configure `configs/runtime.local.json` or set `FLASHOCC_DEVICE` and
`FLASHOCC_INSIGHT_API`. Source URLs, geometry, FPS and output ports live in the
source JSON. Model paths, preprocessing and thread counts live in the execution
preset. Host and device must see the same shared workspace path.

```bash
python3 scripts/run_insight_demo.py start \
  --config artifacts/fps-inputs/12/config.json \
  --execution configs/execution/qat-ev74-bgr.json \
  --output-dir artifacts/live --check-finite-logits false
python3 scripts/run_insight_demo.py status \
  --config artifacts/fps-inputs/12/config.json --output-dir artifacts/live
python3 scripts/run_insight_demo.py stop \
  --config artifacts/fps-inputs/12/config.json --output-dir artifacts/live
```

These commands assume assets at their reference workspace-relative paths.
For independently restored assets, use `scripts/reproduce_comparison.py` to
materialize paths/configs. Stop an existing instance before changing settings;
the controller does not reconfigure a running process.

The controller reports the actual viewer URL returned by the Insight API.
It owns only the six source indices configured for this app. Video output is
1536×864, showing six camera overlays, two BEV slices and the class legend.
Camera overlays use the first occupied voxel along calibrated rays through the
full 3D prediction; they are model predictions, not ground truth.

For orbitable voxels, use [the host 3D viewer](../viewer/README.md). Its rendering
runs in the browser. Snapshot export is independently capped at 2 Hz by default;
that is not the inference or Insight video FPS.

## Performance and accuracy

The preserved comparison uses 12 FPS inputs, six views per completed update,
overlays, 2 Hz snapshots and the validity scan disabled. It measured 6.50 FPS
for the previous mixed models with EV74 and 8.31 FPS for QAT with EV74. Enabling
the QAT logit scan measured 6.38 FPS. Input rates above application capacity
cause older synchronized groups to be skipped.

These are complete-app measurements. See [the detailed comparison](QAT_COMPARISON.md)
for stage means, camera-masked GT accuracy and the distinction between normalized
model-level evaluation and video/EV74 preprocessing. The frozen baseline remains
immutable; cleanup validation is recorded separately.

## Configuration, validation and history

- [Execution settings](EXECUTION.md): scan toggle, placement, preprocessing,
  worker counts, packed inference and model selection.
- [Replay application](README.md): tensor contracts and numerical checks.
- [Historical Insight investigation](../docs/experiments/INSIGHT_HISTORY.md):
  older FPS runs, projection debugging, diagrams and original observations.

Run `test_insight`, `pooling/test_labels` and `pooling/test_raycast` on Modalix
when changing synchronization, layouts or rendering. The existing tests cover
marker ordering, burst handling, frame pairing, logit validation and atomic
snapshot publication. The benchmark harness retains app logs, raw frame records,
invalid-marker counts and browser quality counters in each results directory.
