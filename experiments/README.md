# Optional deployment investigations

The normal application is documented in the root README. This directory keeps
investigation implementations separate from its build/run workflow.

- `batch6/`: true camera batching, compiler pooling-padding diagnostics, isolated
  Neat transport patch/build and tensor-boundary checks. These do not patch the
  installed runtime as part of normal setup. Historical findings are in
  [CAMERA_BATCH](../docs/experiments/CAMERA_BATCH.md) and
  [NEAT_BATCHING_BUG](../docs/experiments/NEAT_BATCHING_BUG.md).
- `packing/`: spatial packing exports/calibration, validation and reports for
  two, three or six cameras per batch-one invocation.
  [QAT_PACKING](../docs/experiments/QAT_PACKING.md) applies it to the QAT W8A8
  camera model.
- `configs/experiments` at the repository root contains packed and native-NV12
  presets. Native NV12 remains experimental due to its measured preprocessing
  mismatch; retain explicit selection in the application.
- [Layerwise PTQ](../docs/experiments/PTQ_ANALYSIS.md) preserves quantization
  investigations and their numerical scope.

Existing `scripts/` entry points forward to these implementations for CLI/import
compatibility. The old `app/execution-*.json` paths are compatibility links to
canonical presets under `configs/`. Experimental ARM64 targets remain available
in the CMake application build, with the same target names.

ProtoOcc export/compiler work is maintained on
`feature/protoocc-compiler-feasibility`; it is not part of the FlashOCC app.
