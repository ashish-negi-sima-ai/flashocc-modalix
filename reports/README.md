# Results and evidence

The authoritative preserved Modalix comparison is
[`benchmarks/flashocc_20261027`](../benchmarks/flashocc_20261027/README.md).
It contains exact unrounded means/FPS, the same 32-token GT selection, model
hashes, per-frame records, four complete confusion matrices and per-sample
metrics. The [comparison report](../app/QAT_COMPARISON.md) explains scope.

`qat/m0-full` retains training metrics, run settings, ONNX export checks and
compiler/simulator accuracy reports from the QAT training work. These reports
are not the Modalix hardware measurements:

| Execution and model | Camera-masked mIoU |
| --- | ---: |
| Original FP32, 32-sample comparison | 34.58% |
| QAT QDQ ONNX, ONNX Runtime validation report | 36.51% |
| QAT weights with PTQ, AFE simulator report | 34.29% |
| QDQ-hint AFE compilation, simulator report | 33.70% |
| QAT weights with PTQ, Modalix hardware | 31.87% |

These values have different weights/execution paths. Retain their identities
and evaluate the matching model pair before attributing a difference to any
single compiler, quantizer or runtime component. Performance is the complete
six-camera app, not reciprocal synthetic model latency.

Fresh cleanup validation belongs under ignored `artifacts/cleanup-validation/`.
Only its compact final audit should be added here; do not add image tensors,
checkpoints, full compiler IR or model binaries.
