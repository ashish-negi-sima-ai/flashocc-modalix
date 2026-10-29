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

`qat-packed/comparison.json` compares packed QAT W8A8 camera archives with the
deployed QAT camera in the simulator; see
[QAT packing](../docs/experiments/QAT_PACKING.md).
The separate [2026-10-28 packed QAT app benchmark](qat-packed/MODALIX_APP.md)
records Modalix timing, model hashes and all per-update measurements. Sequential
EV74 remains fastest at 8.21 FPS; packed ARM routes measured 5.23–5.29 FPS.
The [packed-3 EV74 follow-up](qat-packed/PACKED3_EV74.md) records the opt-in EV74
implementation, input/output validation and actual throughput with visualization
disabled: 11.79 FPS, versus 12.13 FPS for sequential EV74.
The [direct MLA ELF profile](qat-packed/MLA_ONLY.md) isolates accelerator execution
with AppComplex disabled: six sequential-camera executions take 7.70 ms, versus
8.93 ms for two packed-3 executions; the shared BEV ELF takes 4.08 ms.
The [packed-6 EV74 follow-up](qat-packed/PACKED6_EV74.md) validates the single-call
INT8 packing route and records fresh visualization-disabled throughput:
11.70 FPS for packed-6, 11.79 FPS for packed-3 and 12.08 FPS for sequential EV74.
It includes numerical checks and the observed shutdown limitations.

Fresh cleanup validation belongs under ignored `artifacts/cleanup-validation/`.
Only its compact final audit should be added here; do not add image tensors,
checkpoints, full compiler IR or model binaries.
