FlashOCC layerwise INT8 PTQ analysis
==================================

Measured on the SDK host with AFE 2.1.3 on 2026-10-15, using the saved
fully INT8 experiments and their saved FP32 reference networks. The deployed
BF16-heavy networks were not changed. The Modalix application kept running.

Camera evaluation covers 12 images: all six camera views of one holdout
sample from each of scene-0103 and scene-0916. BEV evaluation covers all eight
holdout pooled tensors. These are small nuScenes-mini numerical checks, not
ground-truth occupancy accuracy measurements.

Each fused AFE operator was evaluated with reference inputs (local feed) and
quantized upstream inputs (global feed). Local error includes quantization of
the reference inputs as well as operator weights, arithmetic and output. Thus
it measures an operator boundary, rather than weight quantization alone.
Global error includes accumulated upstream error. Both use the same reference
tensor for normalization: `sqrt(MSE) / sqrt(mean² + std²)`. The table reports
the mean of the per-input relative RMSE values.

| Network | Fused operator / source | Local error | Accumulated error |
| --- | --- | ---: | ---: |
| Camera | `add_73`: image FPN merge | 7.31% | 26.36% |
| Camera | `conv2d_add_71`: FPN lateral convolution, second branch | 6.17% | 26.94% |
| Camera | `softmax_77`: depth softmax | 5.84% | 20.89% |
| Camera | `strided_slice_78`: feature output | 3.86% | 21.80% |
| BEV | `add_relu_6`: backbone stage 0, block 1 residual merge | 37.70% | 47.45% |
| BEV | `add_relu_13`: backbone stage 1, block 1 residual merge | 26.68% | 50.66% |
| BEV | `add_relu_3`: backbone stage 0, block 0 residual merge | 23.14% | 25.56% |
| BEV | `tuple_concat_22`: neck concatenation | 16.15% | 51.89% |
| BEV | `softplus_30`: occupancy head activation | 1.63% | 39.24% |
| BEV | `conv2d_add_31`: final classifier | 1.21% | 23.98% |

The camera has modest local errors distributed across many operators; its
backbone already supplies accumulated error to the FPN. A BF16 FPN experiment
is a useful candidate, but it cannot be assumed to remove the upstream error.
The BEV residual merges and neck concatenation are stronger candidates for
selective precision experiments. The final classifier and Softplus introduce
much less local error than those merges in this INT8 configuration.

The 12 camera convolutions with all-zero INT8 filters were checked against
their original folded FP32 weights. Within each affected convolution, the
zeroed filters account for at most **3.1e-17 of its original weight energy**.
This weakens the earlier suspicion based on the warnings alone. It does not
prove that every affected channel is irrelevant; biases and downstream
sensitivity are not captured by weight energy.

The next precision experiments should evaluate BEV residual blocks/merges
and neck concatenation first, then camera FPN and selected backbone blocks.
Recompilation and final occupancy comparisons are required to establish which
changes actually recover fidelity. These error rankings are not individual
BF16-ablation results, nor percentages of misclassified voxels. BEV inputs
here use FP32 camera/pooling, so cross-stage quantization error is excluded.

To reproduce on the SDK host:

```bash
activate-model-compiler
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/analyze_ptq_layers.py
python scripts/report_ptq_layers.py
```

The analysis uses the installed SDK's `QuantizedGraphAnalyzer` and
`IntermediatesGraphAnalyzer`. Its adapter accesses `Model._net` and
`Model._fp32_net` because the public high-level analyzer does not return
the per-input measurements needed for both modes. This adapter is specific to
the installed AFE version. Saved `.sima` YAML maps each weight array to its
operator; the independently exported Netron JSON array numbering must not be
used to index the `.sima` NPZ arrays.

The generated [full report](../../artifacts/ptq-layerwise/REPORT.md) contains all
rankings and [camera](../../artifacts/ptq-layerwise/camera.png) /
[BEV](../../artifacts/ptq-layerwise/bev.png) plots. The same directory contains
per-input JSON, all-layer CSV, aggregate JSON and model/reference SHA256 hashes.
Completion checks verified all 12 camera / eight BEV inputs, matching operator
coverage in both modes and reference statistics, and finite nonnegative MSE.
