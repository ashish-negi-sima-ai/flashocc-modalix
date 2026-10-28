QAT W8A8 packed camera models
=============================

The deployed QAT camera archive runs six sequential camera calls. This
experiment applies the guarded spatial packing from
[CAMERA_BATCH](CAMERA_BATCH.md) to the same QAT-trained weights, giving two,
three or six cameras per batch-one call. The recipe matches the deployed
archive: PTQ with `default_quantization` (W8A8) on the Q/DQ-stripped QAT
graph, compiled with batch size one. No weights change.

Graph preparation
-----------------

The stripped QAT export (`artifacts/qat-runs/m0-full/onnx-stripped/camera.fp32.onnx`)
differs from the original FP32 export in three ways that the packing adapter
rejects. `prepare_packed_qat_camera.py` normalizes it before packing:

* The FPN Resize size comes from a Shape→Slice→Concat chain. The input shape
  is static, so it is folded into an initializer, `[1,256,16,44]`.
* 452 dead Constant and 50 dead Identity nodes are pruned.
* The depth output `/Softmax_output_0` is renamed `depth`.

All learned tensors are checked byte for byte. The normalized graph
(`artifacts/onnx-qat/camera.onnx`) is bit-exact with the stripped graph in ONNX
Runtime. The packed 2/3/6-camera exports (`artifacts/onnx-qat-packedN/`) have
the same 18 guard sites as the original packing. On 8 validation groups
(48 images), all unpacked depth and feature outputs are **bit-exact** with six
separate FP32 calls.

Compilation
-----------

Calibration uses all six views of the first 64 QAT calibration samples (384
images), grouped as at runtime: 64, 128 or 192 packed tensors. The deployed
archive was calibrated on 64 CAM_FRONT_LEFT images only. A sequential control
with the same 384 images separates calibration effects from packing effects.

```bash
V=/path/to/afe_develop_venv/bin/python   # SiMa Model SDK 3.0.0
$V experiments/packing/prepare_packed_qat_camera.py
for n in 6 3 2 1; do   # 1 = unpacked sequential control
  $V experiments/packing/compile_packed_qat_camera.py --cameras $n --no-validate
done
# Baseline: the deployed quantized camera model, evaluated with the same code.
$V experiments/packing/compile_packed_qat_camera.py --cameras 1 --validate-only \
  --load-dir artifacts/qat-runs/m0-full/ptq-compile/camera \
  --build-dir artifacts/compile-qat-deployed-eval
for n in 6 3 2 1; do
  $V experiments/packing/compile_packed_qat_camera.py --cameras $n --validate-only \
    --reference artifacts/compile-qat-deployed-eval/camera
done
```

Each archive is `artifacts/compile-qat-packedN/camera/flashocc_m0_camera_mpk.tar.gz`
(control: `compile-qat-sequential`). Each has one MLA stage at batch one, with
EV74 quantize, output unpack/slice and dequantize stages. The float32 outputs are
`[1,16,W,44]` depth and `[1,16,W,64]` features, with W = 274, 136 or 90. On this
host, calibration plus compilation took about 38, 32 and 22 minutes for 6, 3 and
2 cameras when the four jobs ran in parallel.

Simulator accuracy
------------------

Simulator (`use_jax`) results on the 32 QAT validation samples. Camera outputs
are unpacked, pooled with the FP32 reference and passed to the deployed QAT PTQ
BEV model. Metrics follow `reports/qat/m0-full/ptq-compile/accuracy.json`; the
re-evaluated deployed model reproduces its 34.29% exactly.

| Camera model | Calls per frame | Camera-masked mIoU | Accuracy | Labels equal to deployed | Labels equal to control |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deployed QAT PTQ | 6 | 34.29% | 86.35% | — | 95.66% |
| Sequential control | 6 | 34.54% | 86.73% | 95.66% | — |
| Packed, 2 cameras | 3 | 34.20% | 85.92% | 95.69% | 95.37% |
| Packed, 3 cameras | 2 | 34.20% | 86.30% | 95.90% | 95.93% |
| Packed, 6 cameras | 1 | 34.62% | 86.54% | 95.46% | 95.61% |

All variants are within ±0.4 mIoU points of the deployed archive. Recalibration
alone, with no packing, changes about 4.3% of voxel labels. Packing changes a
similar fraction, so this INT8 calibration sensitivity, not packing, dominates
the differences. Unlike the BF16 packed models, INT8 packed outputs are not
bit-identical to sequential outputs, since activation scales come from
calibration. Camera relative RMSE against FP32 is 20–25% for every INT8 variant,
including the deployed one. Details: `reports/qat-packed/comparison.json`.

Application presets and hardware validation
-------------------------------------------

`configs/experiments/qat-packed{6,3,2}.json` select the supplied packed QAT camera
archives under `artifacts/compile_with_qat_better_accuracy/compile-qat-packedN/`
with the deployed compatible QAT BEV archive (`artifacts/qat-app/compatible/`). These presets use ARM
normalization and A65 camera conversion like the BF16 packed presets. The
fastest sequential QAT preset uses EV74 preprocessing instead (8.26 versus
5.05 FPS for sequential ARM FP32).

On 2026-10-28, all three supplied archives passed startup output-contract checks
and eight normalized replay groups on Modalix, with finite BEV logits and valid
semantic labels. Packed outputs agreed with sequential QAT labels on 95.6–96.1%
of voxels in these eight groups; these are prediction-agreement checks, not
ground-truth accuracy measurements.

The full app measured 5.23, 5.29 and 5.25 FPS for two, three and six cameras per
call, respectively. Sequential QAT + EV74 measured 8.21 FPS. Packing reduces
camera runtime to approximately 19–20 ms per six-camera update, but ARM image
preparation and input copies take approximately 83 ms. The sequential EV74 route
takes approximately 5 ms for image preparation and input copies.

See the [hardware timing report](../../reports/qat-packed/MODALIX_APP.md) for
the protocol, model hashes, timing breakdown and reproduction commands. Hardware
ground-truth accuracy remains unmeasured; the simulator accuracy table above is unchanged.

The follow-up [packed-3 EV74 implementation and benchmark](../../reports/qat-packed/PACKED3_EV74.md)
adds `configs/experiments/qat-packed3-ev74-bgr.json`. EV74 normalizes and quantizes
individual cameras before ARM packs the INT8 values. With visualization disabled,
it measured 11.79 FPS versus 6.50 FPS for packed ARM and 12.13 FPS for sequential
EV74. The full Insight route measured 8.22 FPS. These measurements use the same
compiled archives; the EV74 path changes preprocessing and runtime composition.
