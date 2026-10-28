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

Application presets and open work
---------------------------------

`configs/experiments/qat-packed{6,3,2}.json` select each packed QAT camera with the
deployed compatible QAT BEV archive (`artifacts/qat-app/compatible/`). The app
requires `camera_input=arm-fp32` for packed mode, so these presets use ARM
normalization and A65 camera conversion like the BF16 packed presets. The
fastest sequential QAT preset uses EV74 preprocessing instead (8.26 versus
5.05 FPS for sequential ARM FP32).

Not yet measured: Modalix hardware execution, startup output-contract checks,
EV74 versus A65 input quantization speed on the wide inputs, Neat latency, and
full-application FPS. Copy the archives to the device workspace, then follow the
[packed validation workflow](CAMERA_BATCH.md#modalix-validation-workflow-for-the-packed-model)
and the [benchmark command](../../app/EXECUTION.md#packed-camera-inference).
