"""Modalix compilation with explicit synthetic or real NCHW calibration tensors."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import onnx
from afe.apis.defines import default_quantization, gen2_target, bfloat16_scheme
from afe.apis.loaded_net import load_model
from afe.load.importers.general_importer import ImporterParams, ModelFormat
from afe.ir.tensor_type import ScalarType
from afe.core.configs import QuantizationPrecision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['camera', 'bev'], required=True)
    parser.add_argument('--model-path',type=Path,help='override ONNX path; matching audit sidecars are required')
    parser.add_argument('--calibration-dir', type=Path)
    parser.add_argument('--validation-dir', type=Path)
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--verify-count', type=int, default=2)
    parser.add_argument('--precision', choices=['int8','bfloat16','mixed-softplus'], default='int8')
    parser.add_argument('--executor',choices=['normal','jax'],default='normal')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model_path = args.model_path or root / 'artifacts/onnx-modalix' / (args.stage+'.onnx')
    audit_dtype = 'bfloat16' if args.precision == 'mixed-softplus' else args.precision
    audit = json.loads(model_path.with_suffix('.'+audit_dtype+'.audit.json').read_text())
    unsupported = audit['unsupported']
    if args.precision == 'mixed-softplus':
        if args.stage != 'bev' or {u['operator'] for u in unsupported} != {'Softplus'}:
            raise RuntimeError('Mixed policy requires BEV with Softplus as the only BF16 exception')
        int8_audit=json.loads(model_path.with_suffix('.int8.audit.json').read_text())
        if int8_audit['unsupported'] or int8_audit['unknown']:
            raise RuntimeError('INT8 audit for Softplus failed')
        unsupported=[]
    if unsupported or audit['unknown']:
        raise RuntimeError('Resolve the operator audit before compilation')
    model = onnx.load(model_path)
    inp = model.graph.input[0]
    shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
    importer = ImporterParams(format=ModelFormat.onnx, file_paths=[str(model_path)],
                             input_names=[inp.name], input_shapes=[shape],
                             input_types=[ScalarType.float32], layout='NCHW',
                             output_names=[o.name for o in model.graph.output])
    net = load_model(importer, target=gen2_target)
    quantization_config = default_quantization
    if args.precision in ['bfloat16','mixed-softplus']:
        quantization_config = quantization_config.with_activation_quantization(bfloat16_scheme()).with_weight_quantization(bfloat16_scheme())
    if args.precision == 'mixed-softplus':
        # Node names come from the already-quantized INT8 IR, as required by
        # the public custom-quantization API.
        baseline=json.loads((root/'artifacts/compile-real/bev/flashocc_m0_bev.sima.json').read_text())
        layers=baseline['modelTopology']['model_config']['config']['layers']
        softplus_names=[l['name'] for l in layers if 'softplus' in l['class_name'].lower()]
        if len(softplus_names)!=1:
            raise RuntimeError(f'Expected one Softplus node: {softplus_names}')
        quantization_config=quantization_config.with_custom_quantization_configs({
            name:{'quantization_precision':QuantizationPrecision.INT_8} for name in softplus_names})
        print('BF16 network with INT8 Softplus override:',softplus_names,flush=True)
    rng = np.random.default_rng(17)
    if args.calibration_dir:
        calibration = []
        for path in sorted(args.calibration_dir.glob('*.npy')):
            value = np.load(path, allow_pickle=False)
            if value.shape != shape or value.dtype != np.float32:
                raise ValueError(f'Invalid calibration tensor: {path}')
            calibration.append({inp.name: value.transpose(0,2,3,1)})
        if not calibration:
            raise ValueError('No calibration tensors')
        calibration_description = f'real nuScenes mini tensors: {len(calibration)}'
    else:
        calibration = [{inp.name: rng.standard_normal(shape).astype(np.float32).transpose(0, 2, 3, 1)}]
        calibration_description = 'synthetic normal; feasibility only'
    out = (args.build_dir or (root / 'artifacts/compile-smoke')) / args.stage
    out.mkdir(parents=True, exist_ok=True)
    (out/'status.json').write_text(json.dumps({'stage': args.stage, 'calibration': calibration_description,
                                             'production_ready': False, 'compiled': False}, indent=2))
    quant = net.quantize(calibration_data=calibration, quantization_config=quantization_config,
                         model_name='flashocc_m0_'+args.stage, log_level=logging.INFO)
    quant.save(model_name='flashocc_m0_'+args.stage, output_directory=str(out))
    if not args.no_compile:
        quant.compile(output_path=str(out), batch_size=1, log_level=logging.INFO)
    verification = []
    if args.validation_dir:
        for path in sorted(args.validation_dir.glob('*.npy'))[:args.verify_count]:
            value = np.load(path, allow_pickle=False)
            if value.shape != shape or value.dtype != np.float32:
                raise ValueError(f'Invalid validation tensor: {path}')
            inputs = {inp.name: value.transpose(0,2,3,1)}
            actual = list(quant.execute(inputs=inputs, use_jax=args.executor=='jax'))
            reference = list(net.execute(inputs=inputs))
            if len(actual) != len(reference):
                raise ValueError('Quantized and FP32 output counts differ')
            errors = []
            for q, f in zip(actual, reference):
                q, f = np.asarray(q), np.asarray(f)
                if q.shape != f.shape or not np.isfinite(q).all():
                    raise ValueError('Invalid quantized output')
                error = dict(shape=list(q.shape), mae=float(np.mean(np.abs(q-f))),
                             max_abs_error=float(np.max(np.abs(q-f))),
                             relative_rmse=float(np.sqrt(np.mean((q-f)**2))/(np.sqrt(np.mean(f**2))+1e-12)))
                if args.stage == 'bev':
                    if q.shape == (1,288,200,200):
                        q, f = q.transpose(0,2,3,1), f.transpose(0,2,3,1)
                    elif q.shape != (1,200,200,288):
                        raise ValueError(f'Unexpected BEV output layout: {q.shape}')
                    qlabels = q.reshape(1,200,200,16,18).argmax(-1)
                    flabels = f.reshape(1,200,200,16,18).argmax(-1)
                    error['label_agreement'] = float(np.mean(qlabels==flabels))
                errors.append(error)
            verification.append(dict(input=str(path),outputs=errors))
            print('Verification:', verification[-1],flush=True)
    (out/'status.json').write_text(json.dumps({'stage': args.stage, 'calibration': calibration_description,
                                             'production_ready': False, 'compiled': not args.no_compile,
                                             'precision': args.precision, 'executor':args.executor,
                                             'verification': verification}, indent=2))
    print('Completed; hardware and occupancy ground-truth validation remain.', flush=True)


if __name__ == '__main__':
    main()
