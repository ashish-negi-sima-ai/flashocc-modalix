"""Audit, validate, and attempt compiler-3.0.0 compilation of ProtoOcc exports."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import traceback
from importlib.metadata import version

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['camera', 'occupancy'], required=True)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--input-dir', type=Path, default=ROOT/'artifacts/protoocc/onnx')
    p.add_argument('--build-dir', type=Path, default=ROOT/'artifacts/protoocc/compile')
    args = p.parse_args()
    model_path = args.input_dir/f'{args.stage}.onnx'
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    shapes = [[d.dim_value for d in i.type.tensor_type.shape.dim] for i in model.graph.input]
    names = [i.name for i in model.graph.input]
    inputs = {n: np.load(args.input_dir/f'{args.stage}.{n}.npy') for n in names}
    if args.validate_only:
        import onnxruntime as ort
        options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), options, providers=['CPUExecutionProvider'])
        actual = session.run(None, inputs)
        report = {'scope': 'Synthetic input, export equivalence only; not dataset accuracy', 'outputs': []}
        for output, value in zip(model.graph.output, actual):
            reference = np.load(args.input_dir/f'{args.stage}.{output.name}.reference.npy')
            if not np.isfinite(value).all():
                raise ValueError('Non-finite ONNX output')
            np.testing.assert_allclose(value, reference, rtol=1e-3, atol=1e-3)
            report['outputs'].append({'name': output.name, 'shape': list(value.shape),
                'max_abs_error': float(np.max(np.abs(value-reference))),
                'relative_rmse': float(np.linalg.norm(value-reference)/(np.linalg.norm(reference)+1e-12))})
        report['passed'] = True
        (args.input_dir/f'{args.stage}.validation.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
        return
    validation = json.loads((args.input_dir/f'{args.stage}.validation.json').read_text())
    if not validation['passed']:
        raise RuntimeError('ONNX validation required first')
    out = args.build_dir/args.stage; out.mkdir(parents=True, exist_ok=True)
    guard = Path('/home/ashishnegi/.codex/skills/sima-model-surgery/scripts/model_surgery_guard.py')
    audits = {}
    for dtype in ['int8', 'bfloat16']:
        result = subprocess.run([sys.executable, str(guard), 'audit-model', '--model', str(model_path),
                                 '--dtype', dtype, '--json'], capture_output=True, text=True)
        audits[dtype] = json.loads(result.stdout)
        (out/f'{dtype}.audit.json').write_text(result.stdout)
    report = {'model': str(model_path), 'precision': 'bfloat16', 'input_shapes': dict(zip(names,shapes)),
        'onnx_sha256': hashlib.file_digest(model_path.open('rb'), 'sha256').hexdigest(),
        'scope': 'Real checkpoint; synthetic calibration; feasibility only; no device execution',
        'audit_database_release': audits['bfloat16']['release'],
        'audit_policy': 'Release 2.1 database is advisory for this explicit compiler-3.0.0 compatibility experiment; unsupported/unknown items are recorded, not presumed supported',
        'packages': {n:version(n) for n in ['sima-frontend','sima-mlc','sima-ml-kernels']},
        'imported': False, 'quantized': False, 'compiled': False}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        from afe.apis.loaded_net import load_model
        from afe.load.importers.general_importer import ImporterParams, ModelFormat
        from afe.apis.defines import gen2_target, default_quantization, bfloat16_scheme
        from afe.ir.tensor_type import ScalarType
        from afe.apis.error_handling_variables import enable_verbose_error_messages
        enable_verbose_error_messages()
        net = load_model(ImporterParams(format=ModelFormat.onnx, file_paths=[str(model_path)],
            input_names=names, input_shapes=shapes, input_types=[ScalarType.float32]*len(names),
            layout='NCDHW' if len(shapes[0])==5 else 'NCHW', flexible_batch_size=False,
            output_names=[i.name for i in model.graph.output]), target=gen2_target)
        report['imported'] = True
        print('IMPORTED', flush=True)
        calibration = {n:np.moveaxis(x,1,-1) for n,x in inputs.items()}
        config = default_quantization.with_activation_quantization(bfloat16_scheme()).with_weight_quantization(bfloat16_scheme())
        quant = net.quantize(calibration_data=[calibration], quantization_config=config,
                             model_name='protoocc_'+args.stage, log_level=logging.INFO)
        report['quantized'] = True
        quant.save(model_name='protoocc_'+args.stage, output_directory=str(out))
        layers = json.loads((out/f'protoocc_{args.stage}.sima.json').read_text())['modelTopology']['model_config']['config']['layers']
        report['layers'] = [{'name':l['name'],'class':l['class_name'],'backend':l.get('config',{}).get('backend')} for l in layers]
        print('QUANTIZED', flush=True)
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        quant.compile(output_path=str(out), batch_size=1, log_level=logging.INFO)
        report['compiled'] = True
        report['archives'] = [str(p) for p in out.glob('*mpk.tar.gz')]
    except (Exception, SystemExit) as exc:
        report['exception'] = type(exc).__name__
        report['error'] = str(exc)
        traceback.print_exc()
    finally:
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print('RESULT',json.dumps({k:v for k,v in report.items() if k!='layers'}),flush=True)
    if not report['compiled']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
