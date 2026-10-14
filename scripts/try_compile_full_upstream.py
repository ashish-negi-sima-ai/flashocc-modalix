"""Diagnostic original-graph AFE attempt after auditing; never rewrites/splits it."""
import json
import logging
from pathlib import Path
import traceback

import numpy as np
from afe.apis.defines import default_quantization,gen2_target
from afe.apis.loaded_net import load_model
from afe.load.importers.general_importer import ImporterParams,ModelFormat
from afe.ir.tensor_type import ScalarType

ROOT=Path(__file__).resolve().parents[1]


def main():
    directory=ROOT/'artifacts/full-upstream'
    report=json.loads((directory/'export-report.json').read_text())
    # User explicitly requested trying the original unsupported graph. Audits
    # are retained as evidence; this import is a diagnostic, not a support claim.
    for dtype in ['int8','bfloat16']:
        if not (directory/f'original.{dtype}.audit.json').is_file():
            raise RuntimeError('Run both operator audits first')
    result=dict(target='Modalix/gen2',compiler='2.1.3',model='flashocc_m0_original.onnx',
                graph_modified=False,graph_split=False,input_shape_binding='representative sample shapes; ONNX dynamic axes preserved',
                stage='import',imported=False,quantized=False,compiled=False)
    try:
        params=ImporterParams(format=ModelFormat.onnx,file_paths=[str(directory/'flashocc_m0_original.onnx')],
                              input_names=report['input_names'],input_shapes=[tuple(s) for s in report['input_shapes']],
                              input_types=[ScalarType.float32]+[ScalarType.int32]*5,layout='NCHW',output_names=['output_0'])
        net=load_model(params,target=gen2_target,log_level=logging.INFO)
        result['imported']=True;result['stage']='quantize'
        with np.load(directory/'sample-inputs.npz') as data:
            calibration={name:data[name].transpose(0,2,3,1) if name=='img' else data[name]
                         for name in report['input_names']}
        quant=net.quantize(calibration_data=[calibration],quantization_config=default_quantization,
                          model_name='flashocc_m0_original',log_level=logging.INFO)
        result['quantized']=True;result['stage']='compile'
        build=directory/'build';build.mkdir(exist_ok=True)
        quant.save(model_name='flashocc_m0_original',output_directory=str(build))
        quant.compile(output_path=str(build),batch_size=1,log_level=logging.INFO)
        result['compiled']=True;result['stage']='complete'
    except (Exception,SystemExit) as error:
        result['exception_type']=type(error).__name__;result['error']=str(error)
        if isinstance(error,SystemExit):
            result['exit_code']=error.code
            result['error']='AFE terminated ONNX import; see afe-attempt.log for the frontend rejection'
        (directory/'afe-traceback.txt').write_text(traceback.format_exc())
        print(traceback.format_exc(),flush=True)
    (directory/'compile-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
