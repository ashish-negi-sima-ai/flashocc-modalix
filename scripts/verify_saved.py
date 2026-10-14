"""Compare a saved SDK model to ONNX on real holdout tensors."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from afe.apis.model import Model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['camera','bev'],required=True)
    parser.add_argument('--count',type=int,default=2)
    parser.add_argument('--executor',choices=['normal','jax'],default='jax')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    directory=root/'artifacts/compile-real'/args.stage
    quant=Model.load('flashocc_m0_'+args.stage,str(directory))
    options=ort.SessionOptions();options.intra_op_num_threads=4
    reference=ort.InferenceSession(str(root/'artifacts/onnx-modalix'/f'{args.stage}.onnx'),options,providers=['CPUExecutionProvider'])
    paths=sorted((root/'artifacts/real-calibration'/args.stage/'validation').glob('*.npy'))
    reports=[]
    for index in np.linspace(0,len(paths)-1,args.count,dtype=int):
        path=paths[index]; value=np.load(path)
        inputs={args.stage+'_input':value.transpose(0,2,3,1)}
        actual=quant.execute(inputs=inputs,use_jax=args.executor=='jax')
        expected=reference.run(None,{args.stage+'_input':value})
        errors=[]
        for q,f in zip(actual,expected):
            f=f.transpose(0,2,3,1)
            if q.shape!=f.shape:raise ValueError('SDK output layout mismatch')
            errors.append(dict(mae=float(np.mean(np.abs(q-f))),max_abs_error=float(np.max(np.abs(q-f))),
                               relative_rmse=float(np.sqrt(np.mean((q-f)**2))/(np.sqrt(np.mean(f**2))+1e-12))))
        reports.append(dict(input=str(path),outputs=errors))
        (directory/f'verification-{args.executor}.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(reports[-1],flush=True)


if __name__=='__main__':main()
