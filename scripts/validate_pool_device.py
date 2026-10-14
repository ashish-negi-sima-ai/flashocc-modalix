"""Check actual Modalix pooling outputs and propagate them through host FP32 BEV."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=ROOT/'artifacts/pool-device-final')
    args=parser.parse_args()
    options=ort.SessionOptions();options.intra_op_num_threads=4
    bev=ort.InferenceSession(str(ROOT/'artifacts/onnx-modalix/bev.onnx'),options,providers=['CPUExecutionProvider'])
    records=json.loads((ROOT/'artifacts/pool-fixtures/manifest.json').read_text())
    reports=[]
    for record in records:
        token=record['token']
        path=args.results/f'{token}.f32'
        actual=np.fromfile(path,dtype='<f4').reshape(1,200,200,64)
        expected=np.load(ROOT/'artifacts/real-calibration/bev/validation'/f'{token}.npy').transpose(0,2,3,1)
        np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-4)
        logits=bev.run(None,{'bev_input':actual.transpose(0,3,1,2).copy()})[0]
        labels=logits.transpose(0,3,2,1).reshape(1,200,200,16,18).argmax(-1)
        reference=np.load(ROOT/'artifacts/real-predictions'/f'{token}.npy')
        report=dict(token=token,scene=record['scene'],max_abs_pooling_error=float(np.max(np.abs(actual-expected))),
                    label_agreement=float(np.mean(labels==reference)),changed_labels=int(np.sum(labels!=reference)))
        reports.append(report);print(report,flush=True)
    summary=dict(device_pooling='Modalix Cortex-A65, NEON FMA, 8 threads',
                 results_directory=str(args.results),
                 downstream='host ONNX FP32 BEV; no MLA inference in this test',
                 tolerance=dict(rtol=1e-4,atol=1e-4),samples=reports)
    (ROOT/'artifacts/pool-device-validation.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
