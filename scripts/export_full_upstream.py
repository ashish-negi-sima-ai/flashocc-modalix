"""Export the upstream M0 TRT forward/symbolic as one ONNX, without graph surgery.

Legacy CUDA/MMCV is unavailable here. Reuse the existing strict-weight CPU adapter
and execute pooling with a CPU reference during tracing only. The ONNX pooling
node uses the verbatim upstream TRTBEVPoolv2.symbolic implementation.
"""
import ast
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from torch.nn import functional as F
from export_flashocc import M0, ROOT
from pool_reference import pool


def original_class():
    path=ROOT/'FlashOCC/projects/mmdet3d_plugin/ops/bev_pool_v2/bev_pool.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TRTBEVPoolv2')
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    op=namespace['TRTBEVPoolv2']

    def cpu_forward(ctx,depth,features,rd,rf,rb,starts,lengths,height=128,width=128,z=1):
        if z!=1:raise ValueError('Only the M0 z=1 pooling configuration is supported')
        value=pool(depth.detach().numpy(),features.detach().numpy().transpose(0,3,1,2),
                   rd.detach().numpy(),rf.detach().numpy(),rb.detach().numpy(),height,width)
        return torch.from_numpy(value.transpose(0,2,3,1).copy())

    op.forward=staticmethod(cpu_forward)
    detector=ROOT/'FlashOCC/projects/mmdet3d_plugin/models/detectors/bevdet_occ.py'
    tree=ast.parse(detector.read_text(errors='replace'))
    source=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BEVDetOCCTRT')
    methods=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ['forward_ori','result_serialize']]
    cls=ast.ClassDef(name='FullM0',bases=[ast.Name(id='M0',ctx=ast.Load())],keywords=[],body=methods,decorator_list=[])
    tree=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
    namespace={'M0':M0,'TRTBEVPoolv2':op,'F':F}
    exec(compile(tree,str(detector),'exec'),namespace)
    return namespace['FullM0']


def main():
    torch.set_num_threads(4)
    destination=ROOT/'artifacts/full-upstream';destination.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'artifacts/real-inputs/manifest.json').read_text())
    record=next(r for r in manifest['samples'] if r['split']=='validation')
    with np.load(ROOT/'artifacts/real-inputs'/record['path']) as sample:
        names=['img','ranks_depth','ranks_feat','ranks_bev','interval_starts','interval_lengths']
        values=[sample['images'].copy()]+[sample[n].astype(np.int32) for n in names[1:]]
    model=original_class()().eval()
    model.load_state_dict(torch.load(ROOT/'artifacts/flashocc-r50-M0.pth',map_location='cpu',weights_only=True)['state_dict'],strict=True)
    model.wocc=True;model.wdet3d=False;model.uni_train=True;model.upsample=False
    model.img_view_transformer.D=44
    model.img_view_transformer.out_channels=64
    model.img_view_transformer.grid_size=torch.tensor([200,200,1])
    model.forward=model.forward_ori
    inputs=tuple(torch.from_numpy(v) for v in values)
    dynamic_axes={n:{0:'M' if n.startswith('ranks_') else 'N'} for n in names[1:]}
    path=destination/'flashocc_m0_original.onnx'
    with torch.no_grad():
        logits=model(*inputs)[0].numpy()
        torch.onnx.export(model,inputs,str(path),opset_version=11,dynamo=False,
                          input_names=names,output_names=['output_0'],dynamic_axes=dynamic_axes)
    graph=onnx.load(path)
    onnx.checker.check_model(graph)
    custom=[n for n in graph.graph.node if n.domain=='mmdeploy']
    assert len(custom)==1 and custom[0].op_type=='bev_pool_v2'
    assert list(custom[0].input)[2:]==names[1:]
    labels=logits.argmax(-1)
    expected=np.load(ROOT/'artifacts/real-predictions'/f"{record['token']}.npy")
    agreement=float(np.mean(labels==expected))
    if agreement<.9999:raise RuntimeError(f'Full CPU forward comparison failed: {agreement}')
    np.savez(destination/'sample-inputs.npz',**dict(zip(names,values)))
    np.save(destination/'sample-reference-logits.npy',logits)
    report=dict(export='single graph; original dense head, original forward_ori and custom pooling symbolic',
                trace_adapter='MMCV CPU adapter and CPU pooling execution; CUDA upstream CLI was not run',
                strict_checkpoint_load=True,graph_surgery=False,opset=11,input_names=names,input_shapes=[list(v.shape) for v in values],
                input_dtypes=[str(v.dtype) for v in values],sample_token=record['token'],
                output_shape=list(logits.shape),cpu_forward_label_agreement=agreement,
                custom_nodes=[dict(domain=n.domain,operator=n.op_type,inputs=list(n.input),
                                   attributes={a.name:onnx.helper.get_attribute_value(a) for a in n.attribute}) for n in custom],
                onnx_checker_passed=True,onnx_runtime_validation='not available without custom mmdeploy operator implementation')
    (destination/'export-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
