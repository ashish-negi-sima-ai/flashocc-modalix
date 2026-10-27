"""Export the official ProtoOcc_1key learned networks without legacy CUDA imports.

Upstream classes supply the computation; small MMCV adapters cover this config
only. Checkpoint loading is strict. Geometry/voxel pooling remains an application
boundary, and the class-prototype boolean gather is rewritten as masked sums.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence, Union
import warnings

import numpy as np
import onnx
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp

from export_flashocc import BaseModule, BasicBlock, ImageBackbone

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/model-selection/ProtoOcc/projects/mmdet3d_plugin/models'
CLASSES = {}


class ConfigDict(dict):
    __getattr__ = dict.__getitem__


def config_dict(value):
    if isinstance(value, dict):
        return ConfigDict({k: config_dict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [config_dict(v) for v in value]
    return value


class ConvModule(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride=1, padding=0, bias='auto',
                 norm_cfg=None, act_cfg='default', conv_cfg=None, **kwargs):
        super().__init__()
        conv = nn.Conv3d if conv_cfg and conv_cfg['type'] == 'Conv3d' else nn.Conv2d
        self.conv = conv(cin, cout, kernel_size, stride, padding,
                         bias=(norm_cfg is None) if bias == 'auto' else bias)
        if norm_cfg is not None:
            self.bn = (nn.BatchNorm3d if conv is nn.Conv3d else nn.BatchNorm2d)(cout)
        self.activate = nn.ReLU() if act_cfg is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        return self.activate(x)


def build_norm_layer(cfg, channels):
    return ('ln', nn.LayerNorm(channels, eps=cfg.get('eps', 1e-5))) if cfg['type'] == 'LN' else ('bn', nn.BatchNorm2d(channels))


def build_module(cfg):
    cfg = config_dict(cfg)
    return CLASSES[cfg.pop('type')](**cfg)


class Attention(BaseModule):
    """MMCV MultiheadAttention's inference residual convention."""
    def __init__(self, embed_dims, num_heads, dropout=0., batch_first=False, **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=batch_first)

    def forward(self, query, key, value, identity=None, attn_mask=None, **kwargs):
        identity = query if identity is None else identity
        # need_weights=True exports explicit MatMul/Softmax in the legacy exporter.
        return identity + self.attn(query, key, value, attn_mask=attn_mask, need_weights=True)[0]


class DecoderSequence(BaseModule):
    def __init__(self, num_layers, transformerlayers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([build_module(transformerlayers) for _ in range(num_layers)])


class AnchorFreeHead(BaseModule):
    pass


class MaskHead(AnchorFreeHead):
    pass


class CpuOnly(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
            return node.func.value
        return node


def load_source(relative):
    path = SOURCE / relative
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)]
    for node in nodes:
        node.decorator_list = []
        for member in node.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                member.decorator_list = []
    tree = ast.fix_missing_locations(CpuOnly().visit(ast.Module(body=nodes, type_ignores=[])))
    scope = dict(nn=nn, torch=torch, F=F, np=np, cp=cp, copy=copy, warnings=warnings,
                 Optional=Optional, Union=Union, Sequence=Sequence, BaseModule=BaseModule, ConfigDict=ConfigDict,
                 BasicBlock=BasicBlock, ConvModule=ConvModule, ModuleList=nn.ModuleList,
                 checkpoint=cp if relative == 'backbones/resnet.py' else cp.checkpoint,
                 build_norm_layer=build_norm_layer, builder=SimpleNamespace(build_backbone=build_module, build_neck=build_module),
                 build_transformer=build_module, build_transformer_layer_sequence=build_module,
                 build_attention=lambda cfg: Attention(**{k:v for k,v in cfg.items() if k != 'type'}),
                 build_feedforward_network=lambda *a: None, TransformerLayerSequence=DecoderSequence,
                 AnchorFreeHead=AnchorFreeHead, MaskHead=MaskHead,
                 build_loss=lambda cfg: SimpleNamespace(class_weight=[1.] * 19),
                 preprocess_occupancy_gt=None, lovasz_softmax=None)
    exec(compile(tree, str(path), 'exec'), scope)
    for node in nodes:
        CLASSES[node.name] = scope[node.name]


def setup():
    for f in ['necks/fpn.py', 'necks/lss_fpn.py', 'backbones/resnet.py',
              'backbones/dual_branch_encoder.py', 'necks/depth_net.py',
              'dense_heads/cnn3d_decoder.py', 'necks/query_self_attention_for_RPL.py',
              'OccHead/Prototype_Query_Decoder_nuScenes.py']:
        load_source(f)
    config = {}
    exec((ROOT/'artifacts/model-selection/ProtoOcc/projects/configs/ProtoOcc/ProtoOcc_1key.py').read_text(), config)
    return config['model']


def load_weights(module, state, prefixes):
    selected = {k:v for k,v in state.items() if k.split('.')[0] in prefixes}
    module.load_state_dict(selected, strict=True)
    return len(selected)


class Camera(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_backbone = ImageBackbone()
        self.img_neck = build_module(cfg['img_neck'])
        depth = dict(cfg['depth_net']); depth['with_cp'] = False
        self.depth_net = build_module(depth)

    def forward(self, image, camera_parameters):
        features = self.img_neck(self.img_backbone(image))[0]
        context, depth = self.depth_net(features.unsqueeze(1), camera_parameters.reshape(1, 1, 27))
        return depth, context


class Occupancy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dual_branch_encoder = build_module(cfg['dual_branch_encoder'])
        self.cnn3d_decoder = build_module(cfg['cnn3d_decoder'])
        head = dict(cfg['prototype_query_decoder']); head['train_cfg'] = None; head['with_cp'] = False
        self.prototype_query_decoder = build_module(head)
        for module in self.modules():
            for attr in ['with_cp', 'use_checkpoint']:
                if hasattr(module, attr):
                    setattr(module, attr, False)

    def encode(self, x):
        features = self.dual_branch_encoder(x)
        prediction, mask = self.cnn3d_decoder(features.permute(0, 1, 4, 2, 3))
        return features, prediction, mask

    def decode(self, features, prediction, mask, static):
        head = self.prototype_query_decoder
        if not static:
            cls, masks, _ = head(features, [{}], mask, prediction)
            return head.format_results(cls[-1], masks[-1])
        probabilities = prediction.permute(0, 4, 1, 2, 3).softmax(1)
        top2 = probabilities.topk(2, dim=1).values
        valid = (1. - (top2[:, 0] - top2[:, 1])) < .9
        labels = probabilities.argmax(1)
        # Exactly the upstream per-class gather/mean, including empty-class zeros.
        assignments = (labels.flatten()[None, :] == torch.arange(18, device=labels.device)[:, None]) & valid.flatten()[None, :]
        weights = assignments.to(mask.dtype)
        # Keep both dynamic operands in a batched MatMul: a rank-2 dynamic RHS
        # is otherwise lowered to Dense by the compiler's importer.
        sums = (weights.reshape(1, 1, 18, -1) @ mask.reshape(1, 1, -1, 32)).reshape(18, 32)
        prototypes = sums / weights.sum(1, keepdim=True).clamp(min=1., max=float(weights.shape[1]))
        q = head.query_feat.weight + head.global_protoEMA_agg(head.prototype_EMA_feat.weight) + head.local_protoEMA_agg(prototypes)
        q = head.for_query_embed(q).unsqueeze(1)
        q = head.query_self_attn(q, [None])
        decoded = head.post_norm(q).transpose(0, 1)
        cls = head.cls_embed(decoded).softmax(-1)[..., :-1]
        embedding = head.mask_embed(decoded)
        feature_map = features.permute(0, 1, 3, 2, 4)
        spatial = feature_map.shape[2:]
        # The two source Einsums are contractions over channels and queries.
        # Flatten voxel axes and express both as explicit 4D batch MatMuls.
        masks = (embedding.reshape(1, 1, 18, 48) @ feature_map.reshape(1, 1, 48, -1)).sigmoid()
        scores = cls.transpose(1, 2).reshape(1, 1, 18, 18) @ masks
        return scores.reshape(1, 18, *spatial)

    def forward(self, voxels):
        return self.decode(*self.encode(voxels), static=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['camera', 'occupancy'], required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'artifacts/protoocc/ProtoOcc_1key.pth')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/protoocc/onnx')
    args = p.parse_args()
    torch.set_num_threads(4); torch.manual_seed(17)
    torch.backends.mha.set_fastpath_enabled(False)
    cfg = setup()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['state_dict']
    model = (Camera(cfg) if args.stage == 'camera' else Occupancy(cfg)).eval()
    prefixes = ['img_backbone', 'img_neck', 'depth_net'] if args.stage == 'camera' else ['dual_branch_encoder', 'cnn3d_decoder', 'prototype_query_decoder']
    count = load_weights(model, state, prefixes)
    del state
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'stage': args.stage, 'checkpoint': str(args.checkpoint), 'strict_checkpoint_load': True,
              'loaded_tensors': count, 'inputs': 'synthetic; not dataset accuracy validation',
              'source_adapters': 'Legacy registry/CUDA imports removed; configuration-specific MMCV adapters; checkpoint keys/shapes checked strictly',
              'geometry_pooling_in_graph': False}
    print('STRICT_LOAD', args.stage, count, flush=True)
    if args.stage == 'camera':
        inputs = (torch.randn(1, 3, 256, 704), torch.randn(1, 27, 1, 1))
        names, outputs = ['image', 'camera_parameters'], ['depth', 'features']
    else:
        inputs = (torch.randn(1, 80, 16, 200, 200) * .1,)
        names, outputs = ['voxels'], ['occupancy_scores']
    with torch.no_grad():
        if args.stage == 'occupancy':
            intermediate = model.encode(inputs[0])
            expected = model.decode(*intermediate, static=True)
            original = model.decode(*intermediate, static=False)
            torch.testing.assert_close(expected, original, rtol=1e-4, atol=1e-4)
            report['prototype_rewrite_max_abs_error'] = float((expected-original).abs().max())
            del intermediate, original
        else:
            expected = model(*inputs)
        expected = expected if isinstance(expected, tuple) else (expected,)
        report['input_shapes'] = [list(x.shape) for x in inputs]
        report['output_shapes'] = [list(x.shape) for x in expected]
        for name, x in zip(names, inputs):
            np.save(args.output/f'{args.stage}.{name}.npy', x.numpy(), allow_pickle=False)
        for name, x in zip(outputs, expected):
            np.save(args.output/f'{args.stage}.{name}.reference.npy', x.numpy(), allow_pickle=False)
        print('REFERENCE', report, flush=True)
        (args.output/f'{args.stage}.export.json').write_text(json.dumps(report, indent=2)+'\n')
        path = args.output/f'{args.stage}.onnx'
        torch.onnx.export(model, inputs, str(path), input_names=names, output_names=outputs,
                          opset_version=17, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(str(path))
    report['onnx_checked'] = True
    report['onnx_sha256'] = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    (args.output/f'{args.stage}.export.json').write_text(json.dumps(report, indent=2)+'\n')
    print('EXPORTED', path, flush=True)


if __name__ == '__main__':
    main()
