"""CPU-only M0 exporter: reuse upstream modules without legacy CUDA/MMCV imports.

The adapter covers only the upstream M0 configuration, not general MMCV.
Strict checkpoint loading rejects architecture/weight mismatches.
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from torchvision.models.resnet import BasicBlock as TVBasicBlock

ROOT = Path(__file__).resolve().parents[1]


class BaseModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias='auto', norm_cfg=None, act_cfg='default', **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=(norm_cfg is None) if bias == 'auto' else bias)
        if norm_cfg is not None:
            assert norm_cfg['type'] == 'BN'
            self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU() if act_cfg is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        return self.activate(x)


class BasicBlock(TVBasicBlock):
    def __init__(self, *args, norm_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)


def upstream_class(relative, name):
    """Execute the original class, stripping only registration decorators."""
    tree = ast.parse((ROOT / 'FlashOCC/projects/mmdet3d_plugin/models' / relative).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node.decorator_list = []
    for member in node.body:
        if isinstance(member, ast.FunctionDef):
            member.decorator_list = []
    scope = dict(nn=nn, torch=torch, F=F, np=np, BaseModule=BaseModule,
                 ConvModule=ConvModule, BasicBlock=BasicBlock,
                 build_norm_layer=lambda cfg, c: ('bn', nn.BatchNorm2d(c)),
                 build_loss=lambda cfg: None)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, 'exec'), scope)
    return scope[name]


class ImageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        reference = resnet50(weights=None)
        for name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']:
            setattr(self, name, getattr(reference, name))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer2(self.layer1(x))
        x = self.layer3(x)
        return [x, self.layer4(x)]


class M0(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_backbone = ImageBackbone()
        self.img_neck = upstream_class('necks/fpn.py', 'CustomFPN')(
            [1024, 2048], 256, 1, out_ids=[0])
        self.img_view_transformer = nn.Module()
        self.img_view_transformer.depth_net = nn.Conv2d(256, 108, 1)
        self.img_bev_encoder_backbone = upstream_class('backbones/resnet.py', 'CustomResNet')(
            64, num_channels=[128, 256, 512])
        self.img_bev_encoder_neck = upstream_class('necks/lss_fpn.py', 'FPN_LSS')(640, 128)
        self.occ_head = upstream_class('dense_heads/bev_occ_head.py', 'BEVOCCHead2D')(
            in_dim=128, out_dim=128, Dz=16, num_classes=18, loss_occ={})

    def camera(self, x):
        x = self.img_view_transformer.depth_net(self.img_neck(self.img_backbone(x))[0])
        return x[:, :44].softmax(1), x[:, 44:]

    def bev(self, x):
        return self.occ_head(self.img_bev_encoder_neck(self.img_bev_encoder_backbone(x)))


class Stage(nn.Module):
    def __init__(self, model, stage, conv_head=False):
        super().__init__()
        self.model, self.stage, self.conv_head = model, stage, conv_head

    def forward(self, x):
        if self.stage == 'camera':
            return self.model.camera(x)
        if not self.conv_head:
            return self.model.bev(x)
        x = self.model.img_bev_encoder_neck(self.model.img_bev_encoder_backbone(x))
        head = self.model.occ_head
        x = head.final_conv(x)
        # Pointwise Linear on NHWC is exactly a 1x1 convolution on NCHW.
        a, activation, b = head.predicter
        x = activation(F.conv2d(x, a.weight[:, :, None, None], a.bias))
        return F.conv2d(x, b.weight[:, :, None, None], b.bias)


def restore_logits(x):
    return x.permute(0, 3, 2, 1).reshape(1, 200, 200, 16, 18)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/onnx')
    parser.add_argument('--conv-head', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    model = M0().eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'strict_checkpoint_load': True, 'validation_inputs': 'seeded synthetic; no dataset accuracy validation', 'stages': {}}
    for stage, shape, outputs in [('camera', (1, 3, 256, 704), ['depth', 'features']),
                                  ('bev', (1, 64, 200, 200), ['logits'])]:
        wrapper = Stage(model, stage, args.conv_head).eval()
        x = torch.randn(shape)
        path = args.output / f'{stage}.onnx'
        with torch.no_grad():
            expected = wrapper(x)
            if stage == 'bev' and args.conv_head:
                reference = model.bev(x)
                restored = restore_logits(expected)
                torch.testing.assert_close(restored, reference, rtol=1e-4, atol=1e-4)
                report['head_rewrite_max_abs_error'] = (restored-reference).abs().max().item()
            torch.onnx.export(wrapper, (x,), str(path), input_names=[stage+'_input'],
                              output_names=outputs, opset_version=13, dynamo=False)
        onnx.checker.check_model(onnx.load(path))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        actual = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider']).run(
            None, {stage+'_input': x.numpy()})
        expected = expected if isinstance(expected, tuple) else (expected,)
        errors = []
        for ref, result in zip(expected, actual):
            np.testing.assert_allclose(result, ref.numpy(), rtol=1e-3, atol=1e-3)
            errors.append(float(np.max(np.abs(result-ref.numpy()))))
        report['stages'][stage] = {'input_shape': shape, 'output_shapes': [list(a.shape) for a in actual],
                                    'onnx_max_abs_errors': errors}
        print(stage, report['stages'][stage], flush=True)
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
