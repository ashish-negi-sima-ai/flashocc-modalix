"""Prepare real M0 inputs, reusing upstream image transforms and geometry.

Reads nuScenes JSON directly, so LiDAR and the legacy training stack are not needed.
"""
import argparse
import ast
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from export_flashocc import BaseModule, ROOT, upstream_class
from nuscenes_metadata import load_selected

CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
MINI_TRAIN = ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757',
              'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100']
MINI_VAL = ['scene-0103', 'scene-0916']


class Quaternion:
    def __init__(self, w, x, y, z):
        q = np.array([w, x, y, z], np.float64)
        self.q = q / np.linalg.norm(q)

    @property
    def rotation_matrix(self):
        w, x, y, z = self.q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def normalize(img):
    # Match mmcv.imnormalize(..., to_rgb=True), including OpenCV float64
    # subtraction/multiplication on the float32 buffer.
    image = np.array(img).astype(np.float32)
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
    mean = np.array([123.675, 116.28, 103.53], np.float32).astype(np.float64).reshape(1, -1)
    inv_std = 1 / np.array([58.395, 57.12, 57.375], np.float32).astype(np.float64).reshape(1, -1)
    cv2.subtract(image, mean, image)
    cv2.multiply(image, inv_std, image)
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


def image_pipeline():
    path = ROOT/'FlashOCC/projects/mmdet3d_plugin/datasets/pipelines/loading.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PrepareImageInputs')
    node.decorator_list = []
    scope = dict(torch=torch, np=np, Image=Image, Quaternion=Quaternion,
                 mmlabNormalize=normalize)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['PrepareImageInputs'](dict(cams=CAMERAS, Ncams=6, input_size=(256, 704),
                                          crop_h=(0., 0.), resize_test=0.))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/real-inputs')
    parser.add_argument('--version', choices=['v1.0-mini', 'v1.0-trainval'], default='v1.0-mini')
    parser.add_argument('--labels', type=Path, help='Occ3D gts directory containing scene-*/TOKEN/labels.npz')
    parser.add_argument('--selection', type=Path, help='Frozen manifest with exact split/token selections')
    parser.add_argument('--train-count', type=int, default=32)
    parser.add_argument('--val-count', type=int, default=8)
    parser.add_argument('--train-scenes', nargs='+')
    parser.add_argument('--val-scenes', nargs='+')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.version == 'v1.0-mini':
        train_names, val_names = MINI_TRAIN, MINI_VAL
    else:
        try:
            from nuscenes.utils.splits import train, val
        except ImportError:
            parser.error('Trainval preparation requires nuscenes-devkit from qat/requirements.txt')
        train_names, val_names = sorted(train), sorted(val)
    if args.train_scenes and not set(args.train_scenes) <= set(train_names):
        parser.error('--train-scenes must be a subset of the official training scenes')
    if args.val_scenes and not set(args.val_scenes) <= set(val_names):
        parser.error('--val-scenes must be a subset of the official validation scenes')
    train_names = args.train_scenes or train_names
    val_names = args.val_scenes or val_names
    selection = json.loads(args.selection.read_text())['samples'] if args.selection else None
    tables, selected = load_selected(args.data, args.version, train_names, val_names,
                                    args.train_count, args.val_count, CAMERAS, selection)
    pipeline = image_pipeline()
    transformer = upstream_class('necks/view_transformer.py', 'LSSViewTransformer')(
        dict(x=[-40, 40, .4], y=[-40, 40, .4], z=[-1, 5.4, 6.4], depth=[1., 45., 1.]),
        (256, 704), in_channels=256, out_channels=64)
    manifest = {'dataset': 'nuScenes ' + args.version, 'camera_order': CAMERAS,
                'normalization': 'upstream PIL channel reversal then mean/std', 'samples': []}
    for split in ['calibration', 'validation']:
        destination = args.output/split
        destination.mkdir(parents=True, exist_ok=True)
        for sample_split, sample in selected:
            if sample_split != split:
                continue
            cams = {}
            paths = []
            for camera in CAMERAS:
                sd = tables['sample_data'][sample['data'][camera]]
                calibration = tables['calibrated_sensor'][sd['calibrated_sensor_token']]
                ego = tables['ego_pose'][sd['ego_pose_token']]
                filename = args.data/sd['filename']
                if not filename.is_file():
                    raise FileNotFoundError(filename)
                cams[camera] = dict(data_path=str(filename), cam_intrinsic=calibration['camera_intrinsic'],
                                    sensor2ego_rotation=calibration['rotation'], sensor2ego_translation=calibration['translation'],
                                    ego2global_rotation=ego['rotation'], ego2global_translation=ego['translation'])
                paths.append(sd['filename'])
            images, sensor2ego, ego2global, intrinsic, post_rot, post_trans = pipeline.get_inputs({'curr': {'cams': cams}})
            # Exactly the key-camera ego frame used in BEVDet.prepare_inputs.
            global2key = torch.inverse(ego2global[0:1].double())
            sensor2key = (global2key @ ego2global.double() @ sensor2ego.double()).float()
            bda = torch.eye(3)[None]
            coor = transformer.get_lidar_coor(sensor2key[None], ego2global[None], intrinsic[None],
                                             post_rot[None], post_trans[None], bda)
            rb, rd, rf, starts, lengths = transformer.voxel_pooling_prepare_v2(coor)
            if rb is None or not len(rb):
                raise ValueError('No valid pooling contributions')
            path = destination/(sample['token']+'.npz')
            labels = {}
            if args.labels:
                scene_name = tables['scene'][sample['scene_token']]['name']
                with np.load(args.labels/scene_name/sample['token']/'labels.npz') as source:
                    labels = {k: source[k] for k in ['semantics', 'mask_camera', 'mask_lidar']}
                if any(a.shape != (200, 200, 16) for a in labels.values()):
                    raise ValueError('Occ3D labels and masks must be [200,200,16]')
                if not np.isin(labels['semantics'], range(18)).all():
                    raise ValueError('Invalid Occ3D semantic class')
                for key in ['mask_camera', 'mask_lidar']:
                    if not np.isin(labels[key], [0, 1]).all():
                        raise ValueError('Invalid Occ3D visibility mask')
            np.savez_compressed(path, images=images.numpy(), ranks_bev=rb.numpy(), ranks_depth=rd.numpy(),
                                ranks_feat=rf.numpy(), interval_starts=starts.numpy(), interval_lengths=lengths.numpy(),
                                sensor2key=sensor2key.numpy(), ego2global=ego2global.numpy(),
                                intrinsics=intrinsic.numpy(), post_rots=post_rot.numpy(), post_trans=post_trans.numpy(), **labels)
            record = dict(split=split, token=sample['token'],
                          scene=tables['scene'][sample['scene_token']]['name'],
                          path=str(path.relative_to(args.output)), image_paths=paths,
                          pooling_contributions=len(rb))
            manifest['samples'].append(record)
            print(split, record['scene'], sample['token'], len(rb), flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
