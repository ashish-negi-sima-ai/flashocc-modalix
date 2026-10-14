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
    parser.add_argument('--train-count', type=int, default=32)
    parser.add_argument('--val-count', type=int, default=8)
    parser.add_argument('--val-scenes', nargs='+', default=MINI_VAL)
    args = parser.parse_args()
    torch.set_num_threads(4)
    tables = {name: {r['token']: r for r in json.loads((args.data/'v1.0-mini'/f'{name}.json').read_text())}
              for name in ['sample', 'sample_data', 'scene', 'calibrated_sensor', 'ego_pose', 'sensor']}
    # nuScenes SDK constructs this reverse index in memory; raw sample.json
    # does not contain sample['data'].
    for sample in tables['sample'].values():
        sample['data'] = {}
    for sd in tables['sample_data'].values():
        if sd['is_key_frame']:
            sensor_token = tables['calibrated_sensor'][sd['calibrated_sensor_token']]['sensor_token']
            channel = tables['sensor'][sensor_token]['channel']
            tables['sample'][sd['sample_token']]['data'][channel] = sd['token']
    pipeline = image_pipeline()
    transformer = upstream_class('necks/view_transformer.py', 'LSSViewTransformer')(
        dict(x=[-40, 40, .4], y=[-40, 40, .4], z=[-1, 5.4, 6.4], depth=[1., 45., 1.]),
        (256, 704), in_channels=256, out_channels=64)
    manifest = {'dataset': 'nuScenes v1.0-mini', 'camera_order': CAMERAS,
                'normalization': 'upstream PIL channel reversal then mean/std', 'samples': []}
    for split, names, count in [('calibration', MINI_TRAIN, args.train_count),
                                 ('validation', args.val_scenes, args.val_count)]:
        per_scene = []
        for scene_name in names:
            scene = next(s for s in tables['scene'].values() if s['name'] == scene_name)
            sequence, token = [], scene['first_sample_token']
            while token:
                sequence.append(tables['sample'][token])
                token = sequence[-1]['next']
            per_scene.append(sequence)
        selected = []
        for i, sequence in enumerate(per_scene):
            scene_count = count//len(names) + int(i < count % len(names))
            for index in np.linspace(0, len(sequence)-1, scene_count, dtype=int):
                selected.append(sequence[index])
        destination = args.output/split
        destination.mkdir(parents=True, exist_ok=True)
        for sample in selected:
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
            np.savez_compressed(path, images=images.numpy(), ranks_bev=rb.numpy(), ranks_depth=rd.numpy(),
                                ranks_feat=rf.numpy(), interval_starts=starts.numpy(), interval_lengths=lengths.numpy(),
                                sensor2key=sensor2key.numpy(), ego2global=ego2global.numpy(),
                                intrinsics=intrinsic.numpy(), post_rots=post_rot.numpy(), post_trans=post_trans.numpy())
            record = dict(split=split, token=sample['token'],
                          scene=tables['scene'][sample['scene_token']]['name'],
                          path=str(path.relative_to(args.output)), image_paths=paths,
                          pooling_contributions=len(rb))
            manifest['samples'].append(record)
            print(split, record['scene'], sample['token'], len(rb), flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
