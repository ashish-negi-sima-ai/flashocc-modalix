"""Create six cropped RTSP clips, per-frame geometry and an Insight demo config."""
import argparse
import json
import struct
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from prepare_nuscenes import image_pipeline

ROOT = Path(__file__).resolve().parents[1]

def marker(index, camera):
    # 40 robust black/white cells: magic, frame, complement, camera/version, checksum.
    values = [0xA5, index, index ^ 255, (camera << 4) | 1, index ^ camera ^ 0x5A]
    return [(v >> bit) & 1 for v in values for bit in range(7, -1, -1)]

def camera_rays(inputs):
    u,v=np.meshgrid((np.arange(176)+.5)*4-.5,(np.arange(64)+.5)*4-.5)
    pixels=np.stack((u,v,np.ones_like(u)),axis=0).reshape(3,-1)
    origins=[];directions=[];errors=[]
    for camera in range(6):
        transform=inputs['sensor2key'][camera].astype(np.float64)
        original=np.linalg.inv(inputs['post_rots'][camera]) @ (pixels-inputs['post_trans'][camera,:,None])
        rays=transform[:3,:3] @ np.linalg.inv(inputs['intrinsics'][camera]) @ original
        origin=transform[:3,3].astype(np.float32);rays=rays.T.astype(np.float32)
        # Verify the saved rays return to their source pixels, with the real crop.
        xyz=origin[:,None]+rays.T*10
        sensor=np.linalg.inv(transform) @ np.vstack((xyz,np.ones((1,xyz.shape[1]))))
        projected=inputs['intrinsics'][camera] @ sensor[:3]
        projected/=projected[2:3]
        projected=inputs['post_rots'][camera] @ projected+inputs['post_trans'][camera,:,None]
        errors.append(float(np.abs(projected[:2]-pixels[:2]).max()))
        origins.append(origin);directions.append(rays)
    assert max(errors)<.002,errors
    return np.asarray(origins),np.asarray(directions),errors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, default=ROOT/'artifacts/insight-inputs')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/insight-demo')
    parser.add_argument('--host', default='10.42.0.1')
    parser.add_argument('--rtsp-port', type=int, default=8554)
    parser.add_argument('--video-port', type=int, default=9000)
    parser.add_argument('--geometry-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.inputs/'manifest.json').read_text())
    records = manifest['samples']
    assert 1 <= len(records) <= 256 and len({r['scene'] for r in records}) == 1
    args.output.mkdir(parents=True, exist_ok=True)
    geometry_dir = args.output/'geometry'; geometry_dir.mkdir(exist_ok=True)
    pipeline = image_pipeline()
    frame_records = []
    roundtrip=[]
    for index, record in enumerate(records):
        with np.load(args.inputs/record['path']) as inputs:
            rd, rf, rb = [inputs[key] for key in ['ranks_depth','ranks_feat','ranks_bev']]
            origins,directions,errors=camera_rays(inputs)
            roundtrip.extend(errors)
            path = geometry_dir/f'{index:03d}.geom'
            with path.open('wb') as f:
                f.write(b'FOCCGEO2');f.write(struct.pack('<Q',len(rd)))
                for array, dtype in [(rd,'<i4'),(rf,'<i4'),(rb,'<i4'),(origins,'<f4'),(directions,'<f4')]:
                    f.write(np.asarray(array,dtype=dtype).tobytes())
        frame_records.append(dict(index=index,token=record['token'],geometry=str(path.resolve())))
    clips = []
    for camera, name in enumerate(manifest['camera_order']):
        path = args.output/f'flashocc-{name.lower()}.mp4'
        clip=dict(camera=name,file=path.name,source_index=camera+1,url=f'rtsp://{args.host}:{args.rtsp_port}/src{camera+1}')
        clips.append(clip)
        if args.geometry_only:
            if not path.is_file():raise FileNotFoundError(path)
            continue
        cmd = ['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
               '-s','704x288','-r','2','-i','pipe:0','-an','-c:v','libx264','-preset','fast',
               '-crf','16','-pix_fmt','yuv420p','-profile:v','baseline','-bf','0','-g','2',
               '-keyint_min','2','-sc_threshold','0','-movflags','+faststart',str(path)]
        process = subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for index, record in enumerate(records):
                with Image.open(ROOT/'artifacts/data/nuscenes-mini'/record['image_paths'][camera]) as image:
                    resize,dims,crop,flip,rotate = pipeline.sample_augmentation(image.height,image.width)
                    cropped = pipeline.img_transform_core(image,dims,crop,flip,rotate)
                    canvas = np.zeros((288,704,3),np.uint8);canvas[:256]=np.asarray(cropped)
                for bit,value in enumerate(marker(index,camera)):
                    canvas[264:288,bit*16:(bit+1)*16]=255 if value else 0
                process.stdin.write(canvas.tobytes())
        finally:
            process.stdin.close()
        if process.wait():raise RuntimeError(f'Encoding failed: {path}')
        print(path,flush=True)
    config = dict(scene=records[0]['scene'],fps=2,width=704,height=288,clips=clips,frames=frame_records,
                  insight_host=args.host,video_port=args.video_port,channel=0,
                  camera_overlay='Full XYZ occupancy volume; calibrated camera raycasting; nearest occupied voxel; 176x64 rays per camera',
                  geometry_format='FOCCGEO2',projection_roundtrip_max_pixels=max(roundtrip),
                  crop='Upstream PIL resize/crop applied before encoding; normalization on Modalix; footer excluded from model input',
                  provenance='nuScenes mini keyframes at 2 Hz, H.264 CRF 16; predictions are not ground truth')
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')

if __name__=='__main__':main()
