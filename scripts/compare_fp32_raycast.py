"""Compare FP32 and device labels on identical inputs through the live raycaster."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageColor,ImageDraw
from prepare_nuscenes import image_pipeline
from preview_real import COLORS

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/fp32-raycast-comparison')
    parser.add_argument('--raycaster',type=Path,default=ROOT/'artifacts/build-pool-native/raycast_volume')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'artifacts/insight-demo/config.json').read_text())
    records=json.loads((ROOT/'artifacts/insight-inputs/manifest.json').read_text())['samples']
    pipeline=image_pipeline();palette=np.array([ImageColor.getrgb(c) for c in COLORS],np.uint8)
    results=[]
    for record in config['frames']:
        token=record['token'];ref_path=ROOT/'artifacts/real-predictions'/f'{token}.npy'
        device_path=ROOT/'artifacts/app-device-final'/f'{token}.labels.u8'
        if not ref_path.exists() or not device_path.exists():continue
        # Earlier device replay and FP32 used these exact uncompressed normalized inputs.
        with np.load(ROOT/'artifacts/real-inputs/validation'/f'{token}.npz') as original, \
             np.load(ROOT/'artifacts/insight-inputs/validation'/f'{token}.npz') as current:
            for key in ['images','ranks_depth','ranks_feat','ranks_bev','sensor2key','intrinsics','post_rots','post_trans']:
                np.testing.assert_array_equal(original[key],current[key])
        reference=np.load(ref_path).reshape(-1).astype(np.uint8)
        device=np.fromfile(device_path,np.uint8);assert device.size==reference.size
        ref_raw=args.output/f'{token}.fp32.u8';reference.tofile(ref_raw)
        visible=[];hits=[]
        for name,labels,path in [('fp32',reference,ref_raw),('modalix',device,device_path)]:
            hit_file=args.output/f'{token}.{name}.hits.i32'
            subprocess.run([str(args.raycaster),str(path),record['geometry'],str(hit_file)],check=True)
            h=np.fromfile(hit_file,np.int32).reshape(6,64,176);hits.append(h)
            v=np.full(h.shape,17,np.uint8);mask=h>=0;v[mask]=labels[h[mask]];visible.append(v)
        union=(visible[0]!=17)|(visible[1]!=17);occupied=(reference!=17)|(device!=17)
        result=dict(index=record['index'],token=token,identical_preprocessed_inputs=True,
            voxel_label_agreement=float((reference==device).mean()),
            occupied_union_voxel_agreement=float((reference[occupied]==device[occupied]).mean()),
            visible_class_agreement=float((visible[0]==visible[1]).mean()),
            occupied_union_visible_class_agreement=float((visible[0][union]==visible[1][union]).mean()),
            first_hit_voxel_agreement=float((hits[0]==hits[1]).mean()),
            per_camera_visible_class_agreement=[float((a==b).mean()) for a,b in zip(*visible)])
        canvas=Image.new('RGB',(2112,6*292+80),'#171a20');draw=ImageDraw.Draw(canvas)
        for col,title in enumerate(['FP32 reference: full 3D raycast','Modalix BF16/mixed: same raycaster','Changed visible classes in red']):
            draw.text((col*704+12,10),title,fill='white')
        sample=records[record['index']]
        for camera,filename in enumerate(sample['image_paths']):
            with Image.open(ROOT/'artifacts/data/nuscenes-mini'/filename) as source:
                resize,dims,crop,flip,rotate=pipeline.sample_augmentation(source.height,source.width)
                image=pipeline.img_transform_core(source,dims,crop,flip,rotate).convert('RGB')
            for col in range(3):
                base=np.array(image).copy()
                if col<2:
                    v=visible[col][camera]
                    mask=np.array(Image.fromarray((v!=17).astype(np.uint8)).resize((704,256),Image.Resampling.NEAREST)).astype(bool)
                    colors=np.array(Image.fromarray(palette[v]).resize((704,256),Image.Resampling.NEAREST))
                    base[mask]=(base[mask]*.65+colors[mask]*.35).astype(np.uint8)
                else:
                    mask=np.array(Image.fromarray((visible[0][camera]!=visible[1][camera]).astype(np.uint8)).resize((704,256),Image.Resampling.NEAREST)).astype(bool)
                    base[mask]=(base[mask]*.35+np.array([255,0,0])*.65).astype(np.uint8)
                y=40+camera*292;canvas.paste(Image.fromarray(base),(col*704,y+22))
                draw.text((col*704+10,y+4),config['clips'][camera]['camera'],fill='white')
        draw.text((12,6*292+48),f'Frame {record["index"]} | voxel agreement {result["voxel_label_agreement"]:.2%} | visible-class agreement {result["visible_class_agreement"]:.2%} | predictions, not ground truth',fill='white')
        image_path=args.output/f'frame-{record["index"]:02d}-comparison.jpg';canvas.save(image_path,quality=95)
        result['visualization']=str(image_path);results.append(result);print(json.dumps(result),flush=True)
    report=dict(comparison='FP32 ONNX reference vs previously measured Modalix hardware outputs',
                input_scope='Identical upstream-preprocessed JPEG-derived inputs; live H264 compression excluded in both branches',
                renderer='Same C++ first_occupied_voxel used by live full-volume raycaster',
                ground_truth_accuracy_measured=False,samples=results)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
