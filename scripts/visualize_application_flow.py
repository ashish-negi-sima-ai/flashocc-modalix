"""Create input / component / output illustrations from measured FlashOCC tensors."""
import io
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageColor
from prepare_nuscenes import image_pipeline
from preview_real import COLORS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/application-flow'
TOKEN='3e8750f331d7499e9b5123e9eb70f2e2'
W,H=580,270
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
INK='#dce6f3';BG='#0e1624';MUTED='#9eb0c7'

def font(size,bold=False):return ImageFont.truetype(BOLD if bold else FONT,size)
def label(image,text,xy,size=18,color=INK,bold=False):
    ImageDraw.Draw(image).text(xy,text,font=font(size,bold),fill=color)
def panel():return Image.new('RGB',(W,H),'#172337')
def fit(image,size):
    result=image.copy();result.thumbnail(size,Image.Resampling.LANCZOS);return result
def heat(values,title,unit='',cmap='magma',limits=None,bev=False):
    if bev:values=values.T[::-1,::-1]
    low,high=limits or np.percentile(values,[1,99]);high=max(float(high),float(low)+1e-8)
    colors=(plt.get_cmap(cmap)(np.clip((values-low)/(high-low),0,1))[...,:3]*255).astype(np.uint8)
    scale=min(544/colors.shape[1],185/colors.shape[0])
    shape=(round(colors.shape[1]*scale),round(colors.shape[0]*scale))
    result=panel();image=Image.fromarray(colors).resize(shape,Image.Resampling.NEAREST)
    result.paste(image,((W-image.width)//2,35));label(result,title,(18,8),17,bold=True)
    gradient=np.tile(np.linspace(0,1,300),(12,1))
    strip=(plt.get_cmap(cmap)(gradient)[...,:3]*255).astype(np.uint8)
    result.paste(Image.fromarray(strip),(18,230))
    label(result,f'{low:.2g} to {high:.2g} {unit}',(330,224),16)
    return result
def merge(a,b,title):
    result=panel();label(result,title,(15,7),18,bold=True)
    for i,p in enumerate([a,b]):
        thumb=fit(p,(280,218));result.paste(thumb,(8+i*286,42+(218-thumb.height)//2))
    return result
def bytes_panel(title,kind='network'):
    result=panel();d=ImageDraw.Draw(result);label(result,title,(18,10),19,bold=True)
    if kind=='network':
        protocols=['SRTP','H.264','UDP / ICE'] if 'WebRTC' in title else ['RTP','H.264','RTSP / TCP']
        for row in range(3):
            for col in range(5):
                x=20+col*110;y=52+row*49
                d.rounded_rectangle((x,y,x+96,y+35),7,fill='#233e62',outline='#4e88c7')
                label(result,protocols[row],(x+10,y+8),12)
        label(result,'Encoded bytes / packets, not an RGB image',(20,223),18)
    else:
        label(result,'00 00 00 01  67 ... SPS',(24,65),23,color='#76c9ef')
        label(result,'00 00 00 01  68 ... PPS',(24,112),23,color='#76c9ef')
        label(result,'00 00 00 01  65 ... frame data',(24,159),21,color='#76c9ef')
        label(result,'Schematic H.264 byte stream',(24,225),18)
    return result

def page(name,title,subtitle,rows):
    width=1900;height=184+len(rows)*390+112
    image=Image.new('RGB',(width,height),BG);d=ImageDraw.Draw(image)
    label(image,title,(36,23),34,bold=True);label(image,subtitle,(36,73),20,color=MUTED)
    for x,text in [(36,'INPUT'),(700,'COMPONENT / WHAT IT DOES'),(1280,'OUTPUT')]:
        label(image,text,(x,123),19,color='#7eb7ec',bold=True)
    units={'HOST CPU':'#4f89b8','ARM CPU':'#438fc5','ARM NEON':'#438fc5','MLA':'#9b75d6',
           'VIDEO HARDWARE':'#d99b46','ARM + VIDEO HW':'#d99b46','HOST / ARM CPU':'#4f89b8','BROWSER':'#4cae99'}
    for index,(left,input_caption,component,unit,actions,right,output_caption) in enumerate(rows):
        y=170+index*390
        image.paste(left,(36,y));image.paste(right,(1280,y))
        for x,caption in [(36,input_caption),(1280,output_caption)]:
            for j,line in enumerate(textwrap.wrap(caption,56)):label(image,line,(x,y+280+j*23),17,color=MUTED)
        d.rounded_rectangle((700,y+13,1215,y+268),15,fill='#1c293d',outline=units[unit],width=2)
        d.rounded_rectangle((716,y+26,1199,y+60),6,fill=units[unit])
        label(image,unit,(729,y+32),17,bold=True,color='#07111d')
        label(image,component,(718,y+77),23,bold=True)
        for j,line in enumerate(textwrap.wrap(actions,38)):
            label(image,line,(718,y+119+j*25),18)
        for x in [635,1224]:
            d.line((x,y+135,x+40,y+135),fill='#90a7c1',width=4)
            d.polygon([(x+40,y+135),(x+30,y+128),(x+30,y+142)],fill='#90a7c1')
    y=height-91
    label(image,'Real sample: nuScenes scene-0103, frame 0. Tensor views use measured Modalix replay dumps.',(36,y),17,color=MUTED)
    label(image,'Heatmaps summarize tensors; they are not semantic images. Transport byte drawings are schematic.',(36,y+27),17,color=MUTED)
    path=OUT/name;image.save(path);return path

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'artifacts/insight-demo/config.json').read_text())
    manifest=json.loads((ROOT/'artifacts/insight-inputs/manifest.json').read_text())
    sample=manifest['samples'][0];assert sample['token']==TOKEN
    pipeline=image_pipeline();views=[]
    for filename in sample['image_paths']:
        with Image.open(ROOT/'artifacts/data/nuscenes-mini'/filename) as image:
            _,dims,crop,flip,rotate=pipeline.sample_augmentation(image.height,image.width)
            views.append(pipeline.img_transform_core(image,dims,crop,flip,rotate).convert('RGB'))
    cameras=panel()
    for i,view in enumerate(views):
        cameras.paste(view.resize((188,68)),(6+i%3*192,38+i//3*108))
        label(cameras,config['clips'][i]['camera'].replace('CAM_',''),(8+i%3*192,18+i//3*108),13)
    label(cameras,'Six views belonging to one frame group',(10,238),18)
    nv12=panel();gray=views[1].convert('L').resize((355,130));nv12.paste(gray,(14,43))
    rgb=np.array(views[1].resize((180,66)));uv=np.asarray(Image.fromarray(rgb).convert('YCbCr'))[:,:,1:]
    uv_rgb=np.zeros((66,180,3),np.uint8);uv_rgb[:,:,0]=uv[:,:,0];uv_rgb[:,:,2]=uv[:,:,1]
    nv12.paste(Image.fromarray(uv_rgb),(386,75))
    label(nv12,'NV12: brightness + subsampled color',(14,9),18,bold=True)
    label(nv12,'Y plane (brightness)',(14,187),16);label(nv12,'UV diagram',(386,155),15)
    label(nv12,'Illustration of the decoded memory layout',(14,230),17,color=MUTED)
    normalized=cameras.copy();label(normalized,'float32 BGR values after mean / std',(10,211),17,color='#8be3bd')
    root=ROOT/'artifacts/app-device-final'
    depth=np.fromfile(root/f'{TOKEN}.depth.f32',np.float32).reshape(6,16,44,44)
    feature=np.fromfile(root/f'{TOKEN}.features.f32',np.float32).reshape(6,16,44,64)
    pooled=np.fromfile(root/f'{TOKEN}.pooled.f32',np.float32).reshape(200,200,64)
    logits=np.fromfile(root/f'{TOKEN}.logits.f32',np.float32).reshape(200,200,16,18)
    labels=np.fromfile(root/f'{TOKEN}.labels.u8',np.uint8).reshape(200,200,16)
    expected=(depth[1]*np.arange(1,45)).sum(-1)/depth[1].sum(-1)
    depth_panel=heat(expected,'Front view: expected optical depth','m','plasma',(1,44))
    feature_panel=heat(np.sqrt(np.mean(feature[1]**2,-1)),'Front view: feature RMS','arbitrary units')
    camera_outputs=merge(depth_panel,feature_panel,'Depth probabilities + learned image features')
    bev_features=heat(np.sqrt(np.mean(pooled**2,-1)),'BEV feature RMS: 64 channels summarized','arbitrary units',bev=True)
    score=heat(logits[:,:,5,:].max(-1),'Largest raw class score at z = 1.2 m','logit',bev=True)
    palette=np.array([ImageColor.getrgb(c) for c in COLORS],np.uint8)
    bev=panel();label(bev,'Class labels at z = 1.2 m',(12,8),18,bold=True)
    bev.paste(Image.fromarray(palette[labels[::-1,::-1,5]]).resize((220,220),Image.Resampling.NEAREST),(15,37))
    label(bev,'Up: vehicle +x',(252,67),17);label(bev,'Left: vehicle +y',(252,105),17)
    label(bev,'Each cell: 0.4 x 0.4 m',(252,150),16);label(bev,'One of 16 height slices',(252,189),16)
    # Actual occupied voxel centres, sampled for a legible standalone 3D view.
    xyz=np.indices(labels.shape).reshape(3,-1).T;classes=labels.reshape(-1)
    keep=np.flatnonzero(classes!=17);rng=np.random.default_rng(0)
    keep=rng.choice(keep,min(12000,len(keep)),replace=False)
    points=xyz[keep]*.4+np.array([-39.8,-39.8,-.8])
    fig=plt.figure(figsize=(4,3),facecolor='#172337');ax=fig.add_subplot(111,projection='3d',facecolor='#172337')
    ax.scatter(*points.T,c=palette[classes[keep]]/255,s=2,alpha=.65,depthshade=False)
    ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',xlim=(-40,40),ylim=(-40,40),zlim=(-1,5.4))
    ax.set_box_aspect((1,1,.4));ax.view_init(26,-125);ax.tick_params(colors='white',labelsize=6)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.label.set_color('white')
    fig.tight_layout();buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=130);plt.close(fig);buffer.seek(0)
    cloud=panel();cloud.paste(Image.open(buffer).convert('RGB').resize((W,H)))
    volume=panel();label(volume,'BEV slice + sampled occupied 3D voxel centres',(12,7),18,bold=True)
    volume.paste(Image.fromarray(palette[labels[::-1,::-1,5]]).resize((190,190),Image.Resampling.NEAREST),(12,43))
    volume.paste(cloud.resize((365,170),Image.Resampling.LANCZOS),(211,55))
    label(volume,'z = 1.2 m',(25,243),15);label(volume,'3D: vertical scale exaggerated',(230,243),15)
    h=np.fromfile(ROOT/'artifacts/fp32-raycast-comparison'/f'{TOKEN}.modalix.hits.i32',np.int32).reshape(6,64,176)
    mosaic=Image.new('RGB',(1056,600),'#172337')
    for i,view in enumerate(views):
        hit=h[i];v=np.full(hit.shape,17,np.uint8);mask=hit>=0;v[mask]=labels.reshape(-1)[hit[mask]]
        color=np.asarray(Image.fromarray(palette[v]).resize((704,256),Image.Resampling.NEAREST))
        mask=np.asarray(Image.fromarray((v!=17).astype(np.uint8)).resize((704,256),Image.Resampling.NEAREST)).astype(bool)
        rgb=np.array(view).copy();rgb[mask]=(rgb[mask]*.65+color[mask]*.35).astype(np.uint8)
        mosaic.paste(Image.fromarray(rgb).resize((352,128)),((i%3)*352,24+(i//3)*152))
        ImageDraw.Draw(mosaic).text(((i%3)*352+4,8+(i//3)*152),config['clips'][i]['camera'],font=font(13),fill='white')
    for j,z in enumerate([2,5]):
        mosaic.paste(Image.fromarray(palette[labels[::-1,::-1,z]]).resize((240,240),Image.Resampling.NEAREST),(24+j*288,342))
    ImageDraw.Draw(mosaic).text((624,372),'Full 3D raycast\n+ two BEV slices\n+ semantic labels',font=font(23),fill='white')
    mosaic_panel=panel();mosaic_panel.paste(fit(mosaic,(W,H)),(52,0))
    packets=bytes_panel('H.264 video carried over RTSP / RTP')
    elementary=bytes_panel('Depacketized H.264 access units','bytes')
    encoded=bytes_panel('Annotated mosaic encoded as H.264','bytes')
    webrtc=bytes_panel('Encoded H.264 delivered over WebRTC')
    browser=panel();screenshot=ROOT/'artifacts/fps-runs/4/browser.png'
    if screenshot.exists():
        thumb=fit(Image.open(screenshot).convert('RGB'),(W,H));browser.paste(thumb,((W-thumb.width)//2,0))
    else:browser=mosaic_panel
    paths=[]
    paths.append(page('01-rtsp-and-preprocessing.png','1 / 3   RTSP to camera tensors',
        'Read each row from left to right. Six camera streams form one synchronized group.',[
        (cameras,'Six prepared camera clips. Resize / crop is already applied offline.','Publish six RTSP streams','HOST CPU','FFmpeg loops H.264 clips; Insight provides the RTSP source endpoints.',packets,'Compressed video packets. Image content is still encoded.'),
        (packets,'Six independent RTSP/TCP connections.','Receive and parse','ARM CPU','Receive RTP, depacketize H.264 and parse encoded access units.',elementary,'H.264 elementary-stream frames, ready for hardware decoding.'),
        (elementary,'Compressed camera frames, 704 x 288 including the marker footer.','Decode six videos','VIDEO HARDWARE','Dedicated video decoder converts compressed H.264 into raw NV12 surfaces.',nv12,'Raw NV12: one Y plane plus interleaved, half-resolution UV.'),
        (nv12,'Six raw decoded surfaces with embedded frame identifiers.','Synchronize and normalize','ARM CPU','Match frame markers; remove footer; NV12 to BGR; subtract mean and divide by std.',normalized,'6 x 256 x 704 x 3 float32 values. Pictures illustrate their content, not their numeric format.')]))
    paths.append(page('02-inference-and-pooling.png','2 / 3   Camera tensors to 3D occupancy',
        'Heatmaps below summarize real device tensors. They are not extra model predictions.',[
        (normalized,'Six normalized camera tensors; the loaded model is reused per view.','Camera network x 6','MLA','Run sequential camera inference. Predict per-pixel depth probabilities and 64 learned features.',camera_outputs,'Per camera: depth [16,44,44 bins] and features [16,44,64]. Front-view summaries shown.'),
        (camera_outputs,'Depth probabilities + image features + calibration-derived rank arrays.','Depth-weighted BEV pooling','ARM NEON','Lift features using depth, accumulate them in BEV cells and fuse all six views. Eight workers.',bev_features,'One fused float32 BEV tensor: [200,200,64]. Color represents feature magnitude, not classes.'),
        (bev_features,'Fused bird\'s-eye-view features, not a road-map image.','BEV network + occupancy head','MLA','Reason over the fused features. Produce 18 class scores at each of 16 voxel heights.',score,'Logits [200,200,288] = 16 heights x 18 classes. One height\'s maximum score is visualized.'),
        (score,'18 raw scores for every voxel; the heatmap shows just one height.','Choose voxel classes','ARM NEON','Take argmax over 18 classes and restore upstream X,Y,Z layout.',volume,'200 x 200 x 16 class IDs: 640,000 voxels. The 3D plot samples occupied centres.')]))
    paths.append(page('03-rendering-and-insight.png','3 / 3   Occupancy to the Insight viewer',
        'Inference finishes on Modalix. Insight receives the already annotated video.',[
        (volume,'Full occupancy volume + synchronized camera images + calibrated camera rays.','Raycast and draw','ARM CPU','Find the nearest occupied voxel on each ray. Blend class colors onto six views and draw BEV slices.',mosaic_panel,'One annotated mosaic: six views, occupancy overlays and two height slices.'),
        (mosaic_panel,'1536 x 864 BGR visualization produced on Modalix.','Convert color and encode','ARM + VIDEO HW','CPU video conversion prepares NV12; dedicated video hardware encodes the mosaic as H.264.',encoded,'One compressed annotated H.264 stream. MLA is not used for video encoding.'),
        (encoded,'Compressed annotated video; overlays are already part of its pixels.','RTP / UDP to Insight','HOST / ARM CPU','ARM packetizes and transmits. Insight ingests RTP on the SDK host and forwards it through WebRTC.',webrtc,'Encoded video transport to the browser. Insight does not run FlashOCC.'),
        (webrtc,'WebRTC video packets reaching the browser.','Decode and display','BROWSER','Browser decodes H.264 and displays the mosaic. Hardware or software decode depends on the browser.',browser,'Actual Insight viewer screenshot, showing another frame from the same replay scene.')]))
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>FlashOCC: visual application flow</title>
<style>body{margin:0;background:#0e1624;color:#dce6f3;font:18px system-ui}main{max-width:1500px;margin:auto;padding:24px}img{width:100%;height:auto;border:1px solid #354763;margin:18px 0}a{color:#8dcaff}p{line-height:1.6}</style><main>
<h1>FlashOCC on Modalix: visual input / component / output</h1>
<p>Blue: ARM CPU / NEON. Purple: MLA. Amber: video hardware. Green: browser. All tensor views are derived from measured device outputs. Transport byte drawings and NV12 layout are explanatory illustrations.</p>'''
    for path in paths:html+=f'<a href="{path.name}"><img src="{path.name}" alt="{path.stem}"></a>'
    html+='<p>Neat handles casts, tessellation and transfers around MLA using AUTO placement (ARM / EV74 selected by the runtime). The diagram does not assign all of these conversions to MLA or EV74. The current prepared replay runs at 4 complete six-camera updates/s. No ground-truth accuracy is claimed.</p></main></html>'
    (OUT/'index.html').write_text(html)
    print('\n'.join(str(p) for p in paths));print(OUT/'index.html')

if __name__=='__main__':main()
