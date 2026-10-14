"""Show real camera views and FP32 occupancy slices; predictions, not ground truth."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
from export_flashocc import ROOT

CLASSES = ['other', 'barrier', 'bicycle', 'bus', 'car', 'construction vehicle',
           'motorcycle', 'pedestrian', 'traffic cone', 'trailer', 'truck', 'road',
           'other flat', 'sidewalk', 'terrain', 'structure', 'vegetation', 'free']
COLORS = ['#777777','#ff9e00','#ff61ff','#ffff00','#008cff','#00ffff',
          '#ff00ff','#ff0000','#ffc800','#6496fa','#50ffff','#96324b',
          '#ff0096','#af004b','#4b004b','#c8c8c8','#00af00','#ffffff']


def main():
    manifest = json.loads((ROOT/'artifacts/real-inputs/manifest.json').read_text())
    samples = [r for r in manifest['samples'] if r['split']=='validation']
    for scene in sorted({r['scene'] for r in samples}):
        sample = next(r for r in samples if r['scene']==scene)
        labels = np.load(ROOT/'artifacts/real-predictions'/f"{sample['token']}.npy")[0]
        fig = plt.figure(figsize=(14,11),layout='constrained')
        grid = fig.add_gridspec(4,3,height_ratios=[1,1,2,0.35])
        for i,(camera,path) in enumerate(zip(manifest['camera_order'],sample['image_paths'])):
            ax=fig.add_subplot(grid[i//3,i%3])
            with Image.open(ROOT/'artifacts/data/nuscenes-mini'/path) as image:
                image.thumbnail((600,338)); ax.imshow(image)
            ax.set_title(camera,fontsize=9); ax.axis('off')
        for i,z in enumerate([2,5]):
            ax=fig.add_subplot(grid[2,i])
            ax.imshow(labels[:,:,z].T,origin='lower',extent=(-40,40,-40,40),
                      cmap=ListedColormap(COLORS),norm=BoundaryNorm(np.arange(-.5,18.5),18),interpolation='nearest')
            ax.scatter(0,0,c='black',s=20,marker='x')
            ax.set_title(f'Predicted height slice: z = {-1+(z+.5)*.4:.1f} m',fontsize=10)
            ax.set_xlabel('Vehicle-frame x (m)');ax.set_ylabel('Vehicle-frame y (m)')
        ax=fig.add_subplot(grid[2,2]);ax.axis('off')
        occupied = labels!=17
        ax.text(0,.95,'FP32 FlashOCC M0 predictions\n\nNo occupancy ground truth is loaded.\nThese plots do not measure accuracy.\n\n'+
                f"Scene: {scene}\nOccupied voxel fraction: {occupied.mean():.1%}",va='top',fontsize=10)
        ax=fig.add_subplot(grid[3,:]);ax.axis('off')
        ax.legend(handles=[Patch(facecolor=c,edgecolor='#666666',label=n) for c,n in zip(COLORS,CLASSES)],
                  loc='center',ncol=6,fontsize=8,frameon=False)
        fig.suptitle(f'Six-camera occupancy reference — {scene}',fontsize=15)
        destination=ROOT/'artifacts/real-predictions'/f'{scene}-preview.png'
        fig.savefig(destination,dpi=130);plt.close(fig)
        print(destination)


if __name__=='__main__':
    main()
