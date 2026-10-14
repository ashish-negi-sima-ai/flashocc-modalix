"""Plot FP32 and mixed-precision predictions after pipeline validation."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
import numpy as np
from export_flashocc import ROOT
from preview_real import COLORS


def main():
    report=json.loads((ROOT/'artifacts/compile-real-mixed/pipeline-validation.json').read_text())
    for sample in report['samples']:
        token=sample['token']
        reference=np.load(ROOT/'artifacts/real-predictions'/f'{token}.npy')[0]
        actual=np.load(ROOT/'artifacts/real-predictions'/f'{token}-compile-real-compile-real-mixed.npy')[0]
        fig,axes=plt.subplots(1,3,figsize=(13,4.7),layout='constrained')
        for ax,labels,title in zip(axes[:2],[reference,actual],['FP32 reference','BF16 with INT8 Softplus']):
            ax.imshow(labels[:,:,5].T,origin='lower',extent=(-40,40,-40,40),cmap=ListedColormap(COLORS),
                      norm=BoundaryNorm(np.arange(-.5,18.5),18),interpolation='nearest')
            ax.set_title(title);ax.scatter(0,0,c='black',marker='x')
        axes[2].imshow((reference[:,:,5]!=actual[:,:,5]).T,origin='lower',extent=(-40,40,-40,40),
                       cmap=ListedColormap(['white','#dc2626']),vmin=0,vmax=1,interpolation='nearest')
        axes[2].set_title('Changed labels in red')
        for ax in axes:ax.set_xlabel('Vehicle-frame x (m)');ax.set_ylabel('Vehicle-frame y (m)')
        fig.suptitle(f"{sample['scene']} — height 1.2 m; full-grid agreement {sample['all_voxel_label_agreement']:.2%}\nHost SDK comparison; no ground-truth accuracy or hardware performance claim",fontsize=12)
        path=ROOT/'artifacts/real-predictions'/f"{sample['scene']}-precision-comparison.png"
        fig.savefig(path,dpi=130);plt.close(fig);print(path)


if __name__=='__main__':main()
