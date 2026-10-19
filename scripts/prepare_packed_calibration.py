"""Group existing normalized calibration views into guarded six-camera tensors."""
import json
from pathlib import Path

import numpy as np

from prepare_packed_camera import pack_images


def main():
    root = Path(__file__).resolve().parents[1]
    report = {}
    for split in ('calibration', 'validation'):
        source = root/'artifacts/real-calibration/camera'/split
        destination = root/'artifacts/real-calibration-packed/camera'/split
        destination.mkdir(parents=True, exist_ok=True)
        tokens = sorted({p.stem.rsplit('-', 1)[0] for p in source.glob('*.npy')})
        for token in tokens:
            views = np.concatenate([np.load(source/f'{token}-{i}.npy', allow_pickle=False)
                                    for i in range(6)])
            if views.shape != (6, 3, 256, 704) or views.dtype != np.float32:
                raise ValueError(f'Invalid saved camera input: {token}')
            np.save(destination/f'{token}.npy', pack_images(views))
        report[split] = {'groups': len(tokens), 'cameras': 6*len(tokens),
                         'shape_nchw': [1, 3, 256, 4384], 'source': str(source)}
    path = root/'artifacts/real-calibration-packed/manifest.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
