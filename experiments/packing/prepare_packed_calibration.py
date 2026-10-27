"""Group normalized calibration views into guarded camera tensors."""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
import argparse
import json
from pathlib import Path

import numpy as np

from prepare_packed_camera import pack_images


def main():
    root = _PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cameras', type=int, choices=[2, 3, 6], default=6)
    parser.add_argument('--guard-width', type=int, default=32)
    parser.add_argument('--output-dir', type=Path, help='default: artifacts/real-calibration-packed, or real-calibration-packedN')
    args = parser.parse_args()
    if args.guard_width < 32 or args.guard_width % 32:
        parser.error('Guard width must be a positive multiple of 32')
    if args.output_dir is None:
        args.output_dir = root/'artifacts'/('real-calibration-packed' if args.cameras == 6 else f'real-calibration-packed{args.cameras}')
    report = {}
    for split in ('calibration', 'validation'):
        source = root/'artifacts/real-calibration/camera'/split
        destination = args.output_dir/'camera'/split
        destination.mkdir(parents=True, exist_ok=True)
        tokens = sorted({p.stem.rsplit('-', 1)[0] for p in source.glob('*.npy')})
        for token in tokens:
            views = np.concatenate([np.load(source/f'{token}-{i}.npy', allow_pickle=False)
                                    for i in range(6)])
            if views.shape != (6, 3, 256, 704) or views.dtype != np.float32:
                raise ValueError(f'Invalid saved camera input: {token}')
            for first in range(0, 6, args.cameras):
                suffix = '' if args.cameras == 6 else f'-group{first//args.cameras}'
                np.save(destination/f'{token}{suffix}.npy',
                        pack_images(views[first:first+args.cameras], args.guard_width))
        report[split] = {'groups': len(tokens)*6//args.cameras, 'cameras': 6*len(tokens),
                         'cameras_per_input': args.cameras,
                         'shape_nchw': [1, 3, 256, 704*args.cameras+args.guard_width*(args.cameras-1)],
                         'source': str(source)}
    path = args.output_dir/'manifest.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
