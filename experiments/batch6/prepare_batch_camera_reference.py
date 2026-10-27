"""Write FP32 per-view camera references for the eight saved six-camera holdouts."""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    root = _PROJECT_ROOT
    destination = root / 'artifacts/compile-batch6/reference'
    destination.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    model = ort.InferenceSession(str(root / 'artifacts/onnx-modalix/camera.onnx'), options,
                                providers=['CPUExecutionProvider'])
    manifest = json.loads((root / 'artifacts/app-replay-inputs/manifest.json').read_text())
    for sample in manifest['samples']:
        token = sample['token']
        results = []
        for camera in range(6):
            x = np.load(root / f'artifacts/real-calibration/camera/validation/{token}-{camera}.npy')
            results.append(model.run(None, {'camera_input': x}))
        for index, name in enumerate(['depth', 'features']):
            value = np.concatenate([r[index].transpose(0, 2, 3, 1) for r in results])
            value.astype('<f4').tofile(destination / (token + '.' + name + '.f32'))
        print('FP32 camera references', token, flush=True)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
