"""Generate compiler-owned MLA inputs and output references for one real camera group."""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np


def main():
    root = _PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=6)
    parser.add_argument('--token', default='3e8750f331d7499e9b5123e9eb70f2e2')
    args = parser.parse_args()
    if args.batch not in (1, 6):
        parser.error('--batch must be 1 or 6')
    directory = root / f'artifacts/compile-batch6/diagnostics/compiler-reference-batch{args.batch}'
    directory.mkdir(parents=True, exist_ok=True)
    source = root / 'artifacts/compile-real/camera'
    audit = json.loads((root / 'artifacts/onnx-modalix/camera.bfloat16.audit.json').read_text())
    if audit['unsupported'] or audit['unknown']:
        raise RuntimeError('BF16 operator audit failed')
    inputs = []
    for camera in range(6):
        path = root / f'artifacts/real-calibration/camera/validation/{args.token}-{camera}.npy'
        value = np.load(path, allow_pickle=False)
        if value.shape != (1, 3, 256, 704) or value.dtype != np.float32:
            raise ValueError(f'Unexpected input tensor: {path}')
        inputs.append({'camera_input': value.transpose(0, 2, 3, 1).copy()})
    record = {'token': args.token, 'batch': args.batch, 'input_count': len(inputs),
              'source_sha256': hashlib.sha256((source / 'flashocc_m0_camera.sima').read_bytes()).hexdigest(),
              'completed': False}
    (directory / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
    from afe.apis.model import Model
    model = Model.load('flashocc_m0_camera', str(source))
    model.generate_elf_and_reference_files(inputs, str(directory), batch_size=args.batch,
                                          log_level=logging.INFO)
    record.update(completed=True, files=sorted(str(p.relative_to(directory))
                                              for p in directory.rglob('*') if p.is_file()))
    (directory / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
