"""Independently pack the first holdout group's FP32 images for direct MLA execution."""
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / 'artifacts/compile-batch6'
    destination = base / 'diagnostics/direct-transport'
    destination.mkdir(parents=True, exist_ok=True)
    token = json.loads((base / 'batch1/accuracy.json').read_text())['samples'][0]['token']
    outputs = []
    report = []
    for camera in range(6):
        image = np.load(root / f'artifacts/real-calibration/camera/validation/{token}-{camera}.npy')
        values = image.transpose(0, 2, 3, 1).copy()[0]
        bits = values.view(np.uint32)
        rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
        tiles = []
        for h in range(0, 256, 13):
            for w in range(0, 704, 141):
                tile = rounded[h:h + 13, w:w + 141]
                lanes = np.stack([(tile & 255).astype(np.uint8), (tile >> 8).astype(np.uint8)], axis=-2)
                payload = lanes.tobytes()
                tiles.append(payload + bytes((-len(payload)) % 16))
        packed = b''.join(tiles)
        assert len(packed) == 1082272
        (destination / f'camera-{camera}.bin').write_bytes(packed)
        captured = (base / 'batch1' / f'premla-{camera}-0.bin').read_bytes()
        report.append({'camera': camera, 'packed_bytes': len(packed), 'captured_bytes': len(captured),
                       'captured_prefix_matches': packed[:len(captured)] == captured})
        outputs.append(packed)
    (destination / 'six-cameras.bin').write_bytes(b''.join(outputs))
    (destination / 'packing.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not all(r['captured_prefix_matches'] for r in report):
        raise RuntimeError('Independent input packing differs from the captured input')


if __name__ == '__main__':
    main()
