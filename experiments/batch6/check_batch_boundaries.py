"""Independently decode captured camera MLA outputs and compare batch boundaries."""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
import json
import tarfile
from pathlib import Path

import numpy as np


def decode(raw, channels, tile):
    """Decode this model's BF16 C16 tiles into natural HWC order."""
    result = np.empty((16, 44, channels), np.float32)
    th, tw, tc = tile
    offset = 0
    for c in range(0, channels, tc):
        cc = min(tc, channels - c)
        stored = (cc + 15) // 16 * 16
        for h in range(0, 16, th):
            hh = min(th, 16 - h)
            for w in range(0, 44, tw):
                ww = min(tw, 44 - w)
                size = hh * ww * stored * 2
                data = np.frombuffer(raw[offset:offset + size], np.uint8)
                # MLA BF16 C16 blocks store low/high bytes in separate lanes.
                lanes = data.reshape(stored // 16, hh, ww, 2, 16).astype(np.uint32)
                bits = (lanes[..., 0, :] | (lanes[..., 1, :] << 8)).transpose(1, 2, 0, 3)
                bits = bits.reshape(hh, ww, stored)[..., :cc]
                result[h:h + hh, w:w + ww, c:c + cc] = (bits << 16).view(np.float32)
                offset += (size + 15) // 16 * 16
    if offset != len(raw):
        raise ValueError(f'Decoder consumed {offset} bytes, received {len(raw)}')
    return result


def main():
    root = _PROJECT_ROOT
    output = root / 'artifacts/compile-batch6'
    b6 = (output / 'batch6/premla-0-0.bin').read_bytes()
    with tarfile.open(output / 'camera/flashocc_m0_camera_mpk.tar.gz') as package:
        manifest = json.load(package.extractfile('flashocc_m0_camera_mpk.json'))
    tess = next(p for p in manifest['plugins'] if (p.get('config_params') or {}).get('kernel') == 'tessellation_transform')
    frame_size = tess['output_nodes'][0]['size'] // 6
    prefixes = []
    for camera in range(6):
        b1 = (output / 'batch1' / f'premla-{camera}-0.bin').read_bytes()
        batched = b6[camera * frame_size:(camera + 1) * frame_size]
        count = min(len(b1), len(batched))
        prefixes.append({'camera': camera, 'compared_bytes': count,
                         'equal': count > 0 and b1[:count] == batched[:count]})
    # Tensor public mappings clip to logical bytes even for a tiled transport.
    # Compare only captured prefixes at the MPK's physical per-frame stride.
    report = {'preprocess_transport_captured_prefixes_equal': all(p['equal'] for p in prefixes),
              'preprocess_capture_is_complete': len(b6) == 6 * frame_size,
              'preprocess_physical_frame_bytes': frame_size,
              'preprocess_prefix_comparisons': prefixes, 'decoded_outputs': []}
    for label, archive in [('batch1', root / 'artifacts/compile-real/camera/flashocc_m0_camera_mpk.tar.gz'),
                           ('batch6', output / 'camera/flashocc_m0_camera_mpk.tar.gz')]:
        accuracy = json.loads((output / label / 'accuracy.json').read_text())
        batch = accuracy['batch']
        token = accuracy['samples'][0]['token']
        with tarfile.open(archive) as package:
            manifest = json.load(package.extractfile('flashocc_m0_camera_mpk.json'))
        stages = [p for p in manifest['plugins'] if (p.get('config_params') or {}).get('kernel') == 'detessellation_transform']
        for head, (name, channels) in enumerate([('depth', 44), ('features', 64)]):
            stage = stages[head]
            frame_bytes = stage['input_nodes'][0]['size'] // batch
            outputs = []
            for group in range(6 // batch):
                data = (output / label / f'rawmla-{group}-{head}.bin').read_bytes()
                assert len(data) == batch * frame_bytes
                for camera in range(batch):
                    outputs.append(decode(data[camera * frame_bytes:(camera + 1) * frame_bytes], channels,
                                          stage['config_params']['params']['slice_shape']))
            decoded = np.stack(outputs)
            actual = np.fromfile(output / label / f'{token}.{name}.f32', np.float32).reshape(decoded.shape)
            report['decoded_outputs'].append({'variant': label, 'head': name,
                'bit_exact_match': bool(np.array_equal(decoded.view(np.uint32), actual.view(np.uint32))),
                'max_absolute_error': float(np.max(np.abs(decoded - actual)))})
    (output / 'boundaries-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['preprocess_transport_captured_prefixes_equal'] or not all(r['bit_exact_match'] for r in report['decoded_outputs']):
        raise RuntimeError('Batch boundary validation failed')


if __name__ == '__main__':
    main()
