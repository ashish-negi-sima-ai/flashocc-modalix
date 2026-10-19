"""Compare our packed camera exports with separate FP32 inference on 48 views."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from prepare_packed_camera import pack_images, unpack_output


def error(actual, expected):
    difference = actual.astype(np.float64) - expected
    return {'bit_exact': bool(np.array_equal(actual, expected)),
            'relative_rmse': float(np.linalg.norm(difference) / np.linalg.norm(expected)),
            'max_abs_error': float(np.abs(difference).max())}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=root/'artifacts/onnx-packed')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    metadata = json.loads((args.model_dir/'packing.json').read_text())
    if metadata['cameras'] != 6:
        parser.error('The saved validation groups contain exactly six cameras')
    for name, checksum in metadata['files_sha256'].items():
        if hashlib.sha256((args.model_dir/name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Export changed since packing: {name}')
    if hashlib.sha256(Path(metadata['source']).read_bytes()).hexdigest() != metadata['source_sha256']:
        raise ValueError('Original model changed since packing')
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads

    def session(path):
        return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])

    original = session(metadata['source'])
    scene = session(args.model_dir/'camera.onnx')
    spatial = session(args.model_dir/'camera_spatial.onnx')
    manifest = json.loads((root/'artifacts/app-replay-inputs/manifest.json').read_text())
    records = []
    first_input = first_outputs = None
    report = {'comparison': 'Packed versus six separate runs of our original FP32 camera ONNX',
              'atol': 1e-6, 'rtol': 1e-5, 'samples': records,
              'compiled': False, 'hardware_validated': False, 'ground_truth_accuracy_measured': False}
    for sample in manifest['samples']:
        token = sample['token']
        views = np.concatenate([np.load(root/f'artifacts/real-calibration/camera/validation/{token}-{i}.npy')
                                for i in range(6)])
        reference_views = [original.run(None, {'camera_input': view[None]}) for view in views]
        expected = [np.concatenate([v[i] for v in reference_views]) for i in range(2)]
        actual = scene.run(None, {'camera_input': views[None]})
        direct = spatial.run(None, {'camera_input': pack_images(views, metadata['input_guard_width'])})
        record = {'token': token, 'outputs': {}}
        for i, name in enumerate(('depth', 'features')):
            np.testing.assert_array_equal(actual[i], direct[i])
            unpacked = unpack_output(actual[i], guard_width=metadata['input_guard_width'])
            np.testing.assert_allclose(unpacked, expected[i], atol=report['atol'], rtol=report['rtol'])
            nhwc = unpack_output(actual[i].transpose(0, 2, 3, 1),
                                 guard_width=metadata['input_guard_width'], layout='NHWC')
            assert nhwc.flags.c_contiguous and unpacked.flags.c_contiguous
            np.testing.assert_array_equal(nhwc, unpacked.transpose(0, 2, 3, 1))
            if name == 'depth':
                np.testing.assert_allclose(unpacked.sum(axis=1), 1, atol=1e-6, rtol=0)
            record['outputs'][name] = error(unpacked, expected[i])
            record['outputs'][name]['per_camera'] = [error(unpacked[j], expected[i][j]) for j in range(6)]
        records.append(record)
        if first_input is None:
            first_input, first_outputs = views.copy(), [x.copy() for x in actual]
        print(json.dumps({'token': token, 'depth': record['outputs']['depth']['max_abs_error'],
                          'features': record['outputs']['features']['max_abs_error']}), flush=True)
    # Check camera independence directly, in addition to the six-view references.
    first_input[1:] = 0
    isolated = scene.run(None, {'camera_input': first_input[None]})
    for i in range(2):
        np.testing.assert_array_equal(unpack_output(isolated[i], guard_width=metadata['input_guard_width'])[0],
                                      unpack_output(first_outputs[i], guard_width=metadata['input_guard_width'])[0])
    report.update(passed=True, groups=len(records), camera_images=6*len(records),
                  camera0_unchanged_when_others_zero=True, scene_vs_spatial_bit_exact=True,
                  nchw_nhwc_unpack_equivalent=True, depth_probabilities_normalized=True,
                  all_outputs_bit_exact=all(x['bit_exact'] for r in records for x in r['outputs'].values()))
    (args.model_dir/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    metadata.update(fp32_validated=True, validation_report='validation.json',
                    validation_groups=len(records), validation_camera_images=6*len(records))
    (args.model_dir/'packing.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
