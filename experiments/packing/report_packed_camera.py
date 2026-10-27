"""Summarize the on-device packed-camera comparison and Neat benchmarks."""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
import hashlib
import json
from pathlib import Path

import numpy as np


def summary(values):
    values = [float(v) for v in values]
    return {'min': min(values), 'max': max(values), 'mean': sum(values)/len(values)}


def main():
    root = _PROJECT_ROOT
    destination = root/'artifacts/compile-packed'
    variants = {}
    for name in ('batch1', 'packed'):
        directory = destination/'device'/name
        accuracy = json.loads((directory/'accuracy.json').read_text())
        benchmark = json.loads((directory/'benchmark/camera.json').read_text())
        archive = Path(accuracy['camera_model'])
        samples = accuracy['samples']
        variants[name] = {'camera_archive': str(archive),
                          'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                          'execution': accuracy['execution'],
                          'benchmark_cvu_pre_target': benchmark.get('cvu_pre_target', 'AUTO'),
                          'benchmark_cvu_post_target': benchmark.get('cvu_post_target', 'AUTO'),
                          'compiled_batch': accuracy['batch'],
                          'depth_relative_rmse': summary(s['depth']['relative_rmse'] for s in samples),
                          'features_relative_rmse': summary(s['features']['relative_rmse'] for s in samples),
                          'occupancy_agreement_vs_fp32': summary(s['fp32_label_agreement'] for s in samples),
                          'occupied_union_agreement_vs_fp32': summary(s['occupied_union_label_agreement'] for s in samples),
                          'camera_with_input_prepare_ms': summary(s['camera_with_input_prepare_ms'] for s in samples),
                          'input_prepare_ms': summary(s['input_prepare_ms'] for s in samples),
                          'output_unpack_ms': summary(s['output_unpack_ms'] for s in samples),
                          'neat_latency_per_call_ms': benchmark['latency_ms'],
                          'neat_six_camera_latency_ms': benchmark['latency_ms']*(6 if name == 'batch1' else 1),
                          'neat_camera_images_per_second': benchmark['camera_images_per_second'],
                          'neat_six_camera_groups_per_second': benchmark['six_camera_groups_per_second'],
                          'samples': samples}
    first, packed = variants['batch1'], variants['packed']
    assert {s['token'] for s in first['samples']} == {s['token'] for s in packed['samples']}
    pairwise = []
    for sample in first['samples']:
        token = sample['token']
        record = {'token': token}
        for head in ('depth', 'features'):
            a, b = [np.fromfile(destination/'device'/name/f'{token}.{head}.f32', '<f4')
                    for name in ('batch1', 'packed')]
            difference = a.astype(np.float64)-b
            record[head+'_relative_rmse_vs_batch1'] = float(np.linalg.norm(difference)/np.linalg.norm(a))
            record[head+'_bit_exact_vs_batch1'] = bool(np.array_equal(a,b))
        a, b = [np.fromfile(destination/'device'/name/f'{token}.labels.u8', np.uint8)
                for name in ('batch1', 'packed')]
        record['occupancy_agreement_vs_batch1'] = float(np.mean(a == b))
        pairwise.append(record)
    report = {'comparison': 'BF16 spatial packing versus six serial BF16 calls using installed Neat',
              'ground_truth_accuracy_measured': False, 'camera_images': len(pairwise)*6,
              'variants': variants, 'pairwise': pairwise,
              'six_camera_neat_latency_speedup': first['neat_six_camera_latency_ms']/packed['neat_six_camera_latency_ms'],
              'neat_throughput_speedup': packed['neat_camera_images_per_second']/first['neat_camera_images_per_second'],
              'all_depth_features_labels_bit_exact_to_batch1': all(
                  r['depth_bit_exact_vs_batch1'] and r['features_bit_exact_vs_batch1']
                  and r['occupancy_agreement_vs_batch1']==1.0 for r in pairwise),
              'timing_scope': 'Neat benchmark excludes packing/unpacking, pooling, BEV, RTSP and visualization; replay timings include camera input preparation and output unpacking'}
    (destination/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('variants', 'pairwise')}, indent=2), flush=True)
    for name, value in variants.items():
        print(name, json.dumps({k: v for k, v in value.items() if k != 'samples'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
