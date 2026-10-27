"""Summarize hardware camera accuracy and Neat batch benchmarks."""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
import json
from pathlib import Path

import numpy as np


def main():
    root = _PROJECT_ROOT
    output = root / 'artifacts/compile-batch6'
    report = {'reference_is_ground_truth': False, 'camera_precision': 'BF16',
              'bev_precision': 'BF16 with INT8 Softplus', 'variants': {}}
    report['runtime'] = json.loads((output / 'test-runtime.json').read_text())
    report['boundaries'] = json.loads((output / 'boundaries-comparison.json').read_text())
    report['deployment_decision'] = 'Retain batch one: batch-six FP32 fidelity degrades substantially'
    for label in ['batch1', 'batch6']:
        accuracy = json.loads((output / label / 'accuracy.json').read_text())
        benchmark = json.loads((output / label / 'benchmark/camera.json').read_text())
        rows = accuracy['samples']
        assert len(rows) == 8
        report['variants'][label] = {'resolved_batch': accuracy['batch'],
            'camera_calls_per_six_views': rows[0]['camera_calls'], 'groups': len(rows),
            'camera_depth_relative_rmse_range': [min(r['depth']['relative_rmse'] for r in rows), max(r['depth']['relative_rmse'] for r in rows)],
            'camera_feature_relative_rmse_range': [min(r['features']['relative_rmse'] for r in rows), max(r['features']['relative_rmse'] for r in rows)],
            'fp32_occupancy_label_agreement_range': [min(r['fp32_label_agreement'] for r in rows), max(r['fp32_label_agreement'] for r in rows)],
            'fp32_occupied_union_label_agreement_range': [min(r['occupied_union_label_agreement'] for r in rows), max(r['occupied_union_label_agreement'] for r in rows)],
            'benchmark': benchmark}
    comparisons = []
    for p in sorted((output / 'batch1').glob('*.labels.u8')):
        a = np.fromfile(p, np.uint8)
        b = np.fromfile(output / 'batch6' / p.name, np.uint8)
        assert a.size == b.size == 640000
        occupied = (a != 17) | (b != 17)
        comparisons.append({'token': p.name.removesuffix('.labels.u8'), 'batch1_vs_batch6_agreement': float(np.mean(a == b)),
                            'occupied_union_agreement': float(np.mean(a[occupied] == b[occupied])) if occupied.any() else 1.})
    report['batching_label_comparison'] = comparisons
    stock = output / 'stock-runtime/batch1'
    report['stock_vs_patched_batch1_labels_bit_exact'] = all(
        p.read_bytes() == (stock / p.name).read_bytes()
        for p in (output / 'batch1').glob('*.labels.u8'))
    b1 = report['variants']['batch1']['benchmark']
    b6 = report['variants']['batch6']['benchmark']
    report['benchmark_comparison'] = {
        'sequential_six_batch1_calls_ms': 6 * b1['latency_ms'],
        'six_camera_batch_latency_ms': b6['latency_ms'],
        'camera_group_latency_speedup': 6 * b1['latency_ms'] / b6['latency_ms'],
        'async_logical_image_throughput_ratio': b6['fps'] / b1['fps'],
        'scope': 'Neat model route with normalized FP32 tensor inputs; excludes image normalization, pooling, BEV and visualization',
    }
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['benchmark_comparison'], indent=2))
    for label, value in report['variants'].items():
        print(label, 'depth', value['camera_depth_relative_rmse_range'],
              'features', value['camera_feature_relative_rmse_range'],
              'FP32 labels', value['fp32_occupancy_label_agreement_range'])
    print('batch1 vs batch6 label agreement', [r['batch1_vs_batch6_agreement'] for r in comparisons])


if __name__ == '__main__':
    main()
