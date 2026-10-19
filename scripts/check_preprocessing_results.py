"""Compare device preprocessing fixtures and summarize timing/fidelity."""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts/preprocess-validation')
    parser.add_argument('--variants', nargs='+', default=['ev74-bgr', 'ev74-nv12'])
    args = parser.parse_args()
    baseline = args.directory / 'arm-fp32'
    report = {'reference': 'Current ARM preprocessing + BF16 models on identical decoded NV12 inputs',
              'ground_truth_accuracy': False, 'variants': {}}
    for mode in ['arm-fp32'] + args.variants:
        directory = args.directory / mode
        summary = json.loads((directory / 'summary.json').read_text())
        records = summary['frames']
        entry = {'execution': summary['execution'], 'measured_groups': len(records),
                 'median_ms': {key: statistics.median(row[key] for row in records)
                               for key in ['preparation_ms', 'camera_six_views_ms', 'compute_ms', 'check_total_ms']},
                 'samples': []}
        for path in sorted(directory.glob('*.labels.u8')):
            token = path.name.removesuffix('.labels.u8')
            ref = np.fromfile(baseline / path.name, np.uint8)
            actual = np.fromfile(path, np.uint8)
            assert ref.size == actual.size == 640000
            occupied = (ref != 17) | (actual != 17)
            sample = {'token': token, 'label_agreement': float(np.mean(ref == actual)),
                      'occupied_union_label_agreement': float(np.mean(ref[occupied] == actual[occupied])),
                      'occupancy_binary_agreement': float(np.mean((ref == 17) == (actual == 17))), 'tensors': {}}
            for tensor in ['depth', 'features', 'pooled', 'logits']:
                name = token + '.' + tensor + '.f32'
                reference = np.fromfile(baseline / name, np.float32)
                value = np.fromfile(directory / name, np.float32)
                assert value.shape == reference.shape and np.isfinite(value).all()
                sample['tensors'][tensor] = {'relative_rmse': float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-30)),
                                             'max_absolute_error': float(np.max(np.abs(value - reference)))}
            entry['samples'].append(sample)
        assert entry['samples'], 'No dumped fixture predictions'
        report['variants'][mode] = entry
    (args.directory / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    for name, value in report['variants'].items():
        print(name, 'median ms', value['median_ms'],
              'label agreement', [round(s['label_agreement'] * 100, 3) for s in value['samples']])


if __name__ == '__main__':
    main()
