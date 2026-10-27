"""Verify retained benchmark assets and independently recalculate the saved tables."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, default=ROOT)
    parser.add_argument('--records-only', action='store_true', help='Recalculate saved evidence without large assets')
    args = parser.parse_args()
    baseline = json.loads((HERE / 'baseline.json').read_text())
    measurements = json.loads((HERE / 'measurements.json').read_text())
    for name, data in measurements['performance'].items():
        reference = baseline['performance'][name]
        rows = data['frames'][reference['warmup_groups']:]
        if len(rows) != 192 or data['total_processed'] != 256 or data['signal'] != 0:
            raise ValueError(f'Incomplete reference run: {name}')
        elapsed_seconds = (rows[-1]['completed_ms'] - rows[0]['completed_ms']) / 1000
        fps = (len(rows) - 1) / elapsed_seconds
        if not math.isclose(fps, reference['output_fps'], rel_tol=1e-12):
            raise ValueError(f'FPS mismatch: {name}')
        for key, value in reference['mean_ms'].items():
            if key == 'other_ms':
                continue
            actual = statistics.mean(r[key] for r in rows)
            if not math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'Timing mismatch: {name}/{key}')
        components = ['preparation_ms', 'camera_input_ms', 'geometry_plan_ms', 'camera_six_views_ms',
                      'pooling_ms', 'bev_input_ms', 'bev_inference_ms', 'argmax_ms',
                      'logit_finite_check_ms', 'render_ms', 'volume_publish_ms']
        other = statistics.mean(r['application_ms'] for r in rows) - sum(
            statistics.mean(r[key] for r in rows) for key in components)
        if not math.isclose(other, reference['mean_ms']['other_ms'], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f'Other-work residual mismatch: {name}')
        print(f'{name}: {fps:.3f} FPS; saved means reproduced')
    for name, data in measurements['accuracy']['models'].items():
        hist = data['summary']['confusion_matrix']
        total = sum(map(sum, hist))
        correct = sum(hist[i][i] for i in range(18))
        ious = []
        for i in range(17):
            union = sum(hist[i]) + sum(row[i] for row in hist) - hist[i][i]
            if union:
                ious.append(hist[i][i] / union)
        miou = statistics.mean(ious)
        reference = baseline['accuracy'][name]
        if total != 2153160 or not math.isclose(miou, reference['miou_excluding_free'], rel_tol=1e-12):
            raise ValueError(f'mIoU mismatch: {name}')
        if not math.isclose(correct / total, reference['camera_masked_accuracy'], rel_tol=1e-12):
            raise ValueError(f'Accuracy mismatch: {name}')
        print(f'{name}: {100*miou:.4f}% mIoU; confusion-matrix result reproduced')
    if args.records_only:
        print('Saved records verified; asset availability was not checked.')
        return
    manifest = json.loads((HERE / 'assets.json').read_text())
    failures = []
    for entry in manifest['files']:
        path = args.asset_root / entry['path']
        if not path.is_file():
            failures.append('Missing: ' + entry['path'])
            continue
        if path.stat().st_size != entry['size_bytes']:
            failures.append('Size mismatch: ' + entry['path'])
            continue
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != entry['sha256']:
            failures.append('Checksum mismatch: ' + entry['path'])
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'All {len(manifest["files"])} required asset files verified.')


if __name__ == '__main__':
    main()
