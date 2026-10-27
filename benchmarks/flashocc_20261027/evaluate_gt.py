"""Evaluate saved predictions with the repository's Occ3D camera-mask protocol."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gt-dir', type=Path, default=ROOT / 'gt')
parser.add_argument('--results', type=Path, default=ROOT / 'artifacts/qat-comparison-20261027')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
OUT = args.results.resolve()
names = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation', 'free']


def metrics(hist):
    union = hist.sum(0) + hist.sum(1) - hist.diagonal()
    iou = np.divide(hist.diagonal(), union, out=np.full(18, np.nan), where=union > 0)
    return {'camera_masked_accuracy': float(hist.trace() / hist.sum()),
            'miou_excluding_free': float(np.nanmean(iou[:17])),
            'voxels': int(hist.sum()),
            'per_class_iou': {n: float(v) if np.isfinite(v) else None for n, v in zip(names, iou)},
            'confusion_matrix': hist.tolist()}


files = sorted(args.gt_dir.glob('*.npz'))
assert len(files) == 32
report = {'ground_truth_accuracy_measured': True,
          'protocol': 'Occ3D mask_camera; aggregate 18x18 confusion matrix; '
                      'nanmean IoU of classes 0-16, excluding free class 17',
          'model_input': 'supplied pre-normalized image tensors and calibrated pooling ranks; '
                         'no RTSP codec or EV74 image normalization',
          'samples': len(files), 'models': {},
          'ground_truth_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
for mode in ['fp32', 'original', 'mixed', 'qat']:
    histogram = np.zeros((18, 18), dtype=np.int64)
    records = []
    for p in files:
        with np.load(p) as a:
            truth = a['semantics']
            mask = a['mask_camera'].astype(bool)
        pred = np.fromfile(OUT / 'gt-replay' / mode / (p.stem + '.labels.u8'), np.uint8)
        assert pred.size == truth.size == 640000 and mask.shape == truth.shape
        pred = pred.reshape(truth.shape)
        assert pred.max() < 18
        valid = mask & (truth < 18)
        hist = np.bincount(18 * truth[valid].astype(np.int64) + pred[valid],
                           minlength=324).reshape(18, 18)
        assert int(hist.sum()) == int(valid.sum())
        histogram += hist
        rec = metrics(hist)
        rec['token'] = p.stem
        rec.pop('confusion_matrix')
        records.append(rec)
    report['models'][mode] = {'summary': metrics(histogram), 'per_sample': records}
    print(mode, {k: v for k, v in metrics(histogram).items()
                 if k in ['camera_masked_accuracy', 'miou_excluding_free', 'voxels']})

sources = [p for p in [ROOT / 'accuracy-reports', *sorted((ROOT / 'accuracy_reports').glob('*.json'))]
           if p.is_file()]
report['supplied_reports'] = []
for source in sources:
    supplied = json.loads(source.read_text())
    if not {'summary', 'per_sample'} <= supplied.keys():
        continue
    supplied_tokens = {r['token'] for r in supplied['per_sample']}
    assert supplied_tokens == {p.stem for p in files}
    report['supplied_reports'].append({'path': str(source),
                                 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                 'summary': supplied['summary'],
                                 'archive_identity_recorded': False,
                                 'execution_backend_recorded': False})
    assert supplied['summary']['camera_masked']['voxels'] == report['models']['qat']['summary']['voxels']
    print('supplied report', supplied['summary']['camera_masked'])
output = args.output or OUT / 'ground-truth.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
