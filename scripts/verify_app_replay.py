"""Independently check device tensor layout, pooling, and occupancy labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from pool_reference import pool

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT/'artifacts/app-device-final')
    args = parser.parse_args()
    summary = json.loads((args.results/'summary.json').read_text())
    checks = []
    for token in sorted({r['token'] for r in summary['frames']}):
        depth = np.fromfile(args.results/f'{token}.depth.f32', np.float32).reshape(6,16,44,44).transpose(0,3,1,2).copy()
        features = np.fromfile(args.results/f'{token}.features.f32', np.float32).reshape(6,16,44,64).transpose(0,3,1,2).copy()
        with np.load(ROOT/'artifacts/real-inputs/validation'/f'{token}.npz') as inp:
            expected = pool(depth, features, inp['ranks_depth'], inp['ranks_feat'], inp['ranks_bev']).transpose(0,2,3,1).reshape(200,200,64)
        actual = np.fromfile(args.results/f'{token}.pooled.f32', np.float32).reshape(expected.shape)
        np.testing.assert_array_equal(actual, expected)
        logits = np.fromfile(args.results/f'{token}.logits.f32', np.float32).reshape(200,200,16,18)
        assert np.isfinite(logits).all()
        labels = np.fromfile(args.results/f'{token}.labels.u8', np.uint8).reshape(200,200,16)
        np.testing.assert_array_equal(labels, logits.argmax(-1).transpose(1,0,2))
        checks.append(dict(token=token, pooling_exact=True, labels_exact=True, finite_logits=True))
    report = dict(passed=True, frames_checked=len(checks), checks=checks)
    (args.results.parent/'app-device-verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
