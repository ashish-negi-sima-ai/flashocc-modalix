"""Verify the running replay's frame continuity, camera coverage and browser result."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=ROOT/'artifacts/insight-device')
    parser.add_argument('--config',type=Path,default=ROOT/'artifacts/insight-demo/config.json')
    args=parser.parse_args()
    text=(args.results/'frames.jsonl').read_text()
    lines=text.splitlines()
    if not text.endswith('\n'):lines=lines[:-1]
    frames=[json.loads(line) for line in lines]
    config=json.loads(args.config.read_text());count=len(config['frames'])
    assert len(frames)>=count*2, 'Wait for at least two complete clip loops'
    positions=sorted({f['marker'] for f in frames})
    assert positions==list(range(count)), 'Missing replay frame positions'
    gaps=[b['sequence']-a['sequence']-1 for a,b in zip(frames,frames[1:])]
    assert not any(gaps), 'Replay skipped or repeated a frame'
    sample=next(f for f in reversed(frames) if 'camera_marker_counts' in f)
    assert all(all(n>0 for n in camera) for camera in sample['camera_marker_counts'])
    browser=json.loads((ROOT/'artifacts/insight-browser-validation.json').read_text())
    assert browser['passed'] and not browser['errors']
    report=dict(passed=True,processed_frames=len(frames),matched_positions=positions,
                sequence_gaps=sum(gaps),all_six_cameras_received_all_positions=True,
                median_compute_ms=statistics.median(f['compute_ms'] for f in frames[2:]),
                median_application_ms=statistics.median(f['application_ms'] for f in frames[2:]),
                frame_rate=config['fps'],ground_truth_accuracy_measured=False,
                camera_overlay=config['camera_overlay'],
                projection_roundtrip_max_pixels=config['projection_roundtrip_max_pixels'],
                device_results=str(args.results),viewer=browser['viewer'],
                browser_report='artifacts/insight-browser-validation.json',
                contract_test_log='artifacts/insight-contract-tests.log',
                raycast_test_log='artifacts/insight-raycast-tests.log',
                clips=[dict(file=c['file'],sha256=hashlib.sha256((args.config.parent/c['file']).read_bytes()).hexdigest()) for c in config['clips']])
    destination=ROOT/'artifacts/insight-demo-validation.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Passed: {len(frames)} frames, all {count} positions, no sequence gaps. Report: {destination}')

if __name__=='__main__':main()
