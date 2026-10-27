"""Regenerate the preserved FlashOCC accuracy and full-application comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
from urllib.parse import urlsplit

from runtime_config import ROOT, defaults, require

REFERENCE = ROOT / 'benchmarks/flashocc_20261027'


def run(command, log=None):
    print(shlex.join(map(str, command)), flush=True)
    if log:
        with log.open('w') as stream:
            subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(list(map(str, command)), check=True)


def summarize(results):
    report = {'accuracy': {}, 'performance': {}}
    gt = results / 'ground-truth.json'
    if gt.exists():
        report['accuracy'] = {n: v['summary'] for n, v in json.loads(gt.read_text())['models'].items()}
    baseline = json.loads((REFERENCE / 'baseline.json').read_text())
    for name in baseline['performance']:
        path = results / 'live' / name / '12/summary.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        rows = data['frames'][64:]
        if data['signal'] or data['total_processed'] != 256 or len(rows) != 192 or any(data['invalid_markers']):
            raise ValueError(f'Incomplete or invalid live run: {name}')
        elapsed = (rows[-1]['completed_ms'] - rows[0]['completed_ms']) / 1000
        means = {key: statistics.mean(r[key] for r in rows)
                 for key in baseline['performance'][name]['mean_ms'] if key != 'other_ms'}
        components = ['preparation_ms', 'camera_input_ms', 'geometry_plan_ms', 'camera_six_views_ms',
                      'pooling_ms', 'bev_input_ms', 'bev_inference_ms', 'argmax_ms',
                      'logit_finite_check_ms', 'render_ms', 'volume_publish_ms']
        means['other_ms'] = means['application_ms'] - sum(means[k] for k in components)
        report['performance'][name] = {'output_fps': (len(rows)-1)/elapsed, 'mean_ms': means}
    (results / 'comparison-report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name, value in report['accuracy'].items():
        print(f'{name}: mIoU {100*value["miou_excluding_free"]:.4f}%, accuracy {100*value["camera_masked_accuracy"]:.4f}%')
    for name, value in report['performance'].items():
        print(f'{name}: {value["output_fps"]:.3f} FPS; mean application {value["mean_ms"]["application_ms"]:.3f} ms')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['accuracy', 'performance', 'summarize'])
    parser.add_argument('--asset-root', type=Path, default=ROOT)
    parser.add_argument('--results', type=Path, default=ROOT / 'artifacts/comparison-repeat')
    parser.add_argument('--device', default=defaults()['device'])
    parser.add_argument('--api', default=defaults()['insight_api'])
    parser.add_argument('--python', default=sys.executable, help='Python with NumPy, PyTorch and ONNX Runtime')
    parser.add_argument('--replay-binary', type=Path, default=ROOT / 'build/modalix/flashocc_replay')
    parser.add_argument('--insight-binary', type=Path, default=ROOT / 'build/modalix/flashocc_insight')
    parser.add_argument('--include-scan-control', action='store_true')
    parser.add_argument('--rtsp-host')
    parser.add_argument('--rtsp-port', type=int, default=8554)
    parser.add_argument('--browser-python', type=Path, default=defaults()['browser_python'])
    args = parser.parse_args()
    args.asset_root = args.asset_root.resolve()
    args.results = args.results.resolve()
    args.results.mkdir(parents=True, exist_ok=True)
    args.results.chmod(0o777)
    if args.action == 'summarize':
        summarize(args.results)
        return
    component = args.results / ('gt-replay' if args.action == 'accuracy' else 'live')
    if component.exists() and any(component.iterdir()):
        parser.error(f'{component} already contains a run; use a fresh results directory')
    require(parser, args.device, '--device', 'FLASHOCC_DEVICE')
    run([sys.executable, ROOT / 'scripts/comparison_assets.py', 'verify', '--asset-root', args.asset_root])
    catalog = json.loads((ROOT / 'models/manifest.json').read_text())
    models = catalog['models']
    if args.action == 'accuracy':
        pair = catalog['pairs']['fp32']
        run([args.python, REFERENCE / 'predict_fp32.py', '--gt-dir', args.asset_root / 'gt',
             '--camera-model', args.asset_root / models[pair[0]]['path'],
             '--bev-model', args.asset_root / models[pair[1]]['path'], '--results', args.results,
             '--replay-input-dir', args.results / 'gt-inputs'])
        expected_frames = REFERENCE / 'replay-inputs.json'
        if expected_frames.exists():
            for row in json.loads(expected_frames.read_text())['samples']:
                frame = args.results / 'gt-inputs' / (row['token'] + '.frame')
                with frame.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                        raise ValueError('Regenerated replay frame differs from reference: ' + row['token'])
        for name in ['original', 'mixed', 'qat']:
            pair = catalog['pairs'][name]
            output = args.results / 'gt-replay' / name
            output.mkdir(parents=True, exist_ok=True)
            output.chmod(0o777)
            command = [args.replay_binary.resolve(), '--workspace', ROOT, '--input-dir', args.results / 'gt-inputs',
                       '--output-dir', output, '--camera-model', args.asset_root / models[pair[0]]['path'],
                       '--bev-model', args.asset_root / models[pair[1]]['path'], '--camera-post-target', 'A65',
                       '--bev-post-target', 'A65', '--threads', '8', '--warmup', '2', '--repeat', '1',
                       '--check-finite-logits', 'true']
            run(['ssh', '-o', 'BatchMode=yes', args.device, shlex.join(map(str, command))], output / 'app.log')
            measured = json.loads((output / 'summary.json').read_text())
            tokens = {p.stem for p in (args.asset_root / 'gt').glob('*.npz')}
            if measured['signal'] or len(measured['frames']) != 32 or {r['token'] for r in measured['frames']} != tokens:
                raise ValueError('Hardware replay did not complete the same 32 samples: ' + name)
        run([args.python, REFERENCE / 'evaluate_gt.py', '--gt-dir', args.asset_root / 'gt', '--results', args.results])
    else:
        require(parser, args.api, '--api', 'FLASHOCC_INSIGHT_API')
        original = args.asset_root / 'artifacts/insight-demo/config.json'
        config = json.loads(original.read_text())
        source = args.results / 'source'
        source.mkdir(exist_ok=True)
        host = args.rtsp_host or urlsplit(args.api).hostname
        config['insight_host'] = urlsplit(args.api).hostname
        for clip in config['clips']:
            target = source / clip['file']
            if target.is_symlink() and target.resolve() != (original.parent / clip['file']).resolve():
                target.unlink()
            if not target.exists():
                target.symlink_to(original.parent / clip['file'])
            clip['url'] = f'rtsp://{host}:{args.rtsp_port}/src{clip["source_index"]}'
        for frame in config['frames']:
            path = frame['geometry']
            relative = 'artifacts/' + path.split('/artifacts/', 1)[1] if '/artifacts/' in path else path
            frame['geometry'] = str(args.asset_root / relative)
        base = source / 'config.json'
        base.write_text(json.dumps(config, indent=2) + '\n')
        baseline = json.loads((REFERENCE / 'baseline.json').read_text())
        variants = ['mixed-arm', 'mixed-ev74', 'qat-arm', 'qat-ev74']
        if args.include_scan_control:
            variants.append('qat-ev74-check')
        for name in variants:
            execution = dict(baseline['performance'][name]['execution'])
            for key in ['camera_model', 'bev_model']:
                execution[key] = str(args.asset_root / execution[key])
            path = args.results / (name + '.json')
            path.write_text(json.dumps(execution, indent=2) + '\n')
            command = [sys.executable, ROOT / 'scripts/benchmark_insight_fps.py', '--device', args.device,
                 '--api', args.api, '--base', base, '--rates', '12', '--frames', '256', '--warmup', '64',
                 '--volume-fps', '2', '--binary', args.insight_binary.resolve(), '--execution', path,
                 '--inputs-dir', args.results / 'inputs', '--output-dir', args.results / 'live' / name]
            if args.browser_python:
                command.extend(['--browser-python', args.browser_python])
            run(command)
    summarize(args.results)


if __name__ == '__main__':
    main()
