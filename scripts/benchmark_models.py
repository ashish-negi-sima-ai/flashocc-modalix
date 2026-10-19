"""Run on Modalix with ~/pyneat/bin/python; benchmark the deployed model routes."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    import pyneat

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('network', choices=['camera', 'bev'])
    parser.add_argument('--samples', type=int, default=300)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--model',type=Path,help='override the compiled model archive')
    parser.add_argument('--expected-batch',type=int,help='fail if Neat resolves a different batch size')
    parser.add_argument('--input-width',type=int,help='tensor input width override, e.g. 4384 for spatial camera packing')
    parser.add_argument('--cameras-per-input',type=int,default=1,help='camera views packed in each logical input')
    parser.add_argument('--pre-target',choices=['AUTO','A65','EV74'],default='AUTO')
    parser.add_argument('--post-target',choices=['AUTO','A65','EV74'],default='AUTO')
    args = parser.parse_args()
    if args.samples <= 0 or args.cameras_per_input < 1 or (args.input_width is not None and args.input_width < 1):
        parser.error('--samples, --input-width and --cameras-per-input must be positive')
    root = Path(__file__).resolve().parents[1]
    relative, shape = {
        'camera': ('artifacts/compile-real/camera/flashocc_m0_camera_mpk.tar.gz', (256, 704, 3)),
        'bev': ('artifacts/compile-real-mixed/bev/flashocc_m0_bev_mpk.tar.gz', (200, 200, 64)),
    }[args.network]
    if args.input_width:
        shape = (shape[0], args.input_width, shape[2])
    archive = args.model.resolve() if args.model else root / relative
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.input_max_height, options.preprocess.input_max_width, options.preprocess.input_max_depth = shape
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.name_suffix = 'benchmark_' + args.network
    options.processcvu.pre_run_target = args.pre_target
    options.processcvu.post_run_target = args.post_target
    model = pyneat.Model(str(archive), options)
    batch = model.compiled_batch_size()
    if args.expected_batch is not None and batch != args.expected_batch:
        raise ValueError(f'Expected batch {args.expected_batch}, got {batch}')
    report = model.benchmark(args.samples, True)
    result = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'network': args.network, 'model': str(archive),
        'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'samples_per_window': args.samples,
        'compiled_batch_size': batch,
        'input': 'synthetic tensor; normalization, resize and color conversion disabled',
        'cvu_pre_target': args.pre_target, 'cvu_post_target': args.post_target,
        'scope': 'Neat model route, including required boundary conversions; excludes application CPU work',
        'latency_ms': report.latency_ms, 'fps': report.fps,
        'latency_scope': 'one model batch; warmed-up single-flight mean',
        'fps_unit': ('packed six-camera inputs/s' if args.cameras_per_input == 6 else 'logical images/s')
                    if args.network == 'camera' else 'logical inputs/s',
        'batches_per_second': report.fps / batch,
        'amortized_latency_per_input_ms': report.latency_ms / batch,
        'avg_power_watts': report.avg_power_watts, 'energy_joules': report.energy_joules,
        'input_specs': [spec(s) for s in model.input_specs()],
        'output_specs': [spec(s) for s in model.output_specs()],
    }
    if args.network == 'camera':
        result.update(camera_views_per_logical_input=args.cameras_per_input,
                      camera_images_per_second=report.fps*args.cameras_per_input,
                      six_camera_groups_per_second=report.fps*args.cameras_per_input/6,
                      amortized_latency_per_camera_ms=report.latency_ms/batch/args.cameras_per_input)
    output = args.output_dir or root / 'artifacts/model-benchmark'
    output.mkdir(parents=True, exist_ok=True)
    (output / (args.network + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


def spec(value):
    return {'shape': list(value.shape), 'rank': value.rank,
            'dtypes': [str(dtype) for dtype in value.dtypes]}


if __name__ == '__main__':
    main()
