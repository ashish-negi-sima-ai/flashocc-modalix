"""Pause the owned live app, validate/benchmark camera batches, then restore it."""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from runtime_config import defaults, require


def main():
    root = _PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default=defaults()['device'])
    parser.add_argument('--samples', type=int, default=300)
    parser.add_argument('--accuracy-only', action='store_true', help='leave existing benchmark results in place')
    parser.add_argument('--dump-boundaries', action='store_true', help='capture first group model boundary tensors')
    parser.add_argument('--results-dir', type=Path, help='save a diagnostic run separately from the original results')
    parser.add_argument('--execution', type=Path, help='checker execution settings; requires --accuracy-only')
    parser.add_argument('--runtime-library-dir', type=Path,
                        help='isolated Neat library override for checks only; live app uses its original runtime')
    parser.add_argument('--packed-model', type=Path, help='test a spatially packed batch-one archive instead of batch six')
    parser.add_argument('--guard-width', type=int, default=32)
    parser.add_argument('--checker', type=Path, help='override the ARM64 checker executable')
    parser.add_argument('--variants', nargs='+', choices=['batch1', 'batch6', 'packed'],
                        help='run only the selected variants, preserving other saved results')
    args = parser.parse_args()
    require(parser,args.device,'--device','FLASHOCC_DEVICE')
    if args.execution and not args.accuracy_only:
        parser.error('--execution requires --accuracy-only; model benchmark has separate placement options')
    if args.samples < 1:
        parser.error('--samples must be positive')
    if args.guard_width < 32 or args.guard_width % 32:
        parser.error('--guard-width must be a positive multiple of 32')
    if args.packed_model and args.dump_boundaries:
        parser.error('--dump-boundaries is not supported for spatial packing')
    output = root / 'artifacts/compile-batch6'
    results = args.results_dir.resolve() if args.results_dir else output
    results.mkdir(parents=True, exist_ok=True)
    checker = args.checker.resolve() if args.checker else root / 'artifacts/build-app-batch/check_camera_batch'
    models = {'batch1': root / 'artifacts/compile-real/camera/flashocc_m0_camera_mpk.tar.gz',
              'batch6': output / 'camera/flashocc_m0_camera_mpk.tar.gz'}
    if args.packed_model:
        models = {'batch1': models['batch1'], 'packed': args.packed_model.resolve()}
    if args.variants:
        if set(args.variants) - models.keys():
            parser.error('Selected variant is unavailable; packed requires --packed-model')
        models = {name: model for name, model in models.items() if name in args.variants}
    for path in [checker, *models.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    ssh = ['ssh', '-o', 'BatchMode=yes', args.device]
    environment = []
    if args.runtime_library_dir:
        directory = args.runtime_library_dir.resolve()
        library = directory / 'libsima_neat.so.5'
        if not library.is_file():
            raise FileNotFoundError(library)
        environment = ['env', 'LD_LIBRARY_PATH=' + str(directory)]
        (results / 'test-runtime.json').write_text(json.dumps({
            'library': str(library.resolve()),
            'sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
            'patch': str(root / 'experiments/batch6/patches/neat-detesscast-batch.patch'),
            'installed_runtime_modified': False,
        }, indent=2) + '\n')
    live = root / 'artifacts/insight-device'
    binary = root / 'build/modalix/flashocc_insight'
    code = f'''import json,os
from pathlib import Path
p=Path({str(live / 'app.pid')!r})
result=None
if p.exists():
 pid=int(p.read_text())
 try:
  if os.readlink('/proc/'+str(pid)+'/exe')=={str(binary)!r}:
   result={{'pid':pid,'argv':[x.decode() for x in Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(bytes([0])) if x]}}
 except FileNotFoundError: pass
print(json.dumps(result))
'''
    active = json.loads(subprocess.check_output(ssh + [shlex.join(['python3', '-c', code])], text=True))
    restart = False
    try:
        if active:
            pid = active['pid']
            script = f'''kill -INT {pid}
for attempt in $(seq 1 80); do
  [ "$(readlink /proc/{pid}/exe)" = {shlex.quote(str(binary))} ] || exit 0
  sleep 0.25
done
exit 1'''
            restart = True
            subprocess.run(ssh + [script], check=True)
            before = results / 'live-before'
            before.mkdir(exist_ok=True)
            (before / 'process.json').write_text(json.dumps(active, indent=2) + '\n')
            for name in ['summary.json', 'execution.json', 'live.log']:
                if (live / name).exists():
                    shutil.copyfile(live / name, before / name)
        for label, model in models.items():
            directory = results / label
            directory.mkdir(exist_ok=True)
            directory.chmod(0o777)
            benchmark = directory / 'benchmark'
            benchmark.mkdir(exist_ok=True)
            benchmark.chmod(0o777)
            print('Hardware accuracy:', label, flush=True)
            command = environment + [str(checker), str(model), str(output / 'reference'), str(directory)]
            if args.dump_boundaries:
                command.append('--dump-boundaries')
            if args.execution:
                command.extend(['--execution', str(args.execution.resolve())])
            if label == 'packed':
                command.extend(['--packed', '--guard-width', str(args.guard_width)])
            with (directory / 'accuracy.log').open('w') as log:
                subprocess.run(ssh + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT, check=True)
            if args.accuracy_only:
                continue
            print('Neat benchmark:', label, flush=True)
            command = environment + ['/home/sima/pyneat/bin/python', str(root / 'scripts/benchmark_models.py'), 'camera',
                       '--model', str(model), '--expected-batch', '6' if label == 'batch6' else '1',
                       '--samples', str(args.samples), '--output-dir', str(benchmark)]
            if label == 'packed':
                command.extend(['--input-width', str(6*704+5*args.guard_width), '--cameras-per-input', '6'])
            with (benchmark / 'benchmark.log').open('w') as log:
                subprocess.run(ssh + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT, check=True)
    finally:
        if restart:
            command = shlex.join(active['argv'])
            script = f'''if [ "$(readlink /proc/{active['pid']}/exe)" = {shlex.quote(str(binary))} ]; then
  printf 'Original live process is still running; no duplicate started\\n'
  exit 0
fi
nohup {command} > {shlex.quote(str(live / 'live.log'))} 2>&1 < /dev/null &
printf '%s\\n' "$!" > {shlex.quote(str(live / 'app.pid'))}
cat {shlex.quote(str(live / 'app.pid'))}'''
            subprocess.run(ssh + [script], check=True)
            print('Restored original live application command', flush=True)


if __name__ == '__main__':
    main()
