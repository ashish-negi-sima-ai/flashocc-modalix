#!/usr/bin/env python3
"""Run extracted MLA ELFs directly on an idle Modalix; restore AppComplex afterward.

Run on the DevKit with system Python (requires PyYAML and passwordless sudo).
This measures ELF execution with synthetic/default inputs, not model accuracy.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import tarfile
import yaml


UNIT = 'simaai-appcomplex.service'


def state(operation):
    return subprocess.run(['systemctl', operation, UNIT], capture_output=True,
                          text=True, check=False).stdout.strip()


def snapshot():
    frequency = Path('/sys/class/devfreq/5100000.simaai,m4-rproc/cur_freq')
    return {'appcomplex_active': state('is-active'),
            'appcomplex_enabled': state('is-enabled'),
            'mla_frequency_hz': int(frequency.read_text()) if frequency.exists() else None}


def run(command, log, timeout):
    started = time.monotonic()
    with log.open('w') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    if code:
        raise RuntimeError(f'{command} exited {code}; see {log}')
    return {'command': command, 'log': str(log),
            'process_wall_seconds': time.monotonic() - started}


def extract(archive, destination):
    archive = archive.resolve()
    destination.mkdir()
    members = []
    with tarfile.open(archive) as package:
        for entry in package.getmembers():
            if entry.isfile() and entry.name.endswith(('.elf', '_mla_stats.yaml')):
                target = destination / Path(entry.name).name
                if target.exists():
                    raise ValueError(f'Duplicate extracted filename: {entry.name}')
                target.write_bytes(package.extractfile(entry).read())
                members.append({'path': str(target), 'bytes': target.stat().st_size,
                                'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    elfs = list(destination.glob('*.elf'))
    if len(elfs) != 1:
        raise ValueError(f'Expected one MLA ELF in {archive}, got {len(elfs)}')
    elf = elfs[0]
    layers = elf.with_name(elf.stem + '_stats.yaml')
    if not layers.is_file():
        raise ValueError(f'Missing matching layer stats: {layers}')
    return {'archive': str(archive),
            'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'members': members, 'elf': str(elf), 'compiler_layers': str(layers),
            'whole_elf_runs': [], 'counter_runs': [], 'layer_profiles': []}


def load_stats(path, samples):
    document = yaml.safe_load(path.read_text())
    if len(document) != 1:
        raise ValueError(f'Unexpected model count in {path}')
    result = next(iter(document.values()))
    for name in ('tile.lat', 'api.lat', 'model.lat'):
        if result[name]['runcount'] != samples or result[name]['avg'] <= 0:
            raise ValueError(f'Invalid {name} stats in {path}')
    return result


def stop_signal(number, _frame):
    raise RuntimeError(f'Interrupted by signal {number}; restoring service state')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=Path, required=True)
    parser.add_argument('--packed-camera', type=Path, required=True)
    parser.add_argument('--bev', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--layer-repeats', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()
    if platform.machine() != 'aarch64':
        parser.error('Run this script on Modalix, not the SDK host')
    if min(args.samples, args.repeats, args.layer_repeats, args.timeout) < 1:
        parser.error('Counts and timeout must be positive')
    if not shutil.which('mla-rt'):
        parser.error('mla-rt is not installed')
    subprocess.run(['sudo', '-n', 'true'], check=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    models = {'sequential-camera': extract(args.camera, output / 'sequential-camera'),
              'packed3-camera': extract(args.packed_camera, output / 'packed3-camera')}
    if args.bev:
        models['bev'] = extract(args.bev, output / 'bev')
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'hostname': platform.node(), 'uname': list(platform.uname()),
              'scope': 'Direct MLA ELF; tile.lat is hardware execution; no Neat, EV74 or app',
              'input': 'No IFM file supplied; mla-rt default input allocation; no accuracy check',
              'sampling': {'samples_per_process': args.samples, 'repeats': args.repeats,
                           'layer_repeats': args.layer_repeats,
                           'warmup': 'No samples removed; process/load time excluded from tile.lat'},
              'mla_rt_version': subprocess.run(['mla-rt', '-V'], text=True,
                                               capture_output=True, check=True).stdout.strip(),
              'mla_rt_sha256': hashlib.sha256(Path(shutil.which('mla-rt')).read_bytes()).hexdigest(),
              'benchmark_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'before': snapshot(), 'models': models}
    (output / 'mla-rt-help.txt').write_text(subprocess.run(
        ['mla-rt', '-h'], text=True, capture_output=True).stdout)
    before = report['before']
    if before['appcomplex_active'] not in ('active', 'inactive'):
        raise RuntimeError(f'Unexpected AppComplex state: {before}')
    if before['appcomplex_enabled'] not in ('enabled', 'disabled'):
        raise RuntimeError(f'Unexpected AppComplex enablement: {before}')
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, stop_signal)
    def checkpoint():
        (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')

    checkpoint()
    try:
        subprocess.run(['sudo', '-n', 'systemctl', 'disable', '--now', UNIT], check=True)
        report['during'] = snapshot()
        if report['during']['appcomplex_active'] != 'inactive':
            raise RuntimeError('AppComplex did not stop')
        print('AppComplex disabled and stopped', flush=True)
        for repeat in range(args.repeats):
            names = list(models)
            if repeat % 2:
                names.reverse()
            for name in names:
                model = models[name]
                stats = output / name / f'whole-{repeat}.yaml'
                command = ['sudo', '-n', 'mla-rt', '-a', 'mod,a65', '-m', str(args.samples),
                           '-vs:usec:3', '--stats', str(stats), model['elf']]
                measurement = run(command, stats.with_suffix('.log'), args.timeout)
                measurement['stats_us'] = load_stats(stats, args.samples)
                measurement['environment'] = snapshot()
                model['whole_elf_runs'].append(measurement)
                checkpoint()
                print(f'{name} repeat {repeat}: {measurement["stats_us"]["tile.lat"]}', flush=True)
        for name, model in models.items():
            stats = output / name / 'counters.yaml'
            command = ['sudo', '-n', 'mla-rt', '-a', 'mod,a65', '-m', str(args.samples),
                       '-e', '2', '-e', '5', '-e', '6', '-e', '7',
                       '-vs:usec:3', '--stats', str(stats), model['elf']]
            measurement = run(command, stats.with_suffix('.log'), args.timeout)
            measurement['stats_us'] = load_stats(stats, args.samples)
            model['counter_runs'].append(measurement)
            checkpoint()
            for repeat in range(args.layer_repeats):
                layers = output / name / f'layers-{repeat}.yaml'
                shutil.copyfile(model['compiler_layers'], layers)
                command = ['sudo', '-n', 'mla-rt', '-a', 'mod,a65', '-vs:usec:3',
                           '-p', str(layers), model['elf']]
                measurement = run(command, layers.with_suffix('.log'), args.timeout)
                layer_output = layers.with_name(layers.stem + '_output.yaml')
                actual = yaml.safe_load(layer_output.read_text())
                expected = yaml.safe_load(layers.read_text())
                expected = {k: v for k, v in expected.items()
                            if k != 0 and 'placeholder' not in v['name']}
                if actual.keys() != expected.keys():
                    raise ValueError(f'Layer IDs differ for {name}')
                for key, row in actual.items():
                    for field in ('name', 'start_cycle', 'end_cycle'):
                        if row[field] != expected[key][field]:
                            raise ValueError(f'Layer {key} {field} mismatch for {name}')
                measurement['layers'] = actual
                model['layer_profiles'].append(measurement)
                checkpoint()
                print(f'{name} layer profile {repeat}: {len(actual)} layers', flush=True)
            model['mean_tile_ms'] = statistics.mean(
                v['stats_us']['tile.lat']['avg'] for v in model['whole_elf_runs']) / 1000
            model['mean_direct_api_ms'] = statistics.mean(
                v['stats_us']['api.lat']['avg'] for v in model['whole_elf_runs']) / 1000
        report['completed'] = True
    finally:
        checkpoint()
        try:
            if before['appcomplex_enabled'] == 'enabled':
                subprocess.run(['sudo', '-n', 'systemctl', 'enable', UNIT], check=True)
            if before['appcomplex_active'] == 'active':
                # Layer stepping can leave the accelerator halted. The unit's
                # ExecStartPre uses -N (no reset), so reset via the standard init
                # ELF first, while AppComplex is still stopped.
                report['restore_initialization'] = run(
                    ['sudo', '-n', 'mla-rt', '-a', 'mod,a65', '-vs',
                     '-d', '/lib/firmware/mla_driver.elf', '/usr/bin/mla_init_modalix.elf'],
                    output / 'restore-initialization.log', args.timeout)
                subprocess.run(['sudo', '-n', 'systemctl', 'reset-failed', UNIT], check=True)
                subprocess.run(['sudo', '-n', 'systemctl', 'start', UNIT], check=True)
        except BaseException as error:
            report['restore_error'] = str(error)
            raise
        finally:
            report['after'] = snapshot()
            checkpoint()
        print(f'AppComplex restored: {report["after"]}', flush=True)
    for name, model in models.items():
        print(f'{name}: MLA {model["mean_tile_ms"]:.6f} ms; '
              f'direct API {model["mean_direct_api_ms"]:.6f} ms', flush=True)


if __name__ == '__main__':
    main()
