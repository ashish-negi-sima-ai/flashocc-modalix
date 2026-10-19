"""Build an isolated Neat library with the packed BF16 batch output fix."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neat-source', type=Path, default=Path('/neat-resources/core-src'))
    parser.add_argument('--source-copy', type=Path, default=root / 'artifacts/compile-batch6/neat-source')
    parser.add_argument('--build-dir', type=Path, default=root / 'artifacts/build-neat-batch')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    source = args.source_copy.resolve()
    patch = root / 'app/patches/neat-detesscast-batch.patch'
    # Refuse to replace an existing source copy, which might contain local work.
    shutil.copytree(args.neat_source, source)
    subprocess.run(['patch', '-p1', '--forward', '-i', str(patch)], cwd=source, check=True)
    build = args.build_dir.resolve()
    subprocess.run(['cmake', '-S', str(source), '-B', str(build),
                    '-DCMAKE_TOOLCHAIN_FILE=' + str(root / 'app/toolchain.cmake'),
                    '-DCMAKE_BUILD_TYPE=Release', '-DSIMANEAT_BUILD_TUTORIALS=OFF',
                    '-DSIMANEAT_BUILD_TESTS=OFF', '-DSIMANEAT_BUILD_PCIE_PIPELINE_BUILDER=OFF'], check=True)
    subprocess.run(['cmake', '--build', str(build), '--target', 'sima_neat', '-j', str(args.jobs)], check=True)
    library = build / 'libsima_neat.so.5'
    record = {'source': str(args.neat_source.resolve()), 'source_copy': str(source),
              'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
              'library': str(library.resolve()),
              'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
              'installed_runtime_modified': False}
    (build / 'batch-fix.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
