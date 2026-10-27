"""Prepare MLA simulator inputs and checks against direct hardware output, without Neat adapters."""

from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from pathlib import Path


def signed(values):
    return ' '.join(str(v if v < 128 else v - 256) for v in values)


def main():
    root = _PROJECT_ROOT
    directory = root / 'artifacts/compile-batch6/diagnostics/direct-transport'
    with (directory / 'input.mlc').open('w') as output:
        for camera in range(6):
            data = (directory / f'camera-{camera}.bin').read_bytes()
            output.write(f'.begin_section "ifm.b{camera}"\n')
            for offset in range(0, len(data), 16):
                output.write(f'.dram_data {offset} {signed(data[offset:offset + 16])}\n')
            output.write('.end_section\n')
    depth = (directory / 'batch6/head-0.bin').read_bytes()
    features = (directory / 'batch6/head-1.bin').read_bytes()
    with (directory / 'hardware-check.mlc').open('w') as output:
        for camera in range(6):
            output.write(f'.begin_section "ofm.b{camera}"\n')
            data = depth[camera * 67584:(camera + 1) * 67584]
            # Check the first 32 channels in every C48 tile, excluding undefined C padding.
            for tile in range(16):
                for within_tile in range(0, 4 * 11 * 32 * 2, 16):
                    offset = tile * (4 * 11 * 48 * 2) + within_tile
                    output.write(f'.dram_check {offset} {signed(data[offset:offset + 16])}\n')
            data = features[camera * 90112:(camera + 1) * 90112]
            for offset in range(0, len(data), 16):
                output.write(f'.dram_check {67584 + offset} {signed(data[offset:offset + 16])}\n')
            output.write('.end_section\n')
    print('Wrote six-camera simulator input and checks for 811008 defined output bytes')


if __name__ == '__main__':
    main()
