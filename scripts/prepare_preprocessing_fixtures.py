"""Save identical decoded six-camera NV12 frames for preprocessing comparisons."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'artifacts/insight-demo/config.json')
    parser.add_argument('--output', type=Path, default=root / 'artifacts/preprocess-validation')
    parser.add_argument('--frames', nargs='+', type=int, default=[0, 13])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for index in args.frames:
        if index < 0 or index >= len(config['frames']):
            parser.error('Frame index outside the configured recording')
        record = config['frames'][index]
        paths = []
        for camera, clip in enumerate(config['clips']):
            video = args.config.resolve().parent / clip['file']
            path = output / f'frame{index}-camera{camera}.nv12'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(video),
                            '-vf', f'select=eq(n\\,{index})', '-frames:v', '1',
                            '-pix_fmt', 'nv12', '-f', 'rawvideo', str(path)], check=True)
            if path.stat().st_size != 704 * 288 * 3 // 2:
                raise ValueError('Unexpected decoded frame size')
            paths.append(str(path))
        fixtures.append({'token': record['token'], 'geometry': record['geometry'], 'images': paths})
    (output / 'fixtures.json').write_text(json.dumps(fixtures, indent=2) + '\n')


if __name__ == '__main__':
    main()
