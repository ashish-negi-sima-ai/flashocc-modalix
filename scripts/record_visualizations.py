"""Record Insight and the live host 3D viewer; create MP4s and README GIFs."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time


def encode(source, destination, seconds, tail=False):
    command = ['ffmpeg', '-v', 'error', '-y']
    if tail:
        command += ['-sseof', str(-seconds)]
    command += ['-i', str(source), '-t', str(seconds), '-an', '-vf', 'fps=20,scale=1280:-2',
                '-c:v', 'libx264', '-preset', 'slow', '-crf', '23', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(destination)]
    subprocess.run(command, check=True)
    preview = destination.with_suffix('.gif')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(destination), '-t', str(seconds),
                    '-filter_complex', 'fps=8,scale=800:-1:flags=lanczos,split[a][b];'
                    '[a]palettegen=max_colors=128:stats_mode=diff[p];'
                    '[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle',
                    '-loop', '0', str(preview)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--insight-url', required=True)
    parser.add_argument('--grid-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 30:
        parser.error('--seconds must be 1–30')
    from playwright.sync_api import sync_playwright
    args.output.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'duration_seconds': args.seconds, 'errors': [], 'views': {}}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=[
            '--no-sandbox', '--autoplay-policy=no-user-gesture-required',
            '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        insight_context = browser.new_context(ignore_https_errors=True, viewport={'width': 1440, 'height': 960})
        grid_context = browser.new_context(ignore_https_errors=True, viewport={'width': 1440, 'height': 960},
            record_video_dir=str(args.work_dir), record_video_size={'width': 1440, 'height': 960})
        try:
            insight = insight_context.new_page()
            grid = grid_context.new_page()
            for name, page in [('insight', insight), ('grid', grid)]:
                page.on('pageerror', lambda error, n=name: metadata['errors'].append({'view': n, 'error': str(error)}))
            insight.goto(args.insight_url, wait_until='domcontentloaded')
            insight.wait_for_function('Array.from(document.querySelectorAll("video")).some(v=>v.videoWidth===1536 && v.currentTime>0)', timeout=90000)
            grid.goto(args.grid_url, wait_until='domcontentloaded')
            grid.wait_for_function('window.viewerDebug && viewerDebug().received>=2 && viewerDebug().visibleCount>1000', timeout=90000)
            grid.wait_for_timeout(1500)
            grid.screenshot(path=str(args.work_dir / 'grid-before.png'))
            insight.screenshot(path=str(args.work_dir / 'insight-before.png'))
            print('Both live views ready; recording Insight.', flush=True)
            result = insight.evaluate('''async seconds => {
                const video = Array.from(document.querySelectorAll('video')).find(v=>v.videoWidth===1536 && v.currentTime>0);
                const first = video.getVideoPlaybackQuality().totalVideoFrames;
                const stream = new MediaStream(video.srcObject.getVideoTracks().map(t=>t.clone()));
                const chunks = [];
                const recorder = new MediaRecorder(stream, {mimeType:'video/webm;codecs=vp8',videoBitsPerSecond:4000000});
                const finished = new Promise(resolve => {recorder.onstop = resolve;});
                recorder.ondataavailable = event => {if(event.data.size)chunks.push(event.data);};
                recorder.start(100);
                await new Promise(resolve=>setTimeout(resolve,(seconds+1)*1000));
                recorder.stop(); await finished;
                stream.getTracks().forEach(t=>t.stop());
                const data = await new Promise(resolve=>{
                    const reader=new FileReader(); reader.onload=()=>resolve(reader.result.split(',')[1]);
                    reader.readAsDataURL(new Blob(chunks,{type:'video/webm'}));
                });
                return {data, decoded_frames:video.getVideoPlaybackQuality().totalVideoFrames-first};
            }''', args.seconds)
            (args.work_dir / 'insight.webm').write_bytes(base64.b64decode(result.pop('data')))
            if result['decoded_frames'] < args.seconds * 2:
                raise RuntimeError('Insight did not deliver enough fresh frames')
            metadata['views']['insight'] = result
            print('Recording host 3D view with a real orbit gesture.', flush=True)
            before = grid.evaluate('viewerDebug()')
            bounds = grid.locator('#canvas').bounding_box()
            start_x = bounds['x'] + bounds['width'] * .70
            start_y = bounds['y'] + bounds['height'] * .45
            grid.mouse.move(start_x, start_y)
            grid.mouse.down()
            started = time.monotonic()
            steps = args.seconds * 10 + 10
            for step in range(steps):
                grid.mouse.move(start_x - bounds['width'] * .38 * step / steps,
                                start_y + bounds['height'] * .06 * step / steps)
                delay = started + (step + 1) * .1 - time.monotonic()
                if delay > 0:
                    grid.wait_for_timeout(delay * 1000)
            grid.mouse.up()
            after = grid.evaluate('viewerDebug()')
            if after['received'] - before['received'] < 3:
                raise RuntimeError('3D viewer is not receiving live volume updates')
            if before['camera'] == after['camera'] or after['zScale'] != 1:
                raise RuntimeError('Orbit/physical-scale verification failed')
            metadata['views']['grid'] = {'before': before, 'after': after}
            grid.screenshot(path=str(args.work_dir / 'grid-after.png'))
            grid_video = grid.video
            grid_context.close()
            grid_video.save_as(str(args.work_dir / 'grid.webm'))
            if metadata['errors']:
                raise RuntimeError(metadata['errors'])
        finally:
            insight_context.close()
            grid_context.close()
            browser.close()
    encode(args.work_dir / 'insight.webm', args.output / 'insight.mp4', args.seconds)
    encode(args.work_dir / 'grid.webm', args.output / 'host-3d-grid.mp4', args.seconds, tail=True)
    metadata['files'] = {}
    for path in args.output.iterdir():
        if path.suffix in ['.mp4', '.gif']:
            with path.open('rb') as stream:
                metadata['files'][path.name] = {'size_bytes': path.stat().st_size,
                    'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
    (args.output / 'recording.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'files':metadata['files'],'errors':metadata['errors']}), flush=True)


if __name__ == '__main__':
    main()
