"""Record Insight and the live host 3D viewer; create MP4s and README GIFs."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time


def encode(source, destination, seconds, tail=False, width=1280, height=None, mp4_only=False, crf=23):
    command = ['ffmpeg', '-v', 'error', '-y']
    if tail:
        command += ['-sseof', str(-seconds)]
    scale = f'scale={width}:-2:flags=lanczos'
    if height:
        scale = (f'scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,'
                 f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2')
    command += ['-i', str(source), '-t', str(seconds), '-an', '-vf', f'fps=20,{scale},setsar=1',
                '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(destination)]
    subprocess.run(command, check=True)
    if mp4_only:
        return
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
    parser.add_argument('--width', type=int, default=1280, help='MP4 output width; use 1920 for Full HD')
    parser.add_argument('--height', type=int, help='MP4 output height and browser viewport height')
    parser.add_argument('--mp4-only', action='store_true', help='Skip GIF generation')
    parser.add_argument('--crf', type=int, default=23, help='H.264 quality; lower is better (default 23)')
    parser.add_argument('--browser-executable', type=Path, help='Optional installed Chromium/Chrome executable')
    parser.add_argument('--hardware-webgl', action='store_true', help='Use the browser GPU backend instead of forcing SwiftShader')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 30:
        parser.error('--seconds must be 1–30')
    if args.width < 640 or args.width % 2 or (args.height is not None and (args.height < 360 or args.height % 2)):
        parser.error('Use even dimensions of at least 640×360')
    if not 0 <= args.crf <= 51:
        parser.error('--crf must be 0–51')
    from playwright.sync_api import sync_playwright
    args.output.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    viewport = {'width': args.width, 'height': args.height} if args.height else {'width': 1440, 'height': 960}
    metadata = {'duration_seconds': args.seconds, 'output_width': args.width, 'output_height': args.height,
                'viewport': viewport, 'crf': args.crf, 'hardware_webgl_requested': args.hardware_webgl,
                'errors': [], 'views': {}}
    with sync_playwright() as playwright:
        browser_args = ['--no-sandbox', '--autoplay-policy=no-user-gesture-required']
        if args.hardware_webgl:
            browser_args += ['--enable-gpu', '--ignore-gpu-blocklist']
        else:
            browser_args += ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']
        launch = {'headless': True, 'args': browser_args}
        if args.browser_executable:
            launch['executable_path'] = str(args.browser_executable)
        browser = playwright.chromium.launch(**launch)
        insight_context = browser.new_context(ignore_https_errors=True, viewport=viewport)
        grid_context = browser.new_context(ignore_https_errors=True, viewport=viewport,
            record_video_dir=str(args.work_dir), record_video_size=viewport)
        try:
            insight = insight_context.new_page()
            grid = grid_context.new_page()
            for name, page in [('insight', insight), ('grid', grid)]:
                page.on('pageerror', lambda error, n=name: metadata['errors'].append({'view': n, 'error': str(error)}))
            insight.goto(args.insight_url, wait_until='domcontentloaded')
            insight.wait_for_function('Array.from(document.querySelectorAll("video")).some(v=>v.videoWidth===1536 && v.currentTime>0)', timeout=90000)
            grid.goto(args.grid_url, wait_until='domcontentloaded')
            grid.wait_for_function('''window.viewerDebug && viewerDebug().received>=2 &&
                viewerDebug().visibleCount>1000 && viewerDebug().triangles>1000 &&
                !document.querySelector('#canvas canvas').getContext('webgl2').isContextLost()''', timeout=90000)
            metadata['webgl_renderer'] = grid.locator('#canvas canvas').evaluate('''canvas => {
                const gl=canvas.getContext('webgl2');
                const ext=gl.getExtension('WEBGL_debug_renderer_info');
                return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
            }''')
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
                return {data, decoded_frames:video.getVideoPlaybackQuality().totalVideoFrames-first,
                        source_width:video.videoWidth, source_height:video.videoHeight};
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
            grid.evaluate('''() => {
                document.querySelector('#canvas canvas').addEventListener('pointerdown',
                    event => {window.capturePointerId=event.pointerId;}, {once:true,capture:true});
            }''')
            grid.mouse.down()
            started = time.monotonic()
            # Drive the drag by wall time on rendered frames. Slow WebGL must
            # not turn 150 synchronous automation round-trips into minutes.
            grid.evaluate('''async p => {
                const canvas=document.querySelector('#canvas canvas');
                const started=performance.now();
                await new Promise(resolve => {
                    function step(now) {
                        const t=Math.min(1,(now-started)/((p.seconds+1)*1000));
                        canvas.dispatchEvent(new PointerEvent('pointermove', {
                            bubbles:true,cancelable:true,pointerId:window.capturePointerId,
                            pointerType:'mouse',isPrimary:true,buttons:1,button:-1,
                            clientX:p.x-p.width*.38*t,clientY:p.y+p.height*.06*t
                        }));
                        if(t<1)requestAnimationFrame(step);else resolve();
                    }
                    requestAnimationFrame(step);
                });
            }''', {'seconds': args.seconds, 'x': start_x, 'y': start_y,
                   'width': bounds['width'], 'height': bounds['height']})
            grid.mouse.move(start_x - bounds['width'] * .38, start_y + bounds['height'] * .06)
            grid.mouse.up()
            after = grid.evaluate('viewerDebug()')
            if after['received'] - before['received'] < 3:
                raise RuntimeError('3D viewer is not receiving live volume updates')
            if before['camera'] == after['camera'] or after['zScale'] != 1:
                raise RuntimeError('Orbit/physical-scale verification failed')
            metadata['views']['grid'] = {'before': before, 'after': after,
                                         'orbit_wall_seconds': time.monotonic() - started}
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
    encode(args.work_dir / 'insight.webm', args.output / 'insight.mp4', args.seconds,
           width=args.width, height=args.height, mp4_only=args.mp4_only, crf=args.crf)
    encode(args.work_dir / 'grid.webm', args.output / 'host-3d-grid.mp4', args.seconds, tail=True,
           width=args.width, height=args.height, mp4_only=args.mp4_only, crf=args.crf)
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
