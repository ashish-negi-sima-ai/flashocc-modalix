"""Retimestamp six clips without changing pixels and benchmark the full Modalix app."""
import argparse
import hashlib
import json
import shlex
import statistics
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def decoded_hash(path):
    process=subprocess.Popen(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0',
                              '-vsync','0','-f','rawvideo','-pix_fmt','yuv420p','pipe:1'],stdout=subprocess.PIPE)
    digest=hashlib.sha256()
    while data:=process.stdout.read(1048576):digest.update(data)
    if process.wait():raise RuntimeError('Clip verification failed')
    return digest.hexdigest()

def prepare(base,rate):
    config=json.loads(base.read_text());destination=ROOT/'artifacts/fps-inputs'/str(rate)
    destination.mkdir(parents=True,exist_ok=True);checks=[]
    for clip in config['clips']:
        original=base.parent/clip['file'];target=destination/clip['file']
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(original),'-c:v','copy','-bsf:v',
            f'h264_metadata=tick_rate={rate*2},setts=ts=N/({rate}*TB):duration=1/({rate}*TB)',
            '-an','-movflags','+faststart',str(target)],check=True)
        spec=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=avg_frame_rate,nb_frames,duration','-of','json',str(target)]))['streams'][0]
        assert abs(float(Fraction(spec['avg_frame_rate']))-rate)<.002
        reference=decoded_hash(original);assert decoded_hash(target)==reference
        checks.append(dict(file=clip['file'],decoded_pixels_sha256=reference,probe=spec))
    config['fps']=rate;config['replay_speed_relative_to_recording']=rate/2
    path=destination/'config.json';path.write_text(json.dumps(config,indent=2)+'\n')
    (destination/'clip-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    return path

def report(output,warmup):
    summary=json.loads((output/'summary.json').read_text());frames=summary['frames'][warmup:]
    assert len(frames)>30 and summary['signal']==0
    elapsed=(frames[-1]['completed_ms']-frames[0]['completed_ms'])/1000
    advance=frames[-1]['sequence']-frames[0]['sequence'];intervals=len(frames)-1
    return dict(input_fps=frames[0]['input_fps'],measured_groups=len(frames),warmup_groups=warmup,
        elapsed_seconds=elapsed,output_fps=intervals/elapsed,
        observed_input_fps=advance/elapsed,skipped_groups=advance-intervals,
        skipped_fraction=(advance-intervals)/advance,
        median_compute_ms=statistics.median(f['compute_ms'] for f in frames),
        median_application_ms=statistics.median(f['application_ms'] for f in frames),
        median_render_ms=statistics.median(f['render_ms'] for f in frames),
        invalid_markers=summary['invalid_markers'],camera_received=summary['camera_received'],
        output_directory=str(output))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rates',nargs='+',type=int,default=[2,4,6,8])
    parser.add_argument('--frames',type=int,default=180)
    parser.add_argument('--warmup',type=int,default=60)
    parser.add_argument('--device',default='sima@10.42.0.73')
    parser.add_argument('--volume-fps',type=int,default=0,help='3D export FPS; zero matches the pre-viewer benchmark')
    parser.add_argument('--base',type=Path,default=ROOT/'artifacts/insight-demo/config.json')
    parser.add_argument('--browser-python',type=Path,default=Path('/tmp/flashocc-browser-venv/bin/python'))
    parser.add_argument('--execution',type=Path,help='execution overrides for each live app run')
    parser.add_argument('--binary',type=Path,default=ROOT/'artifacts/build-app/flashocc_insight')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/fps-runs')
    args=parser.parse_args()
    if not all(1<=r<=30 for r in args.rates) or not 0<=args.warmup<args.frames-30 or args.frames>256 or not 0<=args.volume_fps<=30:
        parser.error('Use rates 1–30 and at most 256 frames with at least 30 measured frames')
    if not args.binary.is_file():parser.error('Build the requested application binary first')
    if args.execution and not args.execution.is_file():parser.error('Missing execution settings')
    binary=args.binary.resolve()
    results=[];root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    for rate in args.rates:
        config=prepare(args.base,rate);output=root/str(rate);output.mkdir(exist_ok=True);output.chmod(0o777)
        control=[sys.executable,str(ROOT/'scripts/insight_demo_control.py')]
        with (output/'source-control.log').open('w') as controls:
            subprocess.run(control+['start','--config',str(config)],stdout=controls,stderr=subprocess.STDOUT,check=True)
            app_args=[str(binary),'--config',str(config),'--output-dir',str(output),
                      '--frames',str(args.frames),'--volume-fps',str(args.volume_fps)]
            if args.execution:app_args.extend(['--execution',str(args.execution.resolve())])
            command=shlex.join(app_args)
            app=None;browser=None
            try:
                with (output/'app.log').open('w') as log:
                    app=subprocess.Popen(['ssh','-o','BatchMode=yes',args.device,command],stdout=log,stderr=subprocess.STDOUT)
                    if args.browser_python.exists():
                        browser=subprocess.Popen([str(args.browser_python),str(ROOT/'scripts/probe_insight_fps.py'),
                            '--output',str(output/'browser.json')],stdout=(output/'browser.log').open('w'),stderr=subprocess.STDOUT)
                    code=app.wait(timeout=240)
                    if code:raise RuntimeError(f'Device app failed at {rate} FPS; see {output}/app.log')
                if browser and browser.wait(timeout=60):raise RuntimeError(f'Browser failed; see {output}/browser.log')
            finally:
                if app and app.poll() is None:
                    subprocess.run(['ssh','-o','BatchMode=yes',args.device,
                        'pkill -INT -f '+shlex.quote('^'+str(binary)+' --config '+str(config))],check=False)
                    app.wait(timeout=30)
                if browser and browser.poll() is None:browser.terminate();browser.wait(timeout=10)
                subprocess.run(control+['stop','--config',str(config)],stdout=controls,stderr=subprocess.STDOUT,check=True)
        result=report(output,args.warmup)
        if (output/'browser.json').exists():result['browser']=json.loads((output/'browser.json').read_text())
        results.append(result);(root/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
