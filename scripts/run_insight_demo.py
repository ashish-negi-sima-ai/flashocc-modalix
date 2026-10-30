"""Start/stop the FlashOCC device application and its six owned Insight sources."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit
from runtime_config import defaults, require

ROOT=Path(__file__).resolve().parents[1]

def launch_config(path, insight_host, rtsp_port=None, video_port=None):
    """Resolve paths and endpoint placeholders without changing the source config."""
    config=json.loads(path.read_text())
    for record in config['frames']:
        geometry=Path(record['geometry'])
        if not geometry.is_absolute():geometry=path.parent/geometry
        if not geometry.is_file():raise ValueError(f'Missing geometry: {geometry}')
        record['geometry']=str(geometry.resolve())
    for clip in config['clips']:
        if not (path.parent/clip['file']).is_file():
            raise ValueError(f'Missing clip: {path.parent/clip["file"]}')
        if '{insight_host}' in clip['url'] or '{rtsp_port}' in clip['url']:
            if rtsp_port is None:raise ValueError('Pass --rtsp-port from neat --json (rtsp.tcp hostPortStart)')
            host=f'[{insight_host}]' if ':' in insight_host else insight_host
            clip['url']=clip['url'].format(insight_host=host,rtsp_port=rtsp_port)
    if config['insight_host']=='{insight_host}':config['insight_host']=insight_host
    if config['video_port']=='{video_port}':
        if video_port is None:raise ValueError('Pass --video-port from neat --json (videoUDP hostPortStart)')
        config['video_port']=video_port
    return config

def main():
    runtime=defaults()
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['start','stop','status'])
    parser.add_argument('--device',default=runtime['device'])
    parser.add_argument('--config',type=Path,default=ROOT/'demo/inputs/config.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/insight-device')
    parser.add_argument('--api',default=runtime['insight_api'])
    parser.add_argument('--insight-host',help='DevKit-reachable RTSP/video host; defaults to the API hostname')
    parser.add_argument('--rtsp-port',type=int,help='published rtsp.tcp port from neat --json')
    parser.add_argument('--video-port',type=int,help='published videoUDP base port from neat --json')
    parser.add_argument('--volume-fps',type=int,help='override configured volume_fps (default 2)')
    parser.add_argument('--execution',type=Path,help='JSON object overriding config.execution')
    parser.add_argument('--check-finite-logits',choices=['true','false'],help='enable/disable the full output-logit validity scan; defaults to execution settings')
    parser.add_argument('--visualization',choices=['true','false'],help='enable/disable the Insight mosaic')
    parser.add_argument('--frames',type=int,help='stop after this many updates; zero runs continuously')
    parser.add_argument('--binary',type=Path,default=ROOT/'demo/bin/flashocc_insight')
    args=parser.parse_args()
    require(parser,args.device,'--device','FLASHOCC_DEVICE')
    require(parser,args.api,'--api','FLASHOCC_INSIGHT_API')
    if args.volume_fps is not None and not 0<=args.volume_fps<=30:parser.error('--volume-fps must be 0–30')
    if args.frames is not None and args.frames<0:parser.error('--frames must be nonnegative')
    for port in (args.rtsp_port,args.video_port):
        if port is not None and not 1<=port<=65535:parser.error('Ports must be 1–65535')
    binary=args.binary.resolve()
    extra=[]
    if args.volume_fps is not None:extra.extend(['--volume-fps',str(args.volume_fps)])
    if args.visualization is not None:extra.extend(['--visualization',args.visualization])
    if args.frames is not None:extra.extend(['--frames',str(args.frames)])
    if args.execution:
        if not args.execution.is_file():parser.error('Missing execution JSON')
        extra.extend(['--execution',str(args.execution.resolve())])
    if args.check_finite_logits is not None:
        extra.extend(['--check-finite-logits',args.check_finite_logits])
    launch_extra=' '.join(shlex.quote(value) for value in extra)
    config=args.config.resolve();output=args.output_dir.resolve()
    header=f'''binary={shlex.quote(str(binary))}
output={shlex.quote(str(output))}
pid=""
if [ -f "$output/app.pid" ]; then pid=$(cat "$output/app.pid"); fi
owned() {{
  case "$pid" in ''|*[!0-9]*) return 1;; esac
  [ "$(readlink /proc/$pid/exe)" = "$binary" ] && kill -0 "$pid" 2>/dev/null
}}
'''
    def ssh(script,check=True,capture=False):
        return subprocess.run(['ssh','-o','BatchMode=yes',args.device,header+script],
                              check=check,text=True,capture_output=capture)
    def sources(action):
        subprocess.run([sys.executable,str(ROOT/'scripts/insight_demo_control.py'),action,
                        '--config',str(config),'--api',args.api,
                        '--state-output',str(output/'insight-state.json')],check=True)
    if args.action=='start':
        if not binary.is_file():raise RuntimeError(f'Missing binary: {binary}; restore the demo or pass --binary with a rebuilt executable')
        if ssh('owned',check=False,capture=True).returncode==0:
            print('FlashOCC application is already running.');sources('status');return
        insight_host=args.insight_host or urlsplit(args.api).hostname
        if not insight_host:parser.error('Cannot infer the Insight host; pass --insight-host')
        try:prepared=launch_config(config,insight_host,args.rtsp_port,args.video_port)
        except ValueError as error:parser.error(str(error))
        output.mkdir(parents=True,exist_ok=True);output.chmod(0o777)
        generated=output/'launch-config.json'
        generated.write_text(json.dumps(prepared,indent=2)+'\n')
        sources('start')
        ssh(f'''nohup "$binary" --workspace {shlex.quote(str(ROOT))} --config {shlex.quote(str(generated))} --output-dir "$output" {launch_extra} > "$output/live.log" 2>&1 < /dev/null &
printf '%s\\n' "$!" > "$output/app.pid"
cat "$output/app.pid"
''')
        print(f'Device application launched; startup takes about 20 seconds. Log: {output}/live.log')
    elif args.action=='stop':
        ssh('''if owned; then
  kill -INT "$pid"
  for attempt in $(seq 1 20); do
    if ! owned; then exit 0; fi
    sleep 0.5
  done
  printf 'Application has not exited yet; check live.log\\n' >&2
  exit 1
fi
''')
        sources('stop')
    else:
        result=ssh('owned',check=False,capture=True)
        print('Device application:', 'running' if result.returncode==0 else 'stopped or unreachable')
        sources('status')

if __name__=='__main__':main()
