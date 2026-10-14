"""Start/stop the FlashOCC device application and its six owned Insight sources."""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['start','stop','status'])
    parser.add_argument('--device',default='sima@10.42.0.73')
    parser.add_argument('--config',type=Path,default=ROOT/'artifacts/insight-demo/config.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/insight-device')
    parser.add_argument('--api',default='https://10.42.0.1:9900')
    args=parser.parse_args()
    binary=ROOT/'artifacts/build-app/flashocc_insight'
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
                        '--config',str(config),'--api',args.api],check=True)
    if args.action=='start':
        if not binary.is_file():raise RuntimeError('Build flashocc_insight first')
        if ssh('owned',check=False,capture=True).returncode==0:
            print('FlashOCC application is already running.');sources('status');return
        output.mkdir(parents=True,exist_ok=True);output.chmod(0o777)
        sources('start')
        ssh(f'''nohup "$binary" --config {shlex.quote(str(config))} --output-dir "$output" > "$output/live.log" 2>&1 < /dev/null &
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
