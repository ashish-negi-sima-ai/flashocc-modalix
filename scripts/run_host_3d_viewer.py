"""Manage the Python 3D viewer service from a host terminal outside Docker."""
import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from runtime_config import defaults

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['start','stop','status'])
    parser.add_argument('--port',type=int,default=8090)
    parser.add_argument('--source',type=Path,default=ROOT/'artifacts/insight-device/latest.occ')
    parser.add_argument('--host',default='127.0.0.1',help='Browser-reachable host address to print')
    parser.add_argument('--insight-url',default=defaults()['insight_url'])
    args=parser.parse_args()
    if Path('/.dockerenv').exists():parser.error('Run this command in a host terminal outside Docker')
    directory=ROOT/'artifacts/host-3d-viewer';directory.mkdir(parents=True,exist_ok=True)
    pid_file=directory/'server.pid';script=ROOT/'viewer/server.py'
    try:pid=int(pid_file.read_text())
    except (FileNotFoundError,ValueError):pid=0
    def owned():
        if pid<=0:return False
        try:return str(script).encode() in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):return False
    if args.action=='stop':
        if owned():
            os.kill(pid,signal.SIGTERM)
            for _ in range(50):
                if not owned():break
                time.sleep(.1)
            if owned():raise RuntimeError('Server has not exited; inspect server.log')
        pid_file.unlink(missing_ok=True);print('3D viewer stopped.');return
    if args.action=='start' and not owned():
        with (directory/'server.log').open('a') as log:
            launch=[sys.executable,str(script),'--port',str(args.port),'--source',str(args.source)]
            if args.insight_url:launch.extend(['--insight-url',args.insight_url])
            child=subprocess.Popen(launch,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        pid_file.write_text(str(child.pid)+'\n');pid=child.pid
        for _ in range(30):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/health',timeout=1):break
            except OSError:time.sleep(.1)
        else:raise RuntimeError('Viewer did not become ready; inspect server.log')
    print('Host 3D viewer:', 'running' if owned() else 'stopped')
    if owned():print(f'Open http://{args.host}:{args.port}/')

if __name__=='__main__':main()
