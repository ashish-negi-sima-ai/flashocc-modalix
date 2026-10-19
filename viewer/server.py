"""Host-only occupancy snapshot HTTP service; Python standard library only."""
import argparse
import gzip
import hashlib
import json
import re
import signal
import struct
import threading
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
HEADER=struct.Struct('<8sQQQ32s')
VOXELS=200*200*16
NAMES=['other','barrier','bicycle','bus','car','construction vehicle','motorcycle',
       'pedestrian','traffic cone','trailer','truck','road','other flat','sidewalk',
       'terrain','structure','vegetation','free']
COLORS=['#777777','#ff9e00','#ff61ff','#ffff00','#008cff','#00ffff','#ff00ff',
        '#ff0000','#ffc800','#6496fa','#50ffff','#96324b','#ff0096','#af004b',
        '#4b004b','#c8c8c8','#00af00','#ffffff']

def validate_snapshot(data):
    if len(data)!=HEADER.size+VOXELS:raise ValueError('Invalid snapshot length')
    magic,sequence,marker,timestamp,token=HEADER.unpack_from(data)
    if magic!=b'FOCC3D01' or marker>=256 or timestamp==0 or not re.fullmatch(b'[0-9a-f]{32}',token):
        raise ValueError('Invalid occupancy snapshot header')
    if max(data[HEADER.size:])>17:raise ValueError('Invalid occupancy class')
    return dict(sequence=sequence,marker=marker,timestamp_ms=timestamp,token=token.decode())

class SnapshotStore:
    def __init__(self,path):
        self.path=Path(path);self.lock=threading.Lock();self.key=None
        self.data=None;self.compressed=None;self.etag=None;self.metadata=None
    def get(self):
        with self.lock:
            # The producer replaces a complete file atomically; this open handle
            # stays on one inode even when the next frame is published.
            with self.path.open('rb') as stream:
                header=stream.read(HEADER.size)
                if len(header)!=HEADER.size:raise ValueError('Incomplete snapshot header')
                if header!=self.key:
                    data=header+stream.read(VOXELS+1)
                    metadata=validate_snapshot(data)
                    self.data=data;self.metadata=metadata
                    self.compressed=gzip.compress(data,compresslevel=1)
                    self.etag='"'+hashlib.sha256(header).hexdigest()+'"';self.key=header
            return self.data,self.compressed,self.etag,self.metadata

class Server(ThreadingHTTPServer):
    daemon_threads=True
    allow_reuse_address=True

def make_handler(store):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT/'viewer'),**kwargs)
        def log_message(self,format,*args):
            if not str(args[1] if len(args)>1 else '').startswith(('200','304')):
                super().log_message(format,*args)
        def respond(self,status,body=b'',content_type='application/json',extra=None):
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
            for key,value in (extra or {}).items():self.send_header(key,value)
            self.end_headers()
            if body:self.wfile.write(body)
        def do_GET(self):
            path=urlsplit(self.path).path
            if path=='/api/config':
                config=dict(shape=[200,200,16],voxel_size=.4,lower=[-40,-40,-1],upper=[40,40,5.4],
                    order='XYZ',header_bytes=HEADER.size,classes=[dict(id=i,name=n,color=COLORS[i]) for i,n in enumerate(NAMES)],
                    source='Modalix live occupancy',prediction_is_ground_truth=False)
                return self.respond(200,json.dumps(config).encode())
            if path=='/api/health':
                return self.respond(200,json.dumps(dict(status='ok',service='flashocc-3d-viewer')).encode())
            if path=='/api/frame':
                try:data,compressed,etag,_=store.get()
                except (FileNotFoundError,ValueError,OSError) as error:
                    return self.respond(503,json.dumps(dict(error=str(error),waiting=True)).encode())
                if self.headers.get('If-None-Match')==etag:return self.respond(304,extra={'ETag':etag})
                encoded='gzip' in self.headers.get('Accept-Encoding','')
                headers={'ETag':etag,'Vary':'Accept-Encoding'}
                if encoded:headers['Content-Encoding']='gzip'
                return self.respond(200,compressed if encoded else data,'application/octet-stream',headers)
            if path.startswith('/api/'):
                return self.respond(404,b'{"error":"Unknown endpoint"}')
            return super().do_GET()
    return Handler

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'artifacts/insight-device/latest.occ')
    parser.add_argument('--bind',default='0.0.0.0');parser.add_argument('--port',type=int,default=8090)
    args=parser.parse_args()
    if not (ROOT/'viewer/vendor/three.module.min.js').exists():
        parser.error('Run python3 scripts/setup_3d_viewer.py first')
    server=Server((args.bind,args.port),make_handler(SnapshotStore(args.source)))
    def stop(*_):threading.Thread(target=server.shutdown,daemon=True).start()
    for name in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(name,stop)
    print(f'FlashOCC 3D viewer listening on {args.bind}:{args.port}; source={args.source}',flush=True)
    try:server.serve_forever(poll_interval=.25)
    finally:server.server_close()

if __name__=='__main__':main()
