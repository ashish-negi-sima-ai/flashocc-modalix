"""Upload/control only FlashOCC's six Insight source slots; run on the SDK host."""
import argparse
import json
import ssl
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from runtime_config import defaults, require

ROOT=Path(__file__).resolve().parents[1]

def main():
    runtime=defaults()
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['start','stop','status'])
    parser.add_argument('--config',type=Path,default=ROOT/'artifacts/insight-demo/config.json')
    parser.add_argument('--api',default=runtime['insight_api'])
    args=parser.parse_args()
    require(parser,args.api,'--api','FLASHOCC_INSIGHT_API')
    config=json.loads(args.config.read_text())
    def api(path,data=None):
        request=urllib.request.Request(args.api+path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,context=ssl._create_unverified_context(),timeout=40) as response:
            return json.load(response)
    print('Health:',api('/api/health'))
    sources={s['index']:s for s in api('/api/mediasrc')}
    for clip in config['clips']:
        index=clip['source_index'];source=sources[index]
        if args.action=='start':
            if source['file'] not in ['',clip['file']]:
                raise RuntimeError(f'Source {index} is assigned to {source["file"]}; choose unused slots in config')
            if source['state']=='playing':api('/api/mediasrc/stop',{'index':index})
            subprocess.run(['curl','--fail','--silent','--show-error','-k','-F',
                f'file=@{args.config.parent/clip["file"]}',args.api+'/api/upload/media'],check=True)
            print(api('/api/mediasrc/assign',{'index':index,'file':clip['file'],'transport':'rtsp'}))
        elif args.action=='stop' and source['file']==clip['file']:
            print(api('/api/mediasrc/stop',{'index':index}))
    if args.action=='start':
        # Independent slots start together; serial API probing can stagger six clips.
        with ThreadPoolExecutor(max_workers=6) as workers:
            responses=list(workers.map(lambda clip:api('/api/mediasrc/start',{'index':clip['source_index']}),config['clips']))
        for response in responses:print(response)
    state=api('/api/mediasrc')
    selected=[s for s in state if s['index'] in {c['source_index'] for c in config['clips']}]
    viewer=api('/api/viewer-url?mode=light&src='+str(config['channel']))
    report=dict(sources=selected,viewer=viewer)
    destination=args.config.parent/'insight-state.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
