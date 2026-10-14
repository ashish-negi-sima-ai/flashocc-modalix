"""Run pooling correctness and thread sweeps on the machine executing this script."""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=100)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'measurements.jsonl').write_text('')
    (args.output/'summary.json').unlink(missing_ok=True)
    fixtures=sorted(args.fixtures.glob('*.pool'))
    if not fixtures:raise RuntimeError('No pooling fixtures')
    subprocess.run([str(args.binary),'--self-test'],check=True)
    results=[]
    for fixture in fixtures:
        for kernel in ['scalar','neon','neon-fma']:
            counts=[1] if kernel=='scalar' else [1,2,4,8,16]
            for threads in counts:
                command=[str(args.binary),str(fixture),str(threads),kernel,str(args.iterations)]
                if kernel=='neon-fma' and threads==8:
                    command.append(str(args.output/(fixture.stem+'.f32')))
                result=subprocess.run(command,check=True,capture_output=True,text=True,timeout=120)
                record=json.loads(result.stdout)
                results.append(record)
                with (args.output/'measurements.jsonl').open('a') as f:
                    f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
    groups=[]
    for kernel in ['scalar','neon','neon-fma']:
        for threads in [1,2,4,8,16]:
            subset=[r for r in results if r['kernel']==kernel and r['threads']==threads]
            if subset:
                groups.append(dict(kernel=kernel,threads=threads,
                                   median_of_sample_medians_ms=statistics.median(r['median_ms'] for r in subset),
                                   worst_sample_p95_ms=max(r['p95_ms'] for r in subset),
                                   max_abs_error=max(r['max_abs_error'] for r in subset),
                                   failed_values=sum(r['failed_values'] for r in subset)))
    summary=dict(machine=platform.machine(),hostname=platform.node(),iterations=args.iterations,
                 binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                 fixture_count=len(fixtures),timing_scope='pooling only; includes worker dispatch and output writes; excludes IO, geometry/plan setup, inference and tensor conversion',
                 measurements=results,groups=groups)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
