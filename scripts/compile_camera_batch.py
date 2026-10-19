"""Compile the saved BF16 camera model for a requested batch, without requantizing."""
import argparse
import hashlib
import json
import logging
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=6)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('--batch must be positive')
    if not shutil.which('mla-masm'):
        raise RuntimeError('Activate the model compiler environment; mla-masm must be on PATH')
    root = Path(__file__).resolve().parents[1]
    source = root / 'artifacts/compile-real/camera'
    audit = json.loads((root / 'artifacts/onnx-modalix/camera.bfloat16.audit.json').read_text())
    if audit['unsupported'] or audit['unknown']:
        raise RuntimeError('BF16 operator audit failed')
    output = args.output or root / f'artifacts/compile-batch{args.batch}/camera'
    output.mkdir(parents=True, exist_ok=True)
    name = 'flashocc_m0_camera'
    record = {'requested_batch': args.batch, 'precision': 'bfloat16', 'source': str(source),
              'source_model_sha256': hashlib.sha256((source / (name + '.sima')).read_bytes()).hexdigest(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'requantized': False, 'compiled': False, 'hardware_validated': False,
              'ground_truth_accuracy_measured': False}
    status = output / 'status.json'
    status.write_text(json.dumps(record, indent=2) + '\n')
    from afe.apis.model import Model
    model = Model.load(name, str(source))
    model.compile(output_path=str(output), batch_size=args.batch, log_level=logging.INFO)
    archive = output / (name + '_mpk.tar.gz')
    with tarfile.open(archive) as package:
        manifest = json.load(package.extractfile(name + '_mpk.json'))
        mla = json.load(package.extractfile('0_process_mla.json'))
        stats_name = next(n for n in package.getnames() if n.endswith('_mla_stats.yaml'))
        (output / 'mla_stats.yaml').write_bytes(package.extractfile(stats_name).read())
    record.update(compiled=True, archive=str(archive),
                  archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  mla_runtime_params=mla['simaai__params'], modelpack=manifest)
    status.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'compiled': True, 'mla_runtime_params': record['mla_runtime_params'],
                      'archive': str(archive)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
