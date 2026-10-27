"""Make a batch-one QAT BEV MPK compatible with Neat's rank-three MLA output.

Only the post-MLA dequantization shape metadata changes. The ELF, quantization
parameters, buffer sizes, and every other archived payload remain unchanged.
"""
import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error('Keep the original archive; choose a separate output path')
    with tarfile.open(args.source) as archive:
        members = archive.getmembers()
        payloads = {m.name: archive.extractfile(m).read() for m in members if m.isfile()}
    names = [n for n in payloads if n.endswith('_mpk.json')]
    if len(names) != 1:
        raise ValueError('Expected exactly one MPK manifest')
    name = names[0]
    original = json.loads(payloads[name])
    manifest = copy.deepcopy(original)
    mla = [p for p in manifest['plugins'] if p['processor'] == 'MLA']
    if len(mla) != 1 or mla[0]['config_params']['actual_batch_size'] != 1:
        raise ValueError('This adapter requires one MLA stage with batch one')
    producers = {n['name'] for n in mla[0]['output_nodes']}
    changes = []
    for plugin in manifest['plugins']:
        config = plugin.get('config_params', {})
        if config.get('kernel') != 'dequantization_transform':
            continue
        if not any(n['name'] in producers for n in plugin['input_nodes']):
            continue
        params = config['params']
        if params.get('input_data_type') != 'int8':
            raise ValueError('Expected INT8 output dequantization')
        for key in ['input_shapes', 'output_shapes']:
            for i, shape in enumerate(params[key]):
                if shape != [1, 200, 200, 288]:
                    raise ValueError(f'Unexpected BEV output shape: {shape}')
                reduced = shape[1:]
                assert math.prod(shape) == math.prod(reduced)
                params[key][i] = reduced
                changes.append({'plugin': plugin['name'], 'field': key,
                                'index': i, 'before': shape, 'after': reduced})
    if len(changes) != 2:
        raise ValueError('Expected only the BEV input/output dequantization shapes')
    # Assert that no parameters outside those two shape fields were changed.
    restored = copy.deepcopy(manifest)
    for change in changes:
        plugin = next(p for p in restored['plugins'] if p['name'] == change['plugin'])
        plugin['config_params']['params'][change['field']][change['index']] = change['before']
    assert restored == original
    updated = dict(payloads)
    updated[name] = (json.dumps(manifest, indent=2) + '\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, 'w:gz') as archive:
        for member in members:
            member = copy.copy(member)
            if member.isfile():
                member.size = len(updated[member.name])
                archive.addfile(member, io.BytesIO(updated[member.name]))
            else:
                archive.addfile(member)
    with tarfile.open(args.output) as archive:
        for member in archive.getmembers():
            if member.isfile():
                assert archive.extractfile(member).read() == updated[member.name]
    report = {'source': str(args.source.resolve()), 'output': str(args.output.resolve()),
              'source_sha256': digest(args.source.read_bytes()),
              'output_sha256': digest(args.output.read_bytes()), 'changes': changes,
              'elf_unchanged': True,
              'elf_sha256': {n: digest(v) for n, v in payloads.items() if n.endswith('.elf')},
              'all_other_payloads_unchanged': True}
    args.output.with_suffix('.compatibility.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
