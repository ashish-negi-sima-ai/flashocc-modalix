"""Stream nuScenes metadata and select deterministic, disjoint scene subsets."""
import json
from pathlib import Path


def records(path, chunk_size=1024 * 1024):
    """Read a JSON array without loading sample_data/ego_pose tables into RAM."""
    decoder = json.JSONDecoder()
    with Path(path).open() as stream:
        buffer = ''
        ended = False

        def refill():
            nonlocal buffer, ended
            chunk = stream.read(chunk_size)
            buffer += chunk
            ended = not chunk

        def token():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not ended:
                refill()
                buffer = buffer.lstrip()
            return buffer[:1]

        if token() != '[':
            raise ValueError(f'Expected JSON array: {path}')
        buffer = buffer[1:]
        first = True
        while True:
            if token() == ']':
                buffer = buffer[1:]
                if token():
                    raise ValueError(f'Trailing JSON content: {path}')
                return
            if not first:
                if token() != ',':
                    raise ValueError(f'Missing JSON separator: {path}')
                buffer = buffer[1:]
                if token() == ']':
                    raise ValueError(f'Trailing JSON comma: {path}')
            token()
            while True:
                try:
                    item, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if ended:
                        raise ValueError(f'Truncated or invalid JSON: {path}') from None
                    refill()
            if not isinstance(item, dict):
                raise ValueError(f'Expected metadata object: {path}')
            buffer = buffer[end:]
            first = False
            yield item


def select_samples(scenes, samples, names, count):
    if count < 0:
        raise ValueError('Sample count must be nonnegative')
    by_name = {s['name']: s for s in scenes}
    if len(set(names)) != len(names):
        raise ValueError('Duplicate scene names')
    sequences = []
    for name in names:
        scene = by_name[name]
        sequence, token, seen = [], scene['first_sample_token'], set()
        while token:
            if token in seen:
                raise ValueError(f'Cyclic sample chain: {name}')
            seen.add(token)
            sample = samples[token]
            if sample['scene_token'] != scene['token']:
                raise ValueError(f'Sample outside scene: {name}')
            sequence.append(sample)
            token = sample['next']
        sequences.append(sequence)
    if count > sum(map(len, sequences)):
        raise ValueError('Requested more unique samples than the selected scenes contain')
    quotas = [0] * len(sequences)
    remaining = count
    while remaining:
        for i, sequence in enumerate(sequences):
            if remaining and quotas[i] < len(sequence):
                quotas[i] += 1
                remaining -= 1
    selected = []
    for sequence, quota in zip(sequences, quotas):
        # Same integer linspace selection as the original mini preparation.
        indexes = [int(i * (len(sequence) - 1) / (quota - 1)) for i in range(quota)] if quota > 1 else [0] * quota
        selected.extend(sequence[i] for i in indexes)
    return selected


def load_selected(data, version, train_names, val_names, train_count, val_count, cameras,
                  selection=None):
    if set(train_names) & set(val_names):
        raise ValueError('Calibration and validation scenes must be disjoint')
    directory = Path(data) / version
    scenes = list(records(directory / 'scene.json'))
    scene_tokens = {s['token'] for s in scenes if s['name'] in set(train_names + val_names)}
    samples = {r['token']: r for r in records(directory / 'sample.json') if r['scene_token'] in scene_tokens}
    if selection is None:
        selected = [('calibration', s) for s in select_samples(scenes, samples, train_names, train_count)]
        selected += [('validation', s) for s in select_samples(scenes, samples, val_names, val_count)]
    else:
        by_scene = {s['token']: s['name'] for s in scenes}
        allowed = {'calibration': set(train_names), 'validation': set(val_names)}
        selected = []
        for record in selection:
            split, sample = record['split'], samples[record['token']]
            if split not in allowed or by_scene[sample['scene_token']] not in allowed[split]:
                raise ValueError('Selected token violates the official scene split')
            selected.append((split, sample))
        if len({s['token'] for _, s in selected}) != len(selected):
            raise ValueError('Duplicate selected tokens')
    wanted = {s['token'] for _, s in selected}
    sensors = {r['token']: r for r in records(directory / 'sensor.json')}
    sensor_for_calibration = {r['token']: r['sensor_token'] for r in records(directory / 'calibrated_sensor.json')}
    sample_data = {}
    for record in records(directory / 'sample_data.json'):
        if record['sample_token'] not in wanted or not record['is_key_frame']:
            continue
        channel = sensors[sensor_for_calibration[record['calibrated_sensor_token']]]['channel']
        if channel in cameras:
            sample_data[record['token']] = record
            samples[record['sample_token']].setdefault('data', {})[channel] = record['token']
    calibration_tokens = {r['calibrated_sensor_token'] for r in sample_data.values()}
    ego_tokens = {r['ego_pose_token'] for r in sample_data.values()}
    tables = {'scene': {s['token']: s for s in scenes}, 'sample': samples,
              'sample_data': sample_data,
              'calibrated_sensor': {r['token']: r for r in records(directory / 'calibrated_sensor.json') if r['token'] in calibration_tokens},
              'ego_pose': {r['token']: r for r in records(directory / 'ego_pose.json') if r['token'] in ego_tokens}}
    for _, sample in selected:
        if set(sample.get('data', {})) != set(cameras):
            raise ValueError(f'Sample lacks six keyframe cameras: {sample["token"]}')
    return tables, selected
