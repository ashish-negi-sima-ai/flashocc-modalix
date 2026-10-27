"""Shared runtime defaults: local JSON, overridden by environment and CLI."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def defaults():
    path = Path(os.environ.get('FLASHOCC_CONFIG', ROOT / 'configs/runtime.local.json'))
    data = json.loads(path.read_text()) if path.is_file() else {}
    allowed = {'device', 'insight_api', 'insight_url', 'browser_python'}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError(f'Invalid runtime configuration: {path}')
    for key in allowed:
        value = os.environ.get('FLASHOCC_' + key.upper(), data.get(key))
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f'Invalid runtime setting: {key}')
        data[key] = value
    return data


def require(parser, value, option, environment):
    if not value:
        parser.error(f'Pass {option}, set {environment}, or configure configs/runtime.local.json')
    return value
