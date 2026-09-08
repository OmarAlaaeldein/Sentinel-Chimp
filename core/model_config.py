"""Versioned model profiles, independent of GUI preferences."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

from .ollama import LocalModelError


def config_path(explicit=None):
    override = explicit or os.environ.get('SENTINEL_CONFIG')
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == 'darwin':
        base = Path.home() / 'Library/Application Support/SentinelChimp'
    elif sys.platform == 'win32':
        base = Path(os.environ.get('APPDATA', Path.home())) / 'SentinelChimp'
    else:
        base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'sentinel-chimp'
    return base / 'config.json'


def validate(data):
    if not isinstance(data, dict) or type(data.get('schema_version')) is not int or data['schema_version'] != 1:
        raise ValueError('Expected configuration schema_version 1.')
    if set(data) - {'schema_version', 'default_profile', 'profiles'}:
        raise ValueError('Unknown configuration fields.')
    profiles = data.get('profiles')
    if not isinstance(profiles, dict):
        raise ValueError('profiles must be an object.')
    for name, profile in profiles.items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', name):
            raise ValueError('Profile names must contain 1–64 letters, numbers, dots, underscores or dashes.')
        if not isinstance(profile, dict) or set(profile) != {'model', 'options'}:
            raise ValueError('Each profile needs model and options fields.')
        model = profile['model']
        if not isinstance(model, str) or not model or len(model) > 512 or any(c.isspace() or ord(c) < 32 for c in model):
            raise ValueError('Invalid model name.')
        options = profile['options']
        if not isinstance(options, dict) or set(options) != {'temperature', 'num_ctx', 'num_predict'}:
            raise ValueError('Options must contain temperature, num_ctx and num_predict.')
        for key, lower, upper in [('temperature', 0, 2), ('num_ctx', 256, 131072), ('num_predict', 1, 8192)]:
            value = options[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f'{key} must be between {lower} and {upper}.')
            if key != 'temperature' and type(value) is not int:
                raise ValueError(f'{key} must be an integer.')
    selected = data.get('default_profile')
    if selected is not None and (not isinstance(selected, str) or selected not in profiles):
        raise ValueError('default_profile must refer to an existing profile.')
    return data


def load(path):
    try:
        if not path.exists():
            return {'schema_version': 1, 'default_profile': None, 'profiles': {}}
        return validate(json.loads(path.read_text(encoding='utf-8')))
    except (OSError, ValueError, TypeError) as exc:
        raise LocalModelError('INVALID_CONFIG', f'Cannot read {path}: {exc}') from exc


@contextmanager
def _lock(path):
    # Keep a stable lock inode; OS releases the lock even after a process crash.
    with open(str(path) + '.lock', 'a+b') as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b'0')
            handle.flush()
        deadline = time.monotonic() + 5
        while True:
            try:
                if os.name == 'nt':
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise LocalModelError('CONFIG_BUSY', 'Configuration is busy; retry shortly.')
                time.sleep(.05)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def update(path, mutate):
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock(path):
            data = load(path)
            mutate(data)
            validate(data)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
                temporary = handle.name
                json.dump(data, handle, indent=2, allow_nan=False)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        return data
    except (OSError, ValueError) as exc:
        raise LocalModelError('CONFIG_WRITE_FAILED', f'Cannot update {path}: {exc}') from exc
    finally:
        if temporary:
            os.unlink(temporary)
