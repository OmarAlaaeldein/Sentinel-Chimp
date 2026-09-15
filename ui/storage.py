"""Writable per-user settings and crash-safe JSON replacement."""
import json
import os
from pathlib import Path
import sys
import tempfile


def config_dir():
    override = os.environ.get("SENTINEL_CONFIG_DIR")
    if override:
        return str(Path(override).expanduser())
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return str(base / "Sentinel-Chimp")


def atomic_json(path, data):
    """Raise OSError on failure; never truncate the previous settings file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".settings-", delete=False) as stream:
            name = stream.name
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)
