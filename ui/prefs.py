"""Lightweight JSON preferences for GUI toggles."""
from __future__ import annotations

import json
import os
from typing import Any, Dict

DEFAULT_PREFS: Dict[str, Any] = {
    "use_garch_blend": False,
    "use_smile_vol": False,
    "show_prob_cone": True,
}


def prefs_path(root_dir: str) -> str:
    return os.path.join(root_dir, "user_prefs.json")


def load_prefs(root_dir: str) -> Dict[str, Any]:
    path = prefs_path(root_dir)
    data = dict(DEFAULT_PREFS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            for k in DEFAULT_PREFS:
                if k in loaded:
                    data[k] = bool(loaded[k])
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return data


def save_prefs(root_dir: str, **kwargs) -> None:
    path = prefs_path(root_dir)
    data = load_prefs(root_dir)
    for k, v in kwargs.items():
        if k in DEFAULT_PREFS:
            data[k] = bool(v)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass
