"""Persistent ticker watchlist (JSON next to user_prefs)."""
from __future__ import annotations

import json
import os
import re
from typing import List

_TICKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,15}$")
DEFAULT_WATCHLIST: List[str] = ["AMD", "AAPL", "MSFT", "NVDA", "SPY"]


def watchlist_path(root_dir: str) -> str:
    return os.path.join(root_dir, "watchlist.json")


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _valid_symbol(symbol: str) -> bool:
    return bool(_TICKER_RE.match(symbol))


def load_watchlist(root_dir: str) -> List[str]:
    """Load watchlist; missing/corrupt file → default list (copy)."""
    path = watchlist_path(root_dir)
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            loaded = loaded.get("tickers", loaded.get("watchlist", []))
        if not isinstance(loaded, list):
            return list(DEFAULT_WATCHLIST)
        out: List[str] = []
        seen = set()
        for item in loaded:
            sym = _normalize_symbol(item)
            if _valid_symbol(sym) and sym not in seen:
                seen.add(sym)
                out.append(sym)
        return out if out else list(DEFAULT_WATCHLIST)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return list(DEFAULT_WATCHLIST)


def save_watchlist(root_dir: str, tickers: List[str]) -> List[str]:
    """Persist normalized unique tickers; returns the saved list."""
    out: List[str] = []
    seen = set()
    for item in tickers:
        sym = _normalize_symbol(item)
        if _valid_symbol(sym) and sym not in seen:
            seen.add(sym)
            out.append(sym)
    path = watchlist_path(root_dir)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tickers": out}, f, indent=2)
    except OSError:
        pass
    return out


def add_ticker(root_dir: str, symbol: str) -> List[str]:
    tickers = load_watchlist(root_dir)
    sym = _normalize_symbol(symbol)
    if _valid_symbol(sym) and sym not in tickers:
        tickers.append(sym)
        return save_watchlist(root_dir, tickers)
    return tickers


def remove_ticker(root_dir: str, symbol: str) -> List[str]:
    sym = _normalize_symbol(symbol)
    tickers = [t for t in load_watchlist(root_dir) if t != sym]
    return save_watchlist(root_dir, tickers)
