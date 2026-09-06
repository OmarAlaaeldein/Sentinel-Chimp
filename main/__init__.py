"""Application controller package.

``MarketApp`` is lazy-imported so CLI entry points do not pull in tkinter.
"""
from __future__ import annotations

__all__ = ["MarketApp"]


def __getattr__(name: str):
    if name == "MarketApp":
        from .app import MarketApp
        return MarketApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
