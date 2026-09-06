"""Sentinel-Chimp thin entrypoint.

Keeps Stocks.cmd / build scripts working via ``python sentinel.py``.
* No args → GUI (lazy-imports tkinter / MarketApp).
* Subcommands (``analyze`` / ``scan`` …) → headless CLI (no tkinter).

Re-exports core symbols so existing tests can ``from sentinel import ...``.
"""
from __future__ import annotations

import sys

from core.pricing import VegaChimpCore
from core.technicals import calculate_technicals
from core.sentiment import SentimentEngine, sentiment_engine
from core.data import DataProvider, YFinanceProvider

__all__ = [
    "VegaChimpCore",
    "calculate_technicals",
    "SentimentEngine",
    "sentiment_engine",
    "DataProvider",
    "YFinanceProvider",
    "Tooltip",
    "MarketApp",
]


def __getattr__(name: str):
    """Lazy GUI symbols so ``import sentinel`` stays headless-safe."""
    if name == "MarketApp":
        from main.app import MarketApp
        return MarketApp
    if name == "Tooltip":
        from ui.tooltip import Tooltip
        return Tooltip
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _launch_gui() -> None:
    try:
        import pyi_splash
        if pyi_splash.is_alive():
            pyi_splash.close()
    except ImportError:
        pass
    import tkinter as tk
    from main.app import MarketApp
    root = tk.Tk()
    app = MarketApp(root)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        from main.cli import main as cli_main
        raise SystemExit(cli_main(sys.argv[1:]))
    _launch_gui()
