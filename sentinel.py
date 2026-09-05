"""Sentinel-Chimp thin entrypoint.

Keeps Stocks.cmd / build scripts working via ``python sentinel.py``.
Re-exports core symbols so existing tests can ``from sentinel import ...``.
"""
import tkinter as tk

from core.pricing import VegaChimpCore
from core.technicals import calculate_technicals
from core.sentiment import SentimentEngine, sentiment_engine
from core.data import DataProvider, YFinanceProvider
from ui.tooltip import Tooltip
from main.app import MarketApp

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


if __name__ == "__main__":
    try:
        import pyi_splash
        if pyi_splash.is_alive():
            pyi_splash.close()
    except ImportError:
        pass
    root = tk.Tk()
    app = MarketApp(root)
    root.mainloop()
