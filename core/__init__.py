"""Core business logic packages for Sentinel-Chimp."""
from importlib import import_module

_MODULES = {
    "VegaChimpCore": "pricing", "calculate_technicals": "technicals",
    "SentimentEngine": "sentiment", "sentiment_engine": "sentiment",
    "DataProvider": "data", "YFinanceProvider": "data",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "VegaChimpCore",
    "calculate_technicals",
    "SentimentEngine",
    "sentiment_engine",
    "DataProvider",
    "YFinanceProvider",
]
