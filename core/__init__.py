"""Core business logic packages for Sentinel-Chimp."""
from importlib import import_module

_MODULES = {
    "VegaChimpCore": "pricing", "calculate_technicals": "technicals",
    "DataProvider": "data", "YFinanceProvider": "data",
    "StockGraph": "stock_graph", "StockNode": "stock_graph",
    "GraphEdge": "stock_graph", "build_default_graph": "stock_graph",
    "backend_info": "laya_decisions", "score_headlines": "laya_decisions",
    "classify_decision": "laya_decisions", "laya_opt_in": "laya_decisions",
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
    "DataProvider",
    "YFinanceProvider",
    "StockGraph",
    "StockNode",
    "GraphEdge",
    "build_default_graph",
    "backend_info",
    "score_headlines",
    "classify_decision",
    "laya_opt_in",
]
