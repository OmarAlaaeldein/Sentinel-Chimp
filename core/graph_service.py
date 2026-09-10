"""Shared Stock Relationship Graph helpers for CLI and GUI (no Tk / display deps)."""
from __future__ import annotations

import os
import tempfile
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.stock_graph import (
    PeerDivergence,
    StockGraph,
    build_default_graph,
)


def load_graph(graph_file: Optional[str] = None) -> StockGraph:
    """Load a custom JSON graph or the curated default universe."""
    if graph_file:
        p = Path(graph_file)
        if not p.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_file}")
        return StockGraph.load_json(p)
    return build_default_graph()


def categorize_peers(graph: StockGraph, ticker: str) -> Dict[str, Any]:
    """Return peers payload matching ``graph peers`` CLI JSON shape."""
    sym = ticker.upper().strip()
    node = graph.get_node(sym)
    if not node:
        raise ValueError(f"Ticker {sym!r} not found in stock graph.")
    neighbors = graph.get_neighbors(sym, depth=1)
    categorized: Dict[str, List[dict]] = {}
    for n, edge in neighbors:
        categorized.setdefault(edge.relation, []).append(
            {
                "ticker": n.ticker,
                "name": n.name,
                "sector": n.sector,
                "sub_industry": n.sub_industry,
                "description": edge.description,
                "weight": edge.weight,
            }
        )
    return {
        "action": "peers",
        "ticker": sym,
        "node": node.to_dict(),
        "peer_categories": categorized,
        "peer_count": sum(len(v) for v in categorized.values()),
    }


def show_connections(
    graph: StockGraph, ticker: Optional[str] = None, depth: int = 1
) -> Dict[str, Any]:
    """Return show/summary payload matching ``graph show`` CLI JSON shape."""
    if not ticker:
        nodes = [n.to_dict() for n in graph.nodes.values()]
        return {
            "action": "summary",
            "total_nodes": len(nodes),
            "total_edges": len(graph.edges),
            "sectors": sorted({n["sector"] for n in nodes}),
            "tickers": sorted(n["ticker"] for n in nodes),
        }

    sym = ticker.upper().strip()
    node = graph.get_node(sym)
    if not node:
        raise ValueError(f"Ticker {sym!r} not found in stock graph.")
    neighbors = graph.get_neighbors(sym, depth=depth)
    connections = [
        {
            "neighbor": n.ticker,
            "name": n.name,
            "sector": n.sector,
            "relation": edge.relation,
            "description": edge.description,
            "weight": edge.weight,
        }
        for n, edge in neighbors
    ]
    return {
        "action": "show",
        "ticker": sym,
        "node": node.to_dict(),
        "depth": depth,
        "connection_count": len(connections),
        "connections": connections,
    }


def format_divergence_summary(
    divergences: Sequence[PeerDivergence] | Sequence[dict],
    *,
    max_items: int = 6,
) -> List[str]:
    """Short human-readable divergence lines for GUI / CLI tips."""
    lines: List[str] = []
    items = list(divergences)
    if not items:
        return ["No peer divergence data available (missing history or no peers)."]

    for item in items[: max(0, max_items)]:
        if isinstance(item, PeerDivergence):
            d = item.to_dict()
        else:
            d = dict(item)
        status = d.get("divergence_status", "IN_SYNC")
        summary = d.get("summary")
        if not summary:
            summary = (
                f"{d.get('target_ticker', '?')} vs {d.get('peer_ticker', '?')}: "
                f"{float(d.get('spread_pct', 0)):+.1%} ({status})"
            )
        if status != "IN_SYNC" or len(lines) < 2:
            lines.append(summary)
    return lines or [items[0].summary if isinstance(items[0], PeerDivergence) else str(items[0])]


def export_graph_html_file(
    graph: StockGraph,
    path: Path | str,
    *,
    center_ticker: Optional[str] = None,
    depth: int = 1,
    dim: str = "2d",
) -> str:
    """Build Plotly network and write standalone HTML; returns absolute path."""
    from core.graph_viz import build_network_plotly, export_graph_html

    fig = build_network_plotly(
        graph, center_ticker=center_ticker, depth=depth, dim=dim
    )
    return export_graph_html(fig, path)


def export_graph_temp_html(
    graph: StockGraph,
    *,
    center_ticker: Optional[str] = None,
    depth: int = 1,
    dim: str = "2d",
    prefix: str = "sentinel_graph_",
) -> str:
    """Write graph HTML to a temp file (Lite Mode friendly) and return its path."""
    sym = (center_ticker or "market").upper()
    fd, name = tempfile.mkstemp(prefix=f"{prefix}{sym}_{dim}_", suffix=".html")
    os.close(fd)
    return export_graph_html_file(
        graph, name, center_ticker=center_ticker, depth=depth, dim=dim
    )


def open_html_in_browser(path: str | Path) -> str:
    """Open a local HTML file in the default browser; returns file:// URL used."""
    resolved = Path(path).resolve()
    url = resolved.as_uri()
    webbrowser.open(url)
    return url
