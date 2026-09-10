"""GUI graph helpers testable without a display."""
from __future__ import annotations

from core.graph_service import (
    categorize_peers,
    format_divergence_summary,
    load_graph,
    show_connections,
)
from core.stock_graph import PeerDivergence
from ui.stock_graph import resolve_graph_ticker


def test_resolve_graph_ticker_prefers_entry():
    assert resolve_graph_ticker("amd", "NVDA") == "AMD"
    assert resolve_graph_ticker("  ", "nvda") == "NVDA"
    assert resolve_graph_ticker("", None) == ""


def test_categorize_peers_shared_helper():
    g = load_graph()  # default includes Sectivia when cached
    data = categorize_peers(g, "AMD")
    assert data["ticker"] == "AMD"
    assert data["peer_count"] >= 1
    assert "peer_categories" in data


def test_show_connections_summary():
    g = load_graph()
    summary = show_connections(g, ticker=None)
    assert summary["action"] == "summary"
    assert summary["total_nodes"] > 50


def test_format_divergence_summary():
    divs = [
        PeerDivergence(
            target_ticker="AMD",
            peer_ticker="NVDA",
            relation="COMPETITOR",
            target_return_pct=0.01,
            peer_return_pct=0.10,
            spread_pct=0.09,
            correlation=0.8,
            z_score=1.2,
            divergence_status="LAGGING_PEER",
            summary="AMD lagging NVDA by +9.0%.",
        )
    ]
    lines = format_divergence_summary(divs)
    assert lines and "NVDA" in lines[0]
    assert format_divergence_summary([])[0].startswith("No peer")
