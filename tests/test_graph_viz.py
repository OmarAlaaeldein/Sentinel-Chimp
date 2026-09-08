"""Tests for Plotly 2D/3D network visualizer and HTML export."""
from __future__ import annotations

from pathlib import Path
import pytest
import plotly.graph_objects as go

from core.stock_graph import build_default_graph, StockGraph, StockNode, GraphEdge
from core.graph_viz import compute_spring_layout, build_network_plotly, export_graph_html


def test_spring_layout_dimensions():
    tickers = ["A", "B", "C", "D"]
    edges = [("A", "B", 1.0), ("B", "C", 1.0), ("C", "D", 1.0)]

    # 2D
    pos_2d = compute_spring_layout(tickers, edges, dim=2, iterations=30)
    assert len(pos_2d) == 4
    for t in tickers:
        assert pos_2d[t].shape == (2,)

    # 3D
    pos_3d = compute_spring_layout(tickers, edges, dim=3, iterations=30)
    assert len(pos_3d) == 4
    for t in tickers:
        assert pos_3d[t].shape == (3,)


def test_build_network_plotly_2d():
    g = build_default_graph()
    fig = build_network_plotly(g, dim="2d")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    # Check that scatter traces exist for nodes and lines
    trace_types = [t.type for t in fig.data]
    assert "scatter" in trace_types


def test_build_network_plotly_3d():
    g = build_default_graph()
    fig = build_network_plotly(g, center_ticker="NVDA", depth=1, dim="3d")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    trace_types = [t.type for t in fig.data]
    assert "scatter3d" in trace_types


def test_export_graph_html(tmp_path):
    g = build_default_graph()
    fig = build_network_plotly(g, center_ticker="MSFT", depth=1, dim="2d")
    out_file = tmp_path / "test_network.html"
    saved = export_graph_html(fig, out_file)

    assert Path(saved).exists()
    content = Path(saved).read_text(encoding="utf-8")
    assert "plotly" in content.lower()
    assert "MSFT" in content
