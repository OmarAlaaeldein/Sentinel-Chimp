"""Tests for StockGraph data structures, graph traversal, and peer divergence."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from core.stock_graph import (
    StockGraph,
    StockNode,
    GraphEdge,
    RelationType,
    build_default_graph,
    PeerDivergence,
)


def test_node_and_edge_creation():
    g = StockGraph()
    node_a = StockNode("AAA", "Alpha Corp", "Technology", "Software", "Mega", "Leader in SaaS")
    node_b = StockNode("BBB", "Beta Corp", "Technology", "Hardware", "Large", "Hardware vendor")
    g.add_node(node_a)
    g.add_node(node_b)

    assert g.get_node("AAA") == node_a
    assert g.get_node("bbb") == node_b
    assert g.get_node("CCC") is None

    edge = GraphEdge("AAA", "BBB", RelationType.SUPPLIER_TO.value, 0.9, "Supplies cloud services")
    g.add_edge(edge)

    assert len(g.edges) == 1
    assert g.edges[0].source == "AAA"
    assert g.edges[0].target == "BBB"


def test_bidirectional_edge():
    g = StockGraph()
    g.add_node(StockNode("AAA", "Alpha", "Tech", "Software"))
    g.add_node(StockNode("BBB", "Beta", "Tech", "Software"))

    g.add_edge(GraphEdge("AAA", "BBB", RelationType.COMPETITOR.value, 1.0, "Direct rivals", bidirectional=True))

    # Should appear in both AAA's and BBB's neighbors
    nbrs_a = [n.ticker for n, _ in g.get_neighbors("AAA")]
    nbrs_b = [n.ticker for n, _ in g.get_neighbors("BBB")]
    assert "BBB" in nbrs_a
    assert "AAA" in nbrs_b


def test_neighbor_traversal_and_depth():
    g = StockGraph()
    g.add_node(StockNode("A", "Alpha", "Tech", "A"))
    g.add_node(StockNode("B", "Beta", "Tech", "B"))
    g.add_node(StockNode("C", "Gamma", "Tech", "C"))

    g.add_edge(GraphEdge("A", "B", RelationType.SUPPLIER_TO.value))
    g.add_edge(GraphEdge("B", "C", RelationType.CUSTOMER_OF.value))

    # Depth 1 from A: only B
    d1 = [n.ticker for n, _ in g.get_neighbors("A", depth=1)]
    assert d1 == ["B"]

    # Depth 2 from A: B and C
    d2 = [n.ticker for n, _ in g.get_neighbors("A", depth=2)]
    assert set(d2) == {"B", "C"}

    # Filter by relation
    filtered = [n.ticker for n, _ in g.get_neighbors("A", depth=2, relation_types=[RelationType.SUPPLIER_TO.value])]
    assert filtered == ["B"]


def test_find_paths():
    g = StockGraph()
    g.add_node(StockNode("A", "Alpha", "Tech", "A"))
    g.add_node(StockNode("B", "Beta", "Tech", "B"))
    g.add_node(StockNode("C", "Gamma", "Tech", "C"))

    g.add_edge(GraphEdge("A", "B", RelationType.SUPPLIER_TO.value))
    g.add_edge(GraphEdge("B", "C", RelationType.SUPPLIER_TO.value))
    g.add_edge(GraphEdge("A", "C", RelationType.INVESTED_IN.value))

    paths = g.find_paths("A", "C", max_depth=3)
    assert ["A", "B", "C"] in paths
    assert ["A", "C"] in paths


def test_subgraph():
    default_g = build_default_graph()
    sub = default_g.subgraph(["NVDA", "AMD"], depth=0)
    assert set(sub.nodes.keys()) == {"NVDA", "AMD"}
    assert all(e.source in {"NVDA", "AMD"} and e.target in {"NVDA", "AMD"} for e in sub.edges)


def test_serialization_and_json_roundtrip(tmp_path):
    g = StockGraph()
    g.add_node(StockNode("AAA", "Alpha", "Tech", "SubTech", "Mega", "Desc"))
    g.add_node(StockNode("BBB", "Beta", "Tech", "SubTech", "Large", "Desc2"))
    g.add_edge(GraphEdge("AAA", "BBB", RelationType.POWER_PARTNER.value, 0.8, "Power PPA"))

    d = g.to_dict()
    g2 = StockGraph.from_dict(d)
    assert len(g2.nodes) == 2
    assert g2.get_node("AAA").name == "Alpha"
    assert g2.edges[0].relation == RelationType.POWER_PARTNER.value

    # Test file save and load
    file_path = tmp_path / "custom_graph.json"
    g.save_json(file_path)
    assert file_path.exists()

    g_loaded = StockGraph.load_json(file_path)
    assert set(g_loaded.nodes.keys()) == {"AAA", "BBB"}
    assert g_loaded.edges[0].description == "Power PPA"


def test_default_graph_integrity():
    g = build_default_graph()
    assert len(g.nodes) >= 20
    assert len(g.edges) >= 25
    assert "NVDA" in g.nodes
    assert "TSM" in g.nodes
    assert "AMD" in g.nodes
    assert "VST" in g.nodes
    assert "MSFT" in g.nodes

    # NVDA competitors should include AMD
    nvda_nbrs = {n.ticker: edge.relation for n, edge in g.get_neighbors("NVDA")}
    assert "AMD" in nvda_nbrs
    assert nvda_nbrs["AMD"] == RelationType.COMPETITOR.value


def test_analyze_divergence_with_mock_data():
    g = StockGraph()
    g.add_node(StockNode("LEAD", "Leader Corp", "Tech", "Semis"))
    g.add_node(StockNode("LAG", "Laggard Corp", "Tech", "Semis"))
    g.add_edge(GraphEdge("LAG", "LEAD", RelationType.COMPETITOR.value, bidirectional=True))

    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    lead_prices = np.linspace(100.0, 120.0, 20)  # +20%
    lag_prices = np.linspace(100.0, 102.0, 20)   # +2%

    lead_df = pd.DataFrame({"Close": lead_prices}, index=dates)
    lag_df = pd.DataFrame({"Close": lag_prices}, index=dates)

    class MockProvider:
        def create_ticker(self, sym):
            return sym

        def fetch_history(self, ticker, period="1mo", interval="1d"):
            if ticker == "LEAD":
                return lead_df
            elif ticker == "LAG":
                return lag_df
            return pd.DataFrame()

    divs = g.analyze_divergence(MockProvider(), "LAG", period="1mo")
    assert len(divs) == 1
    d = divs[0]
    assert d.target_ticker == "LAG"
    assert d.peer_ticker == "LEAD"
    assert d.divergence_status == "LAGGING_PEER"
    assert d.spread_pct > 0.10
    assert "LAG lagging LEAD" in d.summary


def test_nasdaq_100_full_coverage():
    from core.graph_data import get_nasdaq_100_tickers
    g = build_default_graph()
    ndx_tickers = get_nasdaq_100_tickers()
    assert len(ndx_tickers) >= 100

    missing = [t for t in ndx_tickers if t not in g.nodes]
    assert not missing, f"Missing NASDAQ-100 tickers: {missing}"


def test_sp500_sectors_coverage():
    g = build_default_graph()
    assert len(g.nodes) >= 200
    assert len(g.edges) >= 400

    # Verify all 11 GICS sectors are represented
    sectors = {node.sector for node in g.nodes.values()}
    expected_sectors = {
        "Technology", "Semiconductors", "Communication Services",
        "Consumer Discretionary", "Consumer Staples", "Healthcare",
        "Financials", "Energy", "Industrials", "Utilities",
        "Real Estate", "Materials", "Index"
    }
    assert expected_sectors.issubset(sectors)


def test_non_tech_peers_resolution():
    g = build_default_graph()

    # Financials: JPM
    jpm_nbrs = {n.ticker: e.relation for n, e in g.get_neighbors("JPM")}
    assert "BAC" in jpm_nbrs
    assert "GS" in jpm_nbrs

    # Healthcare: LLY
    lly_nbrs = {n.ticker: e.relation for n, e in g.get_neighbors("LLY")}
    assert "NVO" in lly_nbrs

    # Energy: XOM
    xom_nbrs = {n.ticker: e.relation for n, e in g.get_neighbors("XOM")}
    assert "CVX" in xom_nbrs
    assert "COP" in xom_nbrs

    # Consumer Staples: COST
    cost_nbrs = {n.ticker: e.relation for n, e in g.get_neighbors("COST")}
    assert "WMT" in cost_nbrs

    # Industrials: CAT
    cat_nbrs = {n.ticker: e.relation for n, e in g.get_neighbors("CAT")}
    assert "DE" in cat_nbrs


def test_cross_sector_supply_chain_paths():
    g = build_default_graph()

    # Nuclear Fuel -> Utility Power -> Hyperscaler Cloud
    paths = g.find_paths("CCJ", "MSFT", max_depth=3)
    assert any("CEG" in p for p in paths)

    # Semi Equipment -> Pure-Play Foundry -> AI Accelerator -> AI Server Rack
    semi_paths = g.find_paths("ASML", "SMCI", max_depth=4)
    assert any("TSM" in p and "NVDA" in p for p in semi_paths)


def test_graph_connectivity_no_isolated_nodes():
    g = build_default_graph()
    degrees = {sym: len(g.get_neighbors(sym)) for sym in g.nodes}
    isolated = [sym for sym, d in degrees.items() if d == 0]
    assert not isolated, f"Isolated nodes with no relationships: {isolated}"

