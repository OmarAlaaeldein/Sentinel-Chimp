"""Sectivia CSV importer + merge policy (fixture-based; no network in default CI)."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.sectivia_import import (
    SECTIVIA_ATTRIBUTION,
    load_cached_sectivia_graph,
    merge_sectivia_into,
    relations_from_csv,
    refresh_sectivia_cache,
)
from core.stock_graph import (
    GraphEdge,
    RelationType,
    StockGraph,
    StockNode,
    build_default_graph,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sectivia"


def test_relations_csv_supplier_to_and_customer_of():
    g, meta = relations_from_csv(
        FIXTURE / "sectivia-relations.csv",
        FIXTURE / "sectivia-companies.csv",
        listed_only=True,
        add_customer_of=True,
    )
    assert meta["attribution"] == SECTIVIA_ATTRIBUTION
    assert g.get_node("MU") is not None
    assert g.get_node("NVDA") is not None
    # SpaceX unlisted / blank ticker skipped
    assert g.get_node("ASTS") is None or all(
        e.source != "ASTS" or e.target for e in g.edges
    )
    rels = {(e.source, e.target, e.relation) for e in g.edges}
    assert ("MU", "NVDA", RelationType.SUPPLIER_TO.value) in rels
    assert ("NVDA", "MU", RelationType.CUSTOMER_OF.value) in rels
    assert ("NVDA", "MSFT", RelationType.SUPPLIER_TO.value) in rels
    assert ("ON", "TSLA", RelationType.SUPPLIER_TO.value) in rels


def test_listed_only_skips_blank_supplier():
    g, meta = relations_from_csv(
        FIXTURE / "sectivia-relations.csv",
        FIXTURE / "sectivia-companies.csv",
        listed_only=True,
        add_customer_of=False,
    )
    # SpaceX row has empty supplier_ticker → skipped
    assert all(e.source for e in g.edges)
    assert meta["skipped_relations"] >= 1


def test_merge_prefer_curated():
    base = StockGraph()
    base.add_node(StockNode("MU", "Micron", "Semiconductors", "Memory"))
    base.add_node(StockNode("NVDA", "NVIDIA", "Semiconductors", "GPU"))
    base.add_edge(
        GraphEdge("MU", "NVDA", RelationType.SUPPLIER_TO.value, 0.95, "curated HBM")
    )

    overlay, _ = relations_from_csv(
        FIXTURE / "sectivia-relations.csv",
        FIXTURE / "sectivia-companies.csv",
        add_customer_of=False,
    )
    stats = merge_sectivia_into(base, overlay, prefer_curated=True)
    assert stats["edges_skipped_duplicate"] >= 1
    # Curated weight preserved
    mu_nvda = [e for e in base.edges if e.source == "MU" and e.target == "NVDA" and e.relation == "SUPPLIER_TO"]
    assert len(mu_nvda) == 1
    assert mu_nvda[0].weight == 0.95
    assert "curated" in mu_nvda[0].description.lower()
    # New Sectivia edge still merged
    assert any(e.source == "ON" and e.target == "TSLA" for e in base.edges)
    assert stats["attribution"].startswith("Supply-chain data: Sectivia")


def test_load_vendored_cache():
    """Repo data/sectivia/ should be present for offline default graph."""
    repo_cache = Path(__file__).resolve().parents[1] / "data" / "sectivia"
    if not (repo_cache / "sectivia-relations.csv").exists():
        pytest.skip("vendored Sectivia cache not present")
    g, meta = load_cached_sectivia_graph(repo_cache, add_customer_of=True)
    assert len(g.edges) > 100
    assert meta["attribution"] == SECTIVIA_ATTRIBUTION


def test_build_default_graph_merges_sectivia():
    curated_only = build_default_graph(include_sectivia=False)
    merged = build_default_graph(include_sectivia=True)
    # Merged should be at least as large as curated
    assert len(merged.nodes) >= len(curated_only.nodes)
    assert len(merged.edges) >= len(curated_only.edges)


def test_refresh_sectivia_cache_live(tmp_path):
    """Optional live fetch — set SENTINEL_TEST_NETWORK=1 to enable."""
    import os
    if os.environ.get("SENTINEL_TEST_NETWORK") != "1":
        pytest.skip("set SENTINEL_TEST_NETWORK=1 for live Sectivia fetch")
    paths = refresh_sectivia_cache(tmp_path, include_json=False)
    assert Path(paths["relations"]).exists()
    assert Path(paths["companies"]).exists()
    assert "Sectivia" in Path(paths["attribution"]).read_text(encoding="utf-8")
