"""Sectivia U.S. supply-chain importer (CC BY 4.0) for Sentinel StockGraph.

Primary public source for supplier→customer edges. Curated ``core/graph_data.py``
remains the base; Sectivia edges are **merged (union)**. Duplicate
``(source, target, relation)`` keys keep the curated edge; Sectivia-only edges
are added as ``SUPPLIER_TO`` (and optionally ``CUSTOMER_OF`` reverse) at weight
``SECTIVIA_EDGE_WEIGHT``.

Offline: reads ``data/sectivia/`` cache. Online refresh:
``refresh_sectivia_cache()``.

Attribution (required): ``Supply-chain data: Sectivia (https://sectivia.com), CC BY 4.0``
"""
from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from core.stock_graph import GraphEdge, RelationType, StockGraph, StockNode

SECTIVIA_ATTRIBUTION = "Supply-chain data: Sectivia (https://sectivia.com), CC BY 4.0"
SECTIVIA_LICENSE = "CC BY 4.0"
SECTIVIA_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
SECTIVIA_RELATIONS_URL = "https://sectivia.com/dataset/sectivia-relations.csv"
SECTIVIA_COMPANIES_URL = "https://sectivia.com/dataset/sectivia-companies.csv"
SECTIVIA_JSON_URL = "https://sectivia.com/dataset/sectivia-supply-chain.json"
SECTIVIA_EDGE_WEIGHT = 0.85

# Repo-relative cache (offline fallback)
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "sectivia"

_SECTOR_MAP = {
    "ai": "Technology",
    "energi": "Energy",
    "forsvar": "Industrials",
    "medicin": "Healthcare",
    "quantum": "Technology",
    "finans": "Financials",
}

_SEGMENT_SECTOR_HINTS = {
    "Chips & accelerators": "Semiconductors",
    "Semiconductor equipment & IP": "Semiconductors",
    "Cloud platforms": "Technology",
    "Data centers": "Technology Hardware",
    "Cooling, power & data center infrastructure": "Industrials",
    "Utilities & power producers": "Utilities",
    "Nuclear energy & uranium": "Energy",
    "Solar & wind": "Energy",
    "Batteries & energy storage": "Energy",
    "Power equipment": "Industrials",
    "Grid & electrification": "Utilities",
    # segment ids (JSON)
    "chips": "Semiconductors",
    "equip": "Semiconductors",
    "cloud": "Technology",
    "datacentre": "Technology Hardware",
    "infra": "Industrials",
    "utilities": "Utilities",
    "nuklear": "Energy",
    "solvind": "Energy",
    "batteri": "Energy",
    "power": "Industrials",
    "elnet": "Utilities",
}


def cache_paths(cache_dir: Path | str | None = None) -> Dict[str, Path]:
    root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    return {
        "dir": root,
        "relations": root / "sectivia-relations.csv",
        "companies": root / "sectivia-companies.csv",
        "json": root / "sectivia-supply-chain.json",
        "attribution": root / "ATTRIBUTION.txt",
        "fetched_at": root / "fetched_at.txt",
    }


def _download(url: str, dest: Path, timeout: float = 30.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Sentinel-Chimp/sectivia-import"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — public CC BY dataset
        dest.write_bytes(resp.read())


def refresh_sectivia_cache(
    cache_dir: Path | str | None = None,
    *,
    timeout: float = 30.0,
    include_json: bool = True,
) -> Dict[str, str]:
    """Fetch latest Sectivia CSVs (and JSON) into the local cache directory."""
    from datetime import datetime, timezone

    paths = cache_paths(cache_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    _download(SECTIVIA_RELATIONS_URL, paths["relations"], timeout=timeout)
    _download(SECTIVIA_COMPANIES_URL, paths["companies"], timeout=timeout)
    if include_json:
        _download(SECTIVIA_JSON_URL, paths["json"], timeout=timeout)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    paths["fetched_at"].write_text(stamp + "\n", encoding="utf-8")
    paths["attribution"].write_text(
        "\n".join(
            [
                SECTIVIA_ATTRIBUTION,
                f"License: {SECTIVIA_LICENSE_URL}",
                f"Source relations: {SECTIVIA_RELATIONS_URL}",
                f"Source companies: {SECTIVIA_COMPANIES_URL}",
                f"Source JSON: {SECTIVIA_JSON_URL}",
                "Cached for offline use by Sentinel-Chimp; refresh via core.sectivia_import.refresh_sectivia_cache().",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {k: str(v) for k, v in paths.items()}


def _truthy_listed(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _map_sector(sector: str, segment: str = "") -> str:
    if segment in _SEGMENT_SECTOR_HINTS:
        return _SEGMENT_SECTOR_HINTS[segment]
    return _SECTOR_MAP.get(str(sector or "").strip().lower(), "Unknown")


def load_companies_csv(path: Path | str) -> Dict[str, dict]:
    """Load companies CSV keyed by uppercase ticker (skips blank tickers)."""
    out: Dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ticker = (row.get("ticker") or "").upper().strip()
            if not ticker:
                continue
            out[ticker] = row
    return out


def load_relations_csv(path: Path | str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def relations_from_csv(
    relations_path: Path | str,
    companies_path: Path | str | None = None,
    *,
    listed_only: bool = True,
    tickers: Optional[Iterable[str]] = None,
    add_customer_of: bool = True,
) -> Tuple[StockGraph, dict]:
    """Build a StockGraph from Sectivia relations (+ optional companies) CSV.

    Each supplier→customer row becomes ``SUPPLIER_TO`` (supplier sells to customer).
    When ``add_customer_of`` is True, also adds reverse ``CUSTOMER_OF`` so peer
    lists show the complementary perspective without relying only on incoming
    traversal.
    """
    allow: Optional[Set[str]] = {t.upper() for t in tickers} if tickers is not None else None
    companies = load_companies_csv(companies_path) if companies_path else {}
    relations = load_relations_csv(relations_path)

    g = StockGraph()

    def ensure_node(ticker: str, name: str = "", sector: str = "", segment: str = "") -> None:
        if g.get_node(ticker) is not None:
            return
        crow = companies.get(ticker, {})
        if listed_only and crow and not _truthy_listed(crow.get("listed")):
            return
        c_sector = crow.get("sector") or sector
        c_segment = crow.get("segment") or segment
        g.add_node(
            StockNode(
                ticker=ticker,
                name=str(crow.get("name") or name or ticker),
                sector=_map_sector(str(c_sector), str(c_segment)),
                sub_industry=str(c_segment or "Unknown"),
                market_cap_tier="Large",
                description="Sectivia supply-chain node",
            )
        )

    added = 0
    skipped = 0
    for row in relations:
        src = (row.get("supplier_ticker") or "").upper().strip()
        tgt = (row.get("customer_ticker") or "").upper().strip()
        if not src or not tgt:
            skipped += 1
            continue
        if allow is not None and (src not in allow or tgt not in allow):
            skipped += 1
            continue
        if listed_only:
            src_row = companies.get(src)
            tgt_row = companies.get(tgt)
            if src_row and not _truthy_listed(src_row.get("listed")):
                skipped += 1
                continue
            if tgt_row and not _truthy_listed(tgt_row.get("listed")):
                skipped += 1
                continue

        sector = row.get("sector") or ""
        s_seg = row.get("supplier_segment") or ""
        c_seg = row.get("customer_segment") or ""
        ensure_node(src, row.get("supplier_name") or "", sector, s_seg)
        ensure_node(tgt, row.get("customer_name") or "", sector, c_seg)
        if g.get_node(src) is None or g.get_node(tgt) is None:
            skipped += 1
            continue

        desc = f"Sectivia: {src} supplies {tgt}"
        if sector:
            desc += f" ({sector})"
        g.add_edge(
            GraphEdge(
                source=src,
                target=tgt,
                relation=RelationType.SUPPLIER_TO.value,
                weight=SECTIVIA_EDGE_WEIGHT,
                description=desc,
            )
        )
        added += 1
        if add_customer_of:
            g.add_edge(
                GraphEdge(
                    source=tgt,
                    target=src,
                    relation=RelationType.CUSTOMER_OF.value,
                    weight=SECTIVIA_EDGE_WEIGHT,
                    description=f"Sectivia: {tgt} customer of {src}",
                )
            )
            added += 1

    meta = {
        "source": "sectivia",
        "attribution": SECTIVIA_ATTRIBUTION,
        "license": SECTIVIA_LICENSE,
        "relations_path": str(relations_path),
        "companies_path": str(companies_path) if companies_path else None,
        "nodes": len(g.nodes),
        "edges": added,
        "skipped_relations": skipped,
        "listed_only": listed_only,
        "add_customer_of": add_customer_of,
        "edge_weight": SECTIVIA_EDGE_WEIGHT,
    }
    return g, meta


def load_cached_sectivia_graph(
    cache_dir: Path | str | None = None,
    *,
    listed_only: bool = True,
    add_customer_of: bool = True,
) -> Tuple[StockGraph, dict]:
    """Load graph from local cache; raises FileNotFoundError if relations CSV missing."""
    paths = cache_paths(cache_dir)
    if not paths["relations"].exists():
        raise FileNotFoundError(
            f"Sectivia cache missing: {paths['relations']}. "
            "Run refresh_sectivia_cache() or vendor CSVs under data/sectivia/."
        )
    companies = paths["companies"] if paths["companies"].exists() else None
    g, meta = relations_from_csv(
        paths["relations"],
        companies,
        listed_only=listed_only,
        add_customer_of=add_customer_of,
    )
    if paths["fetched_at"].exists():
        meta["fetched_at"] = paths["fetched_at"].read_text(encoding="utf-8").strip()
    meta["cache_dir"] = str(paths["dir"])
    return g, meta


def edge_key(edge: GraphEdge) -> Tuple[str, str, str]:
    return (edge.source.upper(), edge.target.upper(), edge.relation)


def merge_sectivia_into(
    base: StockGraph,
    overlay: StockGraph,
    *,
    prefer_curated: bool = True,
) -> dict:
    """Union-merge overlay into base.

    Conflict policy: if ``(source, target, relation)`` already exists in base,
    keep the curated/base edge (``prefer_curated=True``) and skip Sectivia.
    New nodes from overlay are added. Returns merge stats + attribution.
    """
    existing = {edge_key(e) for e in base.edges}
    nodes_added = 0
    edges_added = 0
    edges_skipped_dup = 0

    for node in overlay.nodes.values():
        if base.get_node(node.ticker) is None:
            base.add_node(node)
            nodes_added += 1

    for edge in overlay.edges:
        key = edge_key(edge)
        if key in existing:
            if prefer_curated:
                edges_skipped_dup += 1
                continue
        base.add_edge(edge)
        existing.add(key)
        edges_added += 1

    return {
        "attribution": SECTIVIA_ATTRIBUTION,
        "license": SECTIVIA_LICENSE,
        "nodes_added": nodes_added,
        "edges_added": edges_added,
        "edges_skipped_duplicate": edges_skipped_dup,
        "prefer_curated": prefer_curated,
        "base_nodes": len(base.nodes),
        "base_edges": len(base.edges),
        "conflict_policy": (
            "union; duplicate (source,target,relation) keeps curated/base edge"
            if prefer_curated
            else "union; duplicate keys still append overlay edge"
        ),
    }


def merge_cached_sectivia(
    base: StockGraph,
    cache_dir: Path | str | None = None,
    *,
    listed_only: bool = True,
    add_customer_of: bool = True,
    prefer_curated: bool = True,
    optional: bool = True,
) -> dict:
    """Merge cached Sectivia edges into ``base``. If cache missing and optional, no-op."""
    try:
        overlay, load_meta = load_cached_sectivia_graph(
            cache_dir, listed_only=listed_only, add_customer_of=add_customer_of
        )
    except FileNotFoundError:
        if optional:
            return {
                "attribution": SECTIVIA_ATTRIBUTION,
                "skipped": True,
                "reason": "cache_missing",
                "nodes_added": 0,
                "edges_added": 0,
            }
        raise
    stats = merge_sectivia_into(base, overlay, prefer_curated=prefer_curated)
    stats["load"] = load_meta
    stats["skipped"] = False
    return stats


# --- JSON helpers (optional; CSV is the primary path) ---

def load_sectivia_json(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "relations" not in data:
        raise ValueError("Not a Sectivia supply-chain JSON (missing 'relations').")
    return data


def sectivia_json_to_csv_rows(data: dict) -> List[dict]:
    """Normalize JSON relations to CSV-like row dicts for shared parsing tests."""
    rows = []
    for rel in data.get("relations") or []:
        if not isinstance(rel, dict):
            continue
        rows.append(
            {
                "supplier_ticker": rel.get("supplierTicker") or "",
                "customer_ticker": rel.get("customerTicker") or "",
                "supplier_name": "",
                "customer_name": "",
                "sector": rel.get("sector") or "",
                "supplier_segment": "",
                "customer_segment": "",
            }
        )
    return rows
