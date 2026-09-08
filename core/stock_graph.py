"""Relational Stock Graph for relative-value analysis, peer clustering, and lead-lag detection."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any

import numpy as np
import pandas as pd


class RelationType(str, Enum):
    SUPPLIER_TO = "SUPPLIER_TO"
    CUSTOMER_OF = "CUSTOMER_OF"
    COMPETITOR = "COMPETITOR"
    POWER_PARTNER = "POWER_PARTNER"
    INFRASTRUCTURE_PARTNER = "INFRASTRUCTURE_PARTNER"
    INVESTED_IN = "INVESTED_IN"
    CORRELATED_PEER = "CORRELATED_PEER"


@dataclass
class StockNode:
    ticker: str
    name: str
    sector: str
    sub_industry: str
    market_cap_tier: str = "Large"  # "Mega", "Large", "Mid", "Small"
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> StockNode:
        return cls(
            ticker=data["ticker"].upper(),
            name=data.get("name", data["ticker"]),
            sector=data.get("sector", "Unknown"),
            sub_industry=data.get("sub_industry", "Unknown"),
            market_cap_tier=data.get("market_cap_tier", "Large"),
            description=data.get("description", ""),
        )


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    description: str = ""
    bidirectional: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> GraphEdge:
        return cls(
            source=data["source"].upper(),
            target=data["target"].upper(),
            relation=data.get("relation", RelationType.CORRELATED_PEER.value),
            weight=float(data.get("weight", 1.0)),
            description=data.get("description", ""),
            bidirectional=bool(data.get("bidirectional", False)),
        )


@dataclass
class PeerDivergence:
    target_ticker: str
    peer_ticker: str
    relation: str
    target_return_pct: float
    peer_return_pct: float
    spread_pct: float  # peer - target return
    correlation: float
    z_score: float
    divergence_status: str  # "LAGGING_PEER", "LEADING_PEER", "IN_SYNC"
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


class StockGraph:
    """Directed, typed relational graph of stocks and inter-company dependencies."""

    def __init__(self) -> None:
        self._nodes: Dict[str, StockNode] = {}
        self._edges: List[GraphEdge] = []
        self._adj: Dict[str, List[GraphEdge]] = {}

    @property
    def nodes(self) -> Dict[str, StockNode]:
        return dict(self._nodes)

    @property
    def edges(self) -> List[GraphEdge]:
        return list(self._edges)

    def add_node(self, node: StockNode) -> None:
        sym = node.ticker.upper()
        self._nodes[sym] = node
        if sym not in self._adj:
            self._adj[sym] = []

    def get_node(self, ticker: str) -> Optional[StockNode]:
        return self._nodes.get(ticker.upper())

    def add_edge(self, edge: GraphEdge) -> None:
        src = edge.source.upper()
        tgt = edge.target.upper()
        if src not in self._nodes:
            self.add_node(StockNode(ticker=src, name=src, sector="Unknown", sub_industry="Unknown"))
        if tgt not in self._nodes:
            self.add_node(StockNode(ticker=tgt, name=tgt, sector="Unknown", sub_industry="Unknown"))

        norm_edge = GraphEdge(
            source=src,
            target=tgt,
            relation=edge.relation,
            weight=edge.weight,
            description=edge.description,
            bidirectional=edge.bidirectional,
        )
        self._edges.append(norm_edge)
        self._adj[src].append(norm_edge)

        if norm_edge.bidirectional:
            rev_edge = GraphEdge(
                source=tgt,
                target=src,
                relation=edge.relation,
                weight=edge.weight,
                description=edge.description,
                bidirectional=True,
            )
            self._adj[tgt].append(rev_edge)

    def get_neighbors(
        self,
        ticker: str,
        depth: int = 1,
        relation_types: Optional[Sequence[str]] = None,
    ) -> List[Tuple[StockNode, GraphEdge]]:
        """Find neighboring nodes up to a certain depth."""
        sym = ticker.upper()
        if sym not in self._nodes:
            return []

        rel_filter = set(relation_types) if relation_types else None
        visited: Set[str] = {sym}
        current_layer: Set[str] = {sym}
        results: List[Tuple[StockNode, GraphEdge]] = []

        for _ in range(max(1, depth)):
            next_layer: Set[str] = set()
            for current in current_layer:
                # Direct outgoing
                for edge in self._adj.get(current, []):
                    if rel_filter and edge.relation not in rel_filter:
                        continue
                    nbr = edge.target
                    if nbr not in visited and nbr in self._nodes:
                        visited.add(nbr)
                        next_layer.add(nbr)
                        results.append((self._nodes[nbr], edge))
                # Incoming connections
                for edge in self._edges:
                    if edge.target == current and not edge.bidirectional:
                        if rel_filter and edge.relation not in rel_filter:
                            continue
                        nbr = edge.source
                        if nbr not in visited and nbr in self._nodes:
                            visited.add(nbr)
                            next_layer.add(nbr)
                            results.append((self._nodes[nbr], edge))
            current_layer = next_layer
            if not current_layer:
                break

        return results

    def find_paths(
        self, source: str, target: str, max_depth: int = 3
    ) -> List[List[str]]:
        """Find paths connecting source to target up to max_depth."""
        src = source.upper()
        tgt = target.upper()
        if src not in self._nodes or tgt not in self._nodes:
            return []

        paths: List[List[str]] = []

        def dfs(current: str, path: List[str], depth: int):
            if current == tgt:
                paths.append(list(path))
                return
            if depth >= max_depth:
                return
            for edge in self._adj.get(current, []):
                nxt = edge.target
                if nxt not in path:
                    dfs(nxt, path + [nxt], depth + 1)

        dfs(src, [src], 0)
        return paths

    def subgraph(
        self, tickers: Sequence[str], depth: int = 0
    ) -> StockGraph:
        """Create a subgraph containing the specified tickers and optional depth neighbors."""
        sub = StockGraph()
        target_set: Set[str] = {t.upper() for t in tickers if t.upper() in self._nodes}

        if depth > 0:
            for t in list(target_set):
                nbrs = self.get_neighbors(t, depth=depth)
                for n_node, _ in nbrs:
                    target_set.add(n_node.ticker)

        for sym in target_set:
            sub.add_node(self._nodes[sym])

        for edge in self._edges:
            if edge.source in target_set and edge.target in target_set:
                sub.add_edge(edge)

        return sub

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> StockGraph:
        graph = cls()
        for nd in data.get("nodes", []):
            graph.add_node(StockNode.from_dict(nd))
        for ed in data.get("edges", []):
            graph.add_edge(GraphEdge.from_dict(ed))
        return graph

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> StockGraph:
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def analyze_divergence(
        self,
        data_provider: Any,
        ticker: str,
        period: str = "1mo",
        min_correlation: float = 0.40,
    ) -> List[PeerDivergence]:
        """Compute relative performance and divergence against peer basket."""
        sym = ticker.upper()
        if sym not in self._nodes:
            return []

        neighbors = self.get_neighbors(sym, depth=1)
        if not neighbors:
            return []

        # Fetch historical series for target
        try:
            target_stock = data_provider.create_ticker(sym)
            target_df = data_provider.fetch_history(target_stock, period=period, interval="1d")
        except Exception:
            return []

        if target_df.empty or "Close" not in target_df.columns or len(target_df) < 5:
            return []

        target_close = target_df["Close"].dropna()

        divergences: List[PeerDivergence] = []

        for peer_node, edge in neighbors:
            peer_sym = peer_node.ticker
            try:
                peer_stock = data_provider.create_ticker(peer_sym)
                peer_df = data_provider.fetch_history(peer_stock, period=period, interval="1d")
            except Exception:
                continue

            if peer_df.empty or "Close" not in peer_df.columns or len(peer_df) < 5:
                continue

            peer_close = peer_df["Close"].dropna()

            # Align on date index
            aligned = pd.concat([target_close, peer_close], axis=1, join="inner").dropna()
            if len(aligned) < 5:
                continue

            t_series = aligned.iloc[:, 0]
            p_series = aligned.iloc[:, 1]

            t_pct = (t_series.iloc[-1] / t_series.iloc[0]) - 1.0
            p_pct = (p_series.iloc[-1] / p_series.iloc[0]) - 1.0
            spread = p_pct - t_pct  # positive means peer outperforming target

            # Rolling return correlation
            t_daily = t_series.pct_change().dropna()
            p_daily = p_series.pct_change().dropna()
            corr = float(t_daily.corr(p_daily)) if len(t_daily) > 2 else 0.0
            if math.isnan(corr):
                corr = 0.0

            # Compute spread z-score from daily difference
            daily_diff = p_daily - t_daily
            std_diff = float(daily_diff.std()) if len(daily_diff) > 2 else 0.0
            z_score = float(spread / (std_diff * math.sqrt(len(daily_diff)))) if std_diff > 1e-6 else 0.0

            if abs(spread) < 0.02:
                status = "IN_SYNC"
                summary = f"{sym} and {peer_sym} tracking closely ({spread:+.1%})."
            elif spread > 0.02:
                status = "LAGGING_PEER"
                summary = f"{sym} lagging {peer_sym} by {spread:+.1%} (z={z_score:.1f}). Potential catch-up candidate."
            else:
                status = "LEADING_PEER"
                summary = f"{sym} outperforming {peer_sym} by {abs(spread):+.1%} (z={z_score:.1f})."

            divergences.append(
                PeerDivergence(
                    target_ticker=sym,
                    peer_ticker=peer_sym,
                    relation=edge.relation,
                    target_return_pct=float(t_pct),
                    peer_return_pct=float(p_pct),
                    spread_pct=float(spread),
                    correlation=float(corr),
                    z_score=float(z_score),
                    divergence_status=status,
                    summary=summary,
                )
            )

        # Sort by absolute spread descending
        divergences.sort(key=lambda d: abs(d.spread_pct), reverse=True)
        return divergences


def build_default_graph() -> StockGraph:
    """Build the curated default market relation graph across NASDAQ-100 and S&P 500 titans."""
    from core.graph_data import populate_market_universe

    g = StockGraph()
    populate_market_universe(g)
    return g

