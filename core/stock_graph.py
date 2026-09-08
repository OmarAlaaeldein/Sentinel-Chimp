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
    """Build the curated default market relation graph across tech, semis, energy & infra."""
    g = StockGraph()

    # --- Nodes ---
    nodes = [
        # AI & GPU Designers
        StockNode("NVDA", "NVIDIA Corp", "Semiconductors", "Compute & Networking", "Mega", "AI accelerator & GPU market leader"),
        StockNode("AMD", "Advanced Micro Devices", "Semiconductors", "Compute & Graphics", "Large", "Direct x86 and GPU rival to Intel/Nvidia"),
        StockNode("INTC", "Intel Corp", "Semiconductors", "Processors & Foundries", "Large", "x86 CPU vendor and emerging foundry"),
        StockNode("AVGO", "Broadcom Inc", "Semiconductors", "Custom Silicon & Networking", "Mega", "Custom AI ASICs & datacenter networking"),
        StockNode("ARM", "Arm Holdings plc", "Semiconductors", "Architecture IP", "Large", "CPU architecture IP licensor for mobile & server"),
        StockNode("QCOM", "Qualcomm Inc", "Semiconductors", "Wireless & Edge AI", "Large", "Mobile processors & edge AI platforms"),
        
        # Semiconductor Manufacturing & Equipment
        StockNode("TSM", "Taiwan Semiconductor", "Semiconductors", "Pure-Play Foundry", "Mega", "Leading foundry manufacturing for Nvidia/Apple/AMD"),
        StockNode("ASML", "ASML Holding", "Semiconductors", "Lithography Systems", "Mega", "Sole supplier of EUV lithography machines"),
        StockNode("MU", "Micron Technology", "Semiconductors", "Memory & Storage", "Large", "HBM memory supplier for high-end AI accelerators"),

        # AI Servers & Systems
        StockNode("SMCI", "Super Micro Computer", "Technology Hardware", "AI Server Systems", "Large", "Liquid-cooled server racks for Nvidia GPUs"),
        StockNode("DELL", "Dell Technologies", "Technology Hardware", "Enterprise Hardware", "Large", "Enterprise AI server and storage builder"),

        # Hyperscalers & Platforms
        StockNode("MSFT", "Microsoft Corp", "Technology", "Cloud & Software", "Mega", "Azure cloud hyperscaler and major OpenAI partner"),
        StockNode("GOOGL", "Alphabet Inc", "Communication Services", "Search & Cloud", "Mega", "Google Cloud, Gemini AI, custom TPU accelerators"),
        StockNode("AMZN", "Amazon.com Inc", "Consumer Discretionary", "E-Commerce & AWS", "Mega", "AWS cloud leader and custom Trainium/Inferentia"),
        StockNode("META", "Meta Platforms", "Communication Services", "Social & Open AI", "Mega", "Llama AI developer and massive GPU cluster buyer"),
        StockNode("AAPL", "Apple Inc", "Technology", "Consumer Electronics", "Mega", "Consumer device ecosystem & Apple Intelligence"),

        # Enterprise AI & Data
        StockNode("PLTR", "Palantir Technologies", "Technology", "Enterprise AI Platforms", "Large", "AIP decision-making & defense data infrastructure"),
        StockNode("SNOW", "Snowflake Inc", "Technology", "Cloud Data Platform", "Large", "Data warehouse and analytics platform"),

        # AI Power, Utilities & Nuclear
        StockNode("VST", "Vistra Corp", "Utilities", "Independent Power Producer", "Large", "Nuclear and gas power for hyperscaler datacenters"),
        StockNode("CEG", "Constellation Energy", "Utilities", "Clean Energy & Nuclear", "Large", "Nuclear generation provider powering datacenters"),
        StockNode("CCJ", "Cameco Corp", "Energy", "Uranium Mining", "Large", "Fuel provider for nuclear power generation"),

        # Crypto & Treasury Proxies
        StockNode("COIN", "Coinbase Global", "Financials", "Crypto Exchange", "Large", "Major crypto trading and custody platform"),
        StockNode("MSTR", "MicroStrategy Inc", "Technology", "Bitcoin Treasury", "Large", "Bitcoin holding company and analytics software"),

        # Index Benchmarks
        StockNode("SPY", "SPDR S&P 500 ETF", "Index", "Broad Market", "Mega", "US large-cap equity market benchmark"),
        StockNode("QQQ", "Invesco QQQ Trust", "Index", "Large-Cap Tech", "Mega", "Nasdaq 100 technology-heavy index benchmark"),
    ]

    for n in nodes:
        g.add_node(n)

    # --- Edges ---
    edges = [
        # Semiconductor Foundry & Manufacturing
        GraphEdge("TSM", "NVDA", RelationType.SUPPLIER_TO.value, 1.0, "Exclusive advanced node wafer manufacturing for Blackwell & Hopper"),
        GraphEdge("TSM", "AMD", RelationType.SUPPLIER_TO.value, 0.9, "Primary foundry for EPYC and Instinct GPUs"),
        GraphEdge("TSM", "AAPL", RelationType.SUPPLIER_TO.value, 1.0, "Sole manufacturer of Apple Silicon M-series & A-series chips"),
        GraphEdge("ASML", "TSM", RelationType.SUPPLIER_TO.value, 1.0, "Sole supplier of High-NA and EUV lithography equipment"),
        GraphEdge("ASML", "INTC", RelationType.SUPPLIER_TO.value, 0.8, "EUV scanner supplier for Intel 18A process"),

        # Memory & Packaging
        GraphEdge("MU", "NVDA", RelationType.SUPPLIER_TO.value, 0.9, "Key supplier of HBM3e memory for Nvidia AI GPUs"),
        GraphEdge("MU", "AMD", RelationType.SUPPLIER_TO.value, 0.8, "Memory supplier for MI300 series accelerators"),

        # Direct Competitors
        GraphEdge("NVDA", "AMD", RelationType.COMPETITOR.value, 0.9, "Direct competition in datacenter GPUs and gaming graphics", bidirectional=True),
        GraphEdge("INTC", "AMD", RelationType.COMPETITOR.value, 0.9, "Direct competition in x86 desktop and server CPUs", bidirectional=True),
        GraphEdge("AVGO", "NVDA", RelationType.COMPETITOR.value, 0.7, "Competition between custom hyperscaler ASICs and general GPUs", bidirectional=True),
        GraphEdge("QCOM", "ARM", RelationType.COMPETITOR.value, 0.6, "Oryon architecture licensing dispute and PC CPU competition", bidirectional=True),

        # Datacenter Hardware OEM / Integration
        GraphEdge("NVDA", "SMCI", RelationType.SUPPLIER_TO.value, 0.9, "Supplies Blackwell/Hopper GPUs for liquid-cooled rack clusters"),
        GraphEdge("NVDA", "DELL", RelationType.SUPPLIER_TO.value, 0.9, "Supplies AI accelerator cards for Dell PowerEdge servers"),

        # Hyperscaler AI Chip Buyers & Cloud Partnerships
        GraphEdge("NVDA", "MSFT", RelationType.INFRASTRUCTURE_PARTNER.value, 1.0, "Massive deployment of DGX/GPU clusters on Azure cloud"),
        GraphEdge("NVDA", "GOOGL", RelationType.INFRASTRUCTURE_PARTNER.value, 0.8, "Supplies GPUs alongside Google's internal TPU deployment"),
        GraphEdge("NVDA", "AMZN", RelationType.INFRASTRUCTURE_PARTNER.value, 0.85, "Major supplier for AWS EC2 GPU instances"),
        GraphEdge("NVDA", "META", RelationType.INFRASTRUCTURE_PARTNER.value, 0.9, "Meta is one of Nvidia's largest cluster customers"),

        # Hyperscaler Rivalry
        GraphEdge("MSFT", "GOOGL", RelationType.COMPETITOR.value, 0.9, "Rivalry across cloud, search, and foundational models", bidirectional=True),
        GraphEdge("MSFT", "AMZN", RelationType.COMPETITOR.value, 0.9, "Direct rivalry between Azure and AWS cloud infrastructure", bidirectional=True),

        # Enterprise AI Software
        GraphEdge("PLTR", "MSFT", RelationType.INFRASTRUCTURE_PARTNER.value, 0.85, "Strategic partnership integrating AIP with Azure OpenAI"),
        GraphEdge("SNOW", "AMZN", RelationType.INFRASTRUCTURE_PARTNER.value, 0.8, "Primary cloud data warehousing partner on AWS"),

        # Datacenter Power & Clean Energy
        GraphEdge("VST", "MSFT", RelationType.POWER_PARTNER.value, 0.85, "Datacenter power supply and merchant clean energy agreements"),
        GraphEdge("CEG", "MSFT", RelationType.POWER_PARTNER.value, 0.9, "20-year power purchase agreements for nuclear energy generation"),
        GraphEdge("CCJ", "CEG", RelationType.SUPPLIER_TO.value, 0.8, "Uranium fuel supply for Constellation nuclear reactors"),
        GraphEdge("CCJ", "VST", RelationType.SUPPLIER_TO.value, 0.8, "Nuclear fuel supply for Comanche Peak nuclear generation"),

        # Crypto Ecosystem
        GraphEdge("COIN", "MSTR", RelationType.CORRELATED_PEER.value, 0.85, "High statistical beta to Bitcoin price action and institutional inflows", bidirectional=True),

        # Broad Market Anchors
        GraphEdge("SPY", "AAPL", RelationType.CORRELATED_PEER.value, 0.7, "Top index weighting in S&P 500"),
        GraphEdge("SPY", "MSFT", RelationType.CORRELATED_PEER.value, 0.7, "Top index weighting in S&P 500"),
        GraphEdge("SPY", "NVDA", RelationType.CORRELATED_PEER.value, 0.7, "Top momentum driver in S&P 500"),
        GraphEdge("QQQ", "NVDA", RelationType.CORRELATED_PEER.value, 0.85, "Dominant tech leadership in Nasdaq 100"),
    ]

    for e in edges:
        g.add_edge(e)

    return g
