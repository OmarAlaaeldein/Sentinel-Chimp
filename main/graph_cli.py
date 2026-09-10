"""CLI commands for Sentinel Stock Relationship Graph and Peer Divergence Analysis."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from core.graph_service import (
    categorize_peers,
    export_graph_html_file,
    load_graph,
    show_connections,
)
from core.stock_graph import StockGraph


def add_commands(sub: argparse._SubParsersAction) -> None:
    graph_parser = sub.add_parser(
        "graph",
        help="Stock relationship graph, peer clustering, and lead-lag divergence",
    )
    commands = graph_parser.add_subparsers(dest="graph_command", required=True)

    # 1. show
    show = commands.add_parser("show", help="Inspect network nodes and connections")
    show.add_argument("ticker", nargs="?", help="Center ticker to inspect (omit for entire network summary)")
    show.add_argument("--depth", type=int, default=1, help="Neighbor traversal depth (default: 1)")
    common_flags(show)

    # 2. peers
    peers = commands.add_parser("peers", help="List direct suppliers, customers, competitors, and partners")
    peers.add_argument("ticker", help="Target ticker (e.g. NVDA, AMD, MSFT)")
    common_flags(peers)

    # 3. divergence
    div = commands.add_parser("divergence", help="Analyze relative return spreads and peer lead-lag divergence")
    div.add_argument("ticker", help="Target ticker to evaluate against peer basket")
    div.add_argument("--period", choices=["1mo", "3mo", "6mo", "1y"], default="1mo", help="Historical return period")
    common_flags(div)

    # 4. export
    exp = commands.add_parser("export", help="Export interactive Plotly network graph to standalone HTML")
    exp.add_argument("ticker", nargs="?", help="Center ticker for subgraph (omit for full market network)")
    exp.add_argument("--html", required=True, metavar="PATH", help="Destination HTML file path")
    exp.add_argument("--depth", type=int, default=1, help="Neighbor traversal depth for center ticker")
    exp.add_argument("--dim", choices=["2d", "3d"], default="2d", help="Visualization dimensionality (2d or 3d)")
    common_flags(exp)


def common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--graph-file", metavar="PATH", help="Optional path to custom JSON stock graph")


def get_graph(graph_file: Optional[str] = None) -> StockGraph:
    return load_graph(graph_file)


def run(args: argparse.Namespace) -> dict:
    cmd = args.graph_command
    graph = get_graph(getattr(args, "graph_file", None))

    if cmd == "show":
        return show_connections(graph, ticker=args.ticker, depth=args.depth)

    if cmd == "peers":
        return categorize_peers(graph, args.ticker)

    if cmd == "divergence":
        from core.data import YFinanceProvider
        sym = args.ticker.upper()
        if sym not in graph.nodes:
            raise ValueError(f"Ticker {sym!r} not found in stock graph.")
        provider = YFinanceProvider()
        divs = graph.analyze_divergence(provider, sym, period=args.period)
        return {
            "action": "divergence",
            "ticker": sym,
            "period": args.period,
            "peer_count": len(divs),
            "divergences": [d.to_dict() for d in divs],
        }

    if cmd == "export":
        sym = args.ticker.upper() if args.ticker else None
        saved_path = export_graph_html_file(
            graph, args.html, center_ticker=sym, depth=args.depth, dim=args.dim
        )
        return {
            "action": "export",
            "ticker": sym,
            "path": saved_path,
            "dimension": args.dim,
            "depth": args.depth,
        }

    raise ValueError(f"Unknown graph command: {cmd}")


def render(data: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"schema_version": 1, "status": "ok", "data": data}, indent=2))
        return

    action = data.get("action")

    if action == "summary":
        print(f"Sentinel Stock Relationship Graph ({data['total_nodes']} nodes, {data['total_edges']} edges)")
        print(f"Sectors: {', '.join(data['sectors'])}")
        print(f"Tracked Tickers: {', '.join(data['tickers'])}")
        print("\nTip: Run 'sentinel graph show TICKER' or 'sentinel graph peers TICKER' for details.")

    elif action == "show":
        node = data["node"]
        print(f"=== {node['ticker']} — {node['name']} ({node['sector']} / {node['sub_industry']}) ===")
        print(f"Tier: {node['market_cap_tier']} Cap | Connections (depth {data['depth']}): {data['connection_count']}")
        print(f"Overview: {node['description']}\n")
        print("Connected Companies:")
        for c in data["connections"]:
            rel = c['relation'].replace('_', ' ')
            print(f"  [{rel}] {c['neighbor']} ({c['name']})")
            if c['description']:
                print(f"      Details: {c['description']}")

    elif action == "peers":
        node = data["node"]
        print(f"=== Peer Network for {node['ticker']} ({node['name']}) ===")
        for rel, peer_list in data["peer_categories"].items():
            print(f"\n{rel.replace('_', ' ').upper()}:")
            for p in peer_list:
                print(f"  * {p['ticker']} ({p['name']}) — {p['sector']}")
                if p['description']:
                    print(f"    {p['description']}")

    elif action == "divergence":
        print(f"=== Peer Lead-Lag & Relative Divergence for {data['ticker']} ({data['period']}) ===")
        if not data["divergences"]:
            print("No active divergence or historical price data available for connected peers.")
            return

        print(f"{'Peer':<6} {'Relation':<18} {'Target %':<10} {'Peer %':<10} {'Spread':<10} {'Status':<15}")
        print("-" * 75)
        for d in data["divergences"]:
            rel_short = d['relation'][:17]
            spread_str = f"{d['spread_pct']:+.1%}"
            print(f"{d['peer_ticker']:<6} {rel_short:<18} {d['target_return_pct']:+8.1%} {d['peer_return_pct']:+8.1%} {spread_str:<10} {d['divergence_status']:<15}")
        print("\nObservations:")
        for d in data["divergences"]:
            if d['divergence_status'] != "IN_SYNC":
                print(f"  • {d['summary']}")

    elif action == "export":
        print(f"Wrote interactive {data['dimension'].upper()} graph to: {data['path']}", file=sys.stderr)
        print(f"Open in any browser: file://{data['path']}")
