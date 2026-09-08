"""Plotly 2D & 3D interactive network visualizer for Sentinel Stock Graph."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from core.stock_graph import StockGraph, RelationType


# Sentinel Institutional Dark Theme Palette
BG_COLOR = "#0b0f14"
SURFACE_COLOR = "#0e141c"
GRID_COLOR = "#243041"
TEXT_COLOR = "#e8eef7"
MUTED_TEXT = "#8b9bb4"

# Color mappings per sector
SECTOR_COLORS = {
    "Semiconductors": "#00d2be",         # Teal
    "Technology": "#7952ff",              # Indigo / Purple
    "Technology Hardware": "#3b82f6",     # Blue
    "Communication Services": "#ec4899",  # Pink
    "Consumer Discretionary": "#f97316",  # Orange
    "Consumer Staples": "#e11d48",        # Rose / Crimson
    "Healthcare": "#06b6d4",              # Cyan / Medical Blue
    "Financials": "#10b981",              # Emerald Green
    "Energy": "#84cc16",                  # Lime Green
    "Utilities": "#eab308",               # Yellow / Amber
    "Industrials": "#f59e0b",             # Amber / Industrial
    "Real Estate": "#a855f7",             # Purple
    "Materials": "#d97706",               # Bronze / Ochre
    "Index": "#94a3b8",                   # Slate
    "Unknown": "#64748b",
}

# Color mappings per relationship type
RELATION_COLORS = {
    RelationType.SUPPLIER_TO.value: "#00d2be",            # Cyan / Teal
    RelationType.CUSTOMER_OF.value: "#06b6d4",            # Light Cyan
    RelationType.COMPETITOR.value: "#f43f5e",             # Rose / Red
    RelationType.POWER_PARTNER.value: "#f59e0b",          # Amber
    RelationType.INFRASTRUCTURE_PARTNER.value: "#8b5cf6", # Violet
    RelationType.INVESTED_IN.value: "#10b981",            # Emerald
    RelationType.CORRELATED_PEER.value: "#3b82f6",        # Blue
}


def compute_spring_layout(
    tickers: List[str],
    edges: List[Tuple[str, str, float]],
    dim: int = 2,
    iterations: int = 60,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Force-directed layout (Fruchterman-Reingold) in pure NumPy."""
    n = len(tickers)
    if n == 0:
        return {}
    if n == 1:
        return {tickers[0]: np.zeros(dim)}

    rng = np.random.default_rng(seed)
    # Initial circular/spherical distribution
    if dim == 2:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = np.column_stack([np.cos(angles), np.sin(angles)]) + rng.normal(0, 0.1, (n, 2))
    else:
        pos = rng.normal(0, 1.0, (n, dim))
        pos /= np.linalg.norm(pos, axis=1, keepdims=True) + 1e-9

    idx_map = {t: i for i, t in enumerate(tickers)}
    k = math.sqrt(1.0 / n)  # optimal pairwise distance
    temp = 1.0  # cooling schedule initial temperature

    for it in range(iterations):
        disp = np.zeros_like(pos)

        # Repulsive forces (all pairs)
        for i in range(n):
            delta = pos[i] - pos
            dist = np.linalg.norm(delta, axis=1, keepdims=True)
            dist = np.maximum(dist, 1e-4)
            # F_r = (k^2 / d) * (delta / d)
            repulsion = (k * k / (dist * dist)) * delta
            repulsion[i] = 0.0
            disp[i] += np.sum(repulsion, axis=0)

        # Attractive forces (along edges)
        for src, tgt, weight in edges:
            if src in idx_map and tgt in idx_map:
                i, j = idx_map[src], idx_map[tgt]
                delta = pos[i] - pos[j]
                dist = np.linalg.norm(delta)
                if dist > 1e-4:
                    # F_a = (d^2 / k) * (delta / d) * weight
                    force = (dist / k) * delta * min(weight, 1.5)
                    disp[i] -= force
                    disp[j] += force

        # Displacement and cooling
        disp_len = np.linalg.norm(disp, axis=1, keepdims=True)
        disp_len = np.maximum(disp_len, 1e-4)
        pos += (disp / disp_len) * np.minimum(disp_len, temp)
        temp *= (1.0 - (it / iterations) * 0.5)

    # Center and normalize coordinates
    pos -= np.mean(pos, axis=0)
    max_range = np.max(np.abs(pos))
    if max_range > 0:
        pos = (pos / max_range) * 100.0

    return {t: pos[i] for t, i in idx_map.items()}


def build_network_plotly(
    graph: StockGraph,
    center_ticker: Optional[str] = None,
    depth: int = 1,
    dim: str = "2d",
) -> go.Figure:
    """Construct an interactive 2D or 3D network figure using Plotly."""
    is_3d = (dim.lower() == "3d")

    # Filter by center_ticker if specified
    active_graph = graph
    if center_ticker:
        sym = center_ticker.upper()
        if sym in graph.nodes:
            active_graph = graph.subgraph([sym], depth=depth)

    nodes = list(active_graph.nodes.values())
    tickers = [n.ticker for n in nodes]
    if not tickers:
        fig = go.Figure()
        fig.update_layout(
            title="Empty Stock Graph",
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=BG_COLOR,
            font=dict(color=TEXT_COLOR),
        )
        return fig

    # Compute node degrees
    degrees: Dict[str, int] = {t: 0 for t in tickers}
    edge_tuples: List[Tuple[str, str, float]] = []
    for e in active_graph.edges:
        if e.source in degrees and e.target in degrees:
            degrees[e.source] += 1
            degrees[e.target] += 1
            edge_tuples.append((e.source, e.target, e.weight))

    # Layout positions
    coords = compute_spring_layout(tickers, edge_tuples, dim=3 if is_3d else 2)

    fig = go.Figure()

    # Add edges grouped by relation type for clean legend filtering
    edges_by_relation: Dict[str, List[Tuple[np.ndarray, np.ndarray, str]]] = {}
    for e in active_graph.edges:
        if e.source in coords and e.target in coords:
            edges_by_relation.setdefault(e.relation, []).append(
                (coords[e.source], coords[e.target], f"{e.source} → {e.target}: {e.description}")
            )

    for rel, edge_list in edges_by_relation.items():
        color = RELATION_COLORS.get(rel, "#94a3b8")
        rel_label = rel.replace("_", " ").title()

        if not is_3d:
            edge_x, edge_y = [], []
            for p1, p2, _ in edge_list:
                edge_x.extend([p1[0], p2[0], None])
                edge_y.extend([p1[1], p2[1], None])

            fig.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(width=1.8, color=color),
                    name=rel_label,
                    hoverinfo="none",
                    legendgroup="edges",
                )
            )
        else:
            edge_x, edge_y, edge_z = [], [], []
            for p1, p2, _ in edge_list:
                edge_x.extend([p1[0], p2[0], None])
                edge_y.extend([p1[1], p2[1], None])
                edge_z.extend([p1[2], p2[2], None])

            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    line=dict(width=3, color=color),
                    name=rel_label,
                    hoverinfo="none",
                    legendgroup="edges",
                )
            )

    # Group nodes by sector
    sectors: Set[str] = {n.sector for n in nodes}
    for sector in sorted(sectors):
        sec_nodes = [n for n in nodes if n.sector == sector]
        sec_color = SECTOR_COLORS.get(sector, "#94a3b8")

        x_vals = [coords[n.ticker][0] for n in sec_nodes]
        y_vals = [coords[n.ticker][1] for n in sec_nodes]
        sizes = [
            28 if n.market_cap_tier == "Mega" else 22 if n.market_cap_tier == "Large" else 16
            for n in sec_nodes
        ]
        labels = [n.ticker for n in sec_nodes]
        hover_texts = [
            f"<b>{n.ticker}</b> — {n.name}<br>"
            f"Sector: {n.sector} ({n.sub_industry})<br>"
            f"Tier: {n.market_cap_tier} Cap<br>"
            f"Connections: {degrees[n.ticker]}<br>"
            f"<i>{n.description}</i>"
            for n in sec_nodes
        ]

        if not is_3d:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    textfont=dict(color=TEXT_COLOR, size=11, family="Inter, -apple-system, sans-serif"),
                    marker=dict(
                        size=sizes,
                        color=sec_color,
                        line=dict(width=1.5, color="#ffffff"),
                        opacity=0.92,
                    ),
                    name=sector,
                    hovertext=hover_texts,
                    hoverinfo="text",
                    legendgroup="sectors",
                )
            )
        else:
            z_vals = [coords[n.ticker][2] for n in sec_nodes]
            fig.add_trace(
                go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    textfont=dict(color=TEXT_COLOR, size=11),
                    marker=dict(
                        size=[s * 0.7 for s in sizes],
                        color=sec_color,
                        line=dict(width=1, color="#ffffff"),
                        opacity=0.92,
                    ),
                    name=sector,
                    hovertext=hover_texts,
                    hoverinfo="text",
                    legendgroup="sectors",
                )
            )

    # Layout styling with Sentinel dark theme
    title_text = f"Sentinel Stock Relationship Graph — {center_ticker.upper() if center_ticker else 'Market Tech & Infra'}"
    if not is_3d:
        fig.update_layout(
            title=dict(text=title_text, font=dict(size=16, color=TEXT_COLOR)),
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=SURFACE_COLOR,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=True,
            legend=dict(
                font=dict(size=11, color=MUTED_TEXT),
                bgcolor="rgba(14, 20, 28, 0.8)",
                bordercolor=GRID_COLOR,
                borderwidth=1,
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            hoverlabel=dict(bgcolor=SURFACE_COLOR, font=dict(color=TEXT_COLOR, size=12)),
        )
    else:
        fig.update_layout(
            title=dict(text=title_text, font=dict(size=16, color=TEXT_COLOR)),
            paper_bgcolor=BG_COLOR,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, backgroundcolor=SURFACE_COLOR),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, backgroundcolor=SURFACE_COLOR),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, backgroundcolor=SURFACE_COLOR),
                bgcolor=BG_COLOR,
            ),
            showlegend=True,
            legend=dict(
                font=dict(size=11, color=MUTED_TEXT),
                bgcolor="rgba(14, 20, 28, 0.8)",
                bordercolor=GRID_COLOR,
                borderwidth=1,
            ),
            margin=dict(l=20, r=20, t=50, b=20),
        )

    return fig


def export_graph_html(fig: go.Figure, path: Path | str) -> str:
    """Save the Plotly figure as a standalone interactive HTML document."""
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(out_path),
        include_plotlyjs="cdn",
        full_html=True,
        config=dict(responsive=True, displayModeBar=True),
    )
    return str(out_path)
