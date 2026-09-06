"""Options landscape 3D visualization helpers (Plotly HTML + matplotlib chrome).

Keeps Lite Mode deps: numpy / matplotlib / plotly only. No scipy/pyvista/torch.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

# Category keys used by the in-app filter checkboxes / Plotly hover
CAT_EARN_UNDER = "earnings_under"
CAT_EARN_OVER = "earnings_over"
CAT_UNDER = "under"
CAT_OVER = "over"

# Distinct marker accents for earnings (EV colorbar still drives regular points)
EARN_UNDER_RGB = (0, 230, 230)   # cyan
EARN_OVER_RGB = (200, 80, 255)   # magenta


def categorize_row(is_earnings: bool, is_undervalued: bool) -> str:
    if is_earnings:
        return CAT_EARN_UNDER if is_undervalued else CAT_EARN_OVER
    return CAT_UNDER if is_undervalued else CAT_OVER


def bin_ev_grid(
    days: Sequence[float],
    strikes: Sequence[float],
    evs: Sequence[float],
    *,
    max_bins: int = 24,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build a coarse (days × strike) mean-EV grid for contour/surface underlay.

    Returns ``(day_centers, strike_centers, Z)`` or None if too sparse.
    """
    if len(days) < 6:
        return None
    d = np.asarray(days, dtype=float)
    s = np.asarray(strikes, dtype=float)
    z = np.asarray(evs, dtype=float)
    mask = np.isfinite(d) & np.isfinite(s) & np.isfinite(z)
    d, s, z = d[mask], s[mask], z[mask]
    if d.size < 6:
        return None

    n_d = int(min(max_bins, max(3, len(np.unique(np.round(d, 0))))))
    n_s = int(min(max_bins, max(3, len(np.unique(np.round(s, 2))))))
    d_edges = np.linspace(d.min(), d.max(), n_d + 1)
    s_edges = np.linspace(s.min(), s.max(), n_s + 1)
    # Avoid zero-width bins when all values equal
    if d_edges[-1] <= d_edges[0]:
        d_edges[-1] = d_edges[0] + 1.0
    if s_edges[-1] <= s_edges[0]:
        s_edges[-1] = s_edges[0] + 1.0

    Z = np.full((n_s, n_d), np.nan, dtype=float)
    counts = np.zeros_like(Z)
    di = np.clip(np.digitize(d, d_edges) - 1, 0, n_d - 1)
    si = np.clip(np.digitize(s, s_edges) - 1, 0, n_s - 1)
    for i in range(d.size):
        Z[si[i], di[i]] = np.nan_to_num(Z[si[i], di[i]], nan=0.0) + z[i]
        counts[si[i], di[i]] += 1.0
    with np.errstate(invalid="ignore"):
        Z = np.where(counts > 0, Z / counts, np.nan)
    if np.isfinite(Z).sum() < 4:
        return None
    d_c = 0.5 * (d_edges[:-1] + d_edges[1:])
    s_c = 0.5 * (s_edges[:-1] + s_edges[1:])
    return d_c, s_c, Z


def build_plotly_figure(
    ticker: str,
    option_type: str,
    days: Sequence[float],
    strikes: Sequence[float],
    evs: Sequence[float],
    date_labels: Sequence[str],
    volumes: Sequence[float],
    categories: Sequence[str],
    *,
    include_heatmap: bool = True,
):
    """Return a Plotly Figure: 3D scatter (+ optional EV heatmap underlay).

    Color encodes **EV@Ask ($)** = Fair − Ask (buy-side tradeable edge).
    Earnings contracts keep cyan/magenta marker outlines for quick spotting.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    days_a = np.asarray(days, dtype=float)
    strikes_a = np.asarray(strikes, dtype=float)
    evs_a = np.asarray(evs, dtype=float)
    vols_a = np.asarray(volumes, dtype=float)
    cats = list(categories)
    labels = list(date_labels)

    cat_pretty = {
        CAT_EARN_UNDER: "Earnings Under (buy-side edge near earnings)",
        CAT_EARN_OVER: "Earnings Over (sell-side edge near earnings)",
        CAT_UNDER: "Under — Fair beats Ask (green row)",
        CAT_OVER: "Over — Bid beats Fair (red row)",
    }
    hover = []
    for i in range(len(days_a)):
        cat = cats[i] if i < len(cats) else CAT_UNDER
        hover.append(
            f"<b>{cat_pretty.get(cat, cat)}</b><br>"
            f"Expiry: {labels[i] if i < len(labels) else '?'}<br>"
            f"Days to expiry: {days_a[i]:.0f}<br>"
            f"Strike: ${strikes_a[i]:.2f}<br>"
            f"Volume: {int(vols_a[i]) if np.isfinite(vols_a[i]) else 0}<br>"
            f"<b>EV@Ask: ${evs_a[i]:+.2f}</b><br>"
            f"<span style='color:#8b9bb4'>EV@Ask = Fair $ − Ask (buy-side edge)</span>"
        )

    # Marker size: soft volume scaling, clamped for readability
    if vols_a.size and np.nanmax(vols_a) > 0:
        sizes = 5.0 + 10.0 * np.sqrt(np.clip(vols_a, 0, None) / np.nanmax(vols_a))
    else:
        sizes = np.full(len(days_a), 7.0)

    # Line colors: earnings accent, else match EV (transparent outline)
    line_colors = []
    for cat in cats:
        if cat == CAT_EARN_UNDER:
            line_colors.append(f"rgb{EARN_UNDER_RGB}")
        elif cat == CAT_EARN_OVER:
            line_colors.append(f"rgb{EARN_OVER_RGB}")
        else:
            line_colors.append("rgba(20,28,40,0.85)")

    scatter = go.Scatter3d(
        x=days_a,
        y=strikes_a,
        z=evs_a,
        mode="markers",
        name="Contracts",
        marker=dict(
            size=sizes,
            color=evs_a,
            colorscale="RdYlGn",
            cmid=0.0,
            opacity=0.92,
            line=dict(width=2, color=line_colors),
            colorbar=dict(
                title=dict(text="EV@Ask ($)", font=dict(size=13, color="#e8eef7")),
                tickfont=dict(size=11, color="#8b9bb4"),
                x=1.02,
                len=0.72,
            ),
            showscale=True,
        ),
        text=hover,
        hoverinfo="text",
    )

    grid = bin_ev_grid(days_a, strikes_a, evs_a) if include_heatmap else None
    use_heat = grid is not None
    rows = 2 if use_heat else 1
    specs = [[{"type": "scene"}], [{"type": "xy"}]] if use_heat else [[{"type": "scene"}]]
    titles = (
        [
            f"{ticker} {option_type} — 3D landscape (color / Z = EV@Ask $)",
            "Heatmap — mean EV@Ask ($) by days × strike (clearer when scatter is dense)",
        ]
        if use_heat
        else [f"{ticker} {option_type} — Strike × Expiry × EV@Ask ($)"]
    )

    fig = make_subplots(
        rows=rows,
        cols=1,
        specs=specs,
        subplot_titles=titles,
        row_heights=[0.70, 0.30] if use_heat else [1.0],
        vertical_spacing=0.10,
    )
    fig.add_trace(scatter, row=1, col=1)

    if use_heat:
        d_c, s_c, Z = grid
        fig.add_trace(
            go.Heatmap(
                x=d_c,
                y=s_c,
                z=Z,
                colorscale="RdYlGn",
                zmid=0.0,
                colorbar=dict(
                    title=dict(text="Mean EV@Ask ($)", font=dict(size=12, color="#e8eef7")),
                    tickfont=dict(size=10, color="#8b9bb4"),
                    x=1.12,
                    len=0.28,
                    y=0.14,
                ),
                hovertemplate=(
                    "Days %{x:.0f}<br>Strike $%{y:.2f}<br>"
                    "Mean EV@Ask $%{z:+.2f}<extra></extra>"
                ),
                name="EV heatmap",
            ),
            row=2,
            col=1,
        )

    axis_font = dict(size=12, color="#e8eef7")
    tick_font = dict(size=11, color="#8b9bb4")
    scene = dict(
        xaxis=dict(
            title=dict(text="Days to expiry", font=axis_font),
            backgroundcolor="#0e141c",
            gridcolor="#243041",
            zerolinecolor="#3dd6c6",
            color="#e8eef7",
            tickfont=tick_font,
        ),
        yaxis=dict(
            title=dict(text="Strike ($)", font=axis_font),
            backgroundcolor="#0e141c",
            gridcolor="#243041",
            zerolinecolor="#3dd6c6",
            color="#e8eef7",
            tickfont=tick_font,
        ),
        zaxis=dict(
            title=dict(text="EV@Ask ($)", font=axis_font),
            backgroundcolor="#0e141c",
            gridcolor="#243041",
            zerolinecolor="#f0a05a",
            color="#e8eef7",
            tickfont=tick_font,
        ),
        bgcolor="#0b0f14",
        camera=dict(eye=dict(x=1.55, y=1.55, z=1.15), up=dict(x=0, y=0, z=1)),
        aspectmode="manual",
        aspectratio=dict(x=1.15, y=1.0, z=0.75),
    )

    fig.update_layout(
        title=dict(
            text=(
                f"{ticker} {option_type} landscape — color = EV@Ask ($) "
                f"(Fair − Ask). Cyan/magenta outline = earnings window."
            ),
            font=dict(size=14, color="#e8eef7"),
        ),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        font=dict(color="#e8eef7", size=12),
        margin=dict(l=40, r=90, t=70, b=50),
        height=900 if use_heat else 720,
        legend=dict(font=dict(size=11)),
        annotations=[
            dict(
                text=(
                    "Green EV@Ask → Fair above Ask (buy-side / Under). "
                    "Red → Fair below Ask (no buy edge; Over if bid also beats Fair). "
                    "Z=0 is break-even vs the ask."
                ),
                xref="paper",
                yref="paper",
                x=0.0,
                y=-0.02,
                showarrow=False,
                font=dict(size=11, color="#8b9bb4"),
                align="left",
            )
        ],
    )
    fig.update_scenes(scene)
    if use_heat:
        fig.update_xaxes(
            title_text="Days to expiry",
            color="#e8eef7",
            tickfont=tick_font,
            gridcolor="#243041",
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="Strike ($)",
            color="#e8eef7",
            tickfont=tick_font,
            gridcolor="#243041",
            row=2,
            col=1,
        )

    return fig


def style_mpl_3d_axes(ax, ticker: str, option_type: str) -> None:
    """Readable matplotlib 3D chrome matching the dark terminal."""
    ax.set_xlabel("Days to expiry", color="#e8eef7", fontsize=11, labelpad=8)
    ax.set_ylabel("Strike ($)", color="#e8eef7", fontsize=11, labelpad=8)
    ax.set_zlabel("EV@Ask ($)", color="#e8eef7", fontsize=11, labelpad=8)
    ax.set_title(
        f"{ticker} {option_type} landscape — Z/color = EV@Ask ($)",
        color="#e8eef7",
        fontsize=12,
        pad=12,
    )
    ax.tick_params(colors="#8b9bb4", labelsize=10)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, color="#243041", linestyle="--", linewidth=0.6, alpha=0.85)
    try:
        ax.view_init(elev=22, azim=-58)
    except Exception:
        pass


def filter_rows_for_plot(
    base_data: List[dict],
    *,
    show_earn_under: bool,
    show_earn_over: bool,
    show_under: bool,
    show_over: bool,
) -> List[dict]:
    """Apply checkbox filters; attach ``category`` for Plotly/matplotlib."""
    out = []
    for row in base_data:
        is_earn = bool(row.get("is_earnings"))
        is_good = bool(row.get("is_good"))  # undervalued / Under
        if is_earn:
            if is_good and not show_earn_under:
                continue
            if (not is_good) and not show_earn_over:
                continue
        else:
            if is_good and not show_under:
                continue
            if (not is_good) and not show_over:
                continue
        item = dict(row)
        item["category"] = categorize_row(is_earn, is_good)
        out.append(item)
    return out
