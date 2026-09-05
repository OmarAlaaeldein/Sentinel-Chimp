"""Main price chart rendering (extracted from MarketApp.update_chart)."""
from __future__ import annotations

from datetime import timedelta
from typing import Optional, Tuple

import numpy as np


def prepare_plot_frame(df, interval: Optional[str]):
    """Copy ``df`` and optionally filter to RTH for intraday intervals.

    Returns ``(plot_df, times_for_labels_eastern, x_vals)``.
    """
    plot_df = df.copy()
    intraday = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
    if interval in intraday:
        idx = plot_df.index
        if getattr(idx, "tz", None):
            idx_eastern = idx.tz_convert("America/New_York")
        else:
            idx_eastern = idx.tz_localize("UTC").tz_convert("America/New_York")
        minutes = idx_eastern.hour * 60 + idx_eastern.minute
        mask = (minutes >= 570) & (minutes <= 960) & (idx_eastern.dayofweek < 5)
        filtered = plot_df[mask]
        if not filtered.empty:
            plot_df = filtered

    times_for_labels = plot_df.index
    if getattr(times_for_labels, "tz", None):
        times_for_labels = times_for_labels.tz_convert("America/New_York")
    else:
        times_for_labels = times_for_labels.tz_localize("UTC").tz_convert("America/New_York")

    x_vals = np.arange(len(plot_df))
    return plot_df, times_for_labels, x_vals


def draw_probability_cone(
    ax,
    last_x: float,
    p0: float,
    sigma: float,
    horizon_days: int = 30,
    face_color: str = "#00e6ff",
    alpha: float = 0.18,
    edge_color: str = "#00e6ff",
    edge_alpha: float = 0.55,
) -> Optional[Tuple[float, float, float]]:
    """Draw a translucent vol cone extending ``horizon_days`` right of last_x.

    Returns ``(x_right, y_lo, y_hi)`` useful for axis padding, or None if
    inputs are not drawable.
    """
    from core.vol_models import probability_cone

    if p0 is None or sigma is None:
        return None
    try:
        p0 = float(p0)
        sigma = float(sigma)
    except (TypeError, ValueError):
        return None
    if p0 <= 0 or sigma < 0 or not np.isfinite(p0) or not np.isfinite(sigma):
        return None
    if horizon_days <= 0:
        return None

    days, upper, lower = probability_cone(p0, sigma, horizon_days=horizon_days)
    x_cone = last_x + days
    ax.fill_between(
        x_cone, lower, upper,
        color=face_color, alpha=alpha, linewidth=0, zorder=1, label="Prob Cone",
    )
    ax.plot(x_cone, upper, color=edge_color, alpha=edge_alpha, linewidth=0.9, linestyle="--")
    ax.plot(x_cone, lower, color=edge_color, alpha=edge_alpha, linewidth=0.9, linestyle="--")
    return float(x_cone[-1]), float(lower.min()), float(upper.max())


def apply_tick_labels(ax, times_for_labels, period: str) -> None:
    """Index-based custom tick labels matching the prior MarketApp logic."""
    if len(times_for_labels) == 0:
        return

    if period == "1mo":
        target_count = 21
    elif period == "3mo":
        target_count = 63
    elif period == "1y":
        target_count = 52
    elif period == "5y" or period == "10y":
        target_count = 60
    elif period == "25y":
        target_count = 25
    elif period == "1d":
        target_count = 7
    else:
        target_count = 6

    tick_count = min(target_count, len(times_for_labels))
    tick_idx = np.linspace(0, len(times_for_labels) - 1, tick_count, dtype=int)
    ax.set_xticks(tick_idx)

    final_labels = []
    for i in tick_idx:
        ts = times_for_labels[i]
        if period == "1d":
            label = ts.strftime("%H:%M")
        else:
            if ts.hour == 15 and ts.minute == 55:
                ts = ts + timedelta(days=1)
            label = ts.strftime("%Y-%m-%d")
        final_labels.append(label)

    ax.set_xticklabels(final_labels, rotation=45, ha="right", fontsize=8)


def draw_main_chart(
    ax,
    figure,
    plot_df,
    times_for_labels,
    x_vals,
    ticker: str,
    period: str,
    cone: Optional[dict] = None,
) -> None:
    """Render price / EMAs / VWAP / Fib / optional probability cone onto ``ax``.

    ``cone`` keys (all optional unless noted):
      show (bool), p0 (float), sigma (float), horizon_days (int, default 30)
    """
    ax.clear()

    ax.plot(x_vals, plot_df["Close"], label="Price", color="#00e6ff", linewidth=1.5)

    if "EMA_5" in plot_df.columns:
        ax.plot(x_vals, plot_df["EMA_5"], label="EMA 5", color="#ff00ff", linewidth=1, alpha=0.8)
    if "EMA_21" in plot_df.columns:
        ax.plot(x_vals, plot_df["EMA_21"], label="EMA 21", color="#ffe100", linewidth=1, alpha=0.8)
    if "EMA_63" in plot_df.columns:
        ax.plot(x_vals, plot_df["EMA_63"], label="EMA 63", color="#9900ff", linewidth=1, alpha=0.8)
    if "EMA_200" in plot_df.columns and plot_df["EMA_200"].notna().sum() > 0:
        ax.plot(x_vals, plot_df["EMA_200"], label="EMA 200", color="#ff3333", linewidth=1.5)

    if "VWAP" in plot_df.columns and plot_df["VWAP"].notna().sum() > 0:
        ax.plot(
            x_vals, plot_df["VWAP"], label="VWAP",
            color="#ffd700", linewidth=1.5, linestyle="--",
        )

    fib_high = plot_df["High"].max()
    fib_low = plot_df["Low"].min()
    fib_range = fib_high - fib_low
    if fib_range > 0:
        fib_levels = {
            "23.6%": fib_high - 0.236 * fib_range,
            "38.2%": fib_high - 0.382 * fib_range,
            "50.0%": fib_high - 0.500 * fib_range,
            "61.8%": fib_high - 0.618 * fib_range,
        }
        fib_colors = {
            "23.6%": "#ff9800", "38.2%": "#e91e63",
            "50.0%": "#9c27b0", "61.8%": "#2196f3",
        }
        for level_name, level_val in fib_levels.items():
            ax.axhline(
                y=level_val, color=fib_colors[level_name],
                linestyle=":", linewidth=0.7, alpha=0.6,
            )
            ax.text(
                x_vals[-1], level_val, f" {level_name}",
                color=fib_colors[level_name], fontsize=6, va="center", alpha=0.8,
            )

    x_right = float(x_vals[-1])
    y_lo = float(plot_df["Low"].min()) if "Low" in plot_df.columns else float(plot_df["Close"].min())
    y_hi = float(plot_df["High"].max()) if "High" in plot_df.columns else float(plot_df["Close"].max())

    if cone and cone.get("show"):
        extent = draw_probability_cone(
            ax,
            last_x=float(x_vals[-1]),
            p0=cone.get("p0"),
            sigma=cone.get("sigma"),
            horizon_days=int(cone.get("horizon_days", 30)),
        )
        if extent is not None:
            x_right = max(x_right, extent[0])
            y_lo = min(y_lo, extent[1])
            y_hi = max(y_hi, extent[2])

    ax.set_xlim(left=float(x_vals[0]), right=x_right)
    if y_hi > y_lo:
        pad = 0.02 * (y_hi - y_lo)
        ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.set_title(f"{ticker} Price Action ({period})", color="white", fontweight="bold")
    ax.legend(loc="upper right", fontsize="small", frameon=False, labelcolor="white")
    ax.grid(True, color="#2a2a2a", linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#444444")
    ax.spines["left"].set_color("#444444")
    ax.tick_params(axis="x", colors="gray")
    ax.tick_params(axis="y", colors="gray")

    apply_tick_labels(ax, times_for_labels, period)

    ax.set_facecolor("#121212")
    figure.patch.set_facecolor("#121212")
