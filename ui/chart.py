"""Main price chart rendering (extracted from MarketApp.update_chart)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ui.theme import chart_colors


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
    bars_per_day: float = 1.0,
    face_color: str = "#4db8c4",
    alpha: float = 0.16,
    edge_color: str = "#4db8c4",
    edge_alpha: float = 0.5,
) -> Optional[Tuple[float, float, float]]:
    """Draw a translucent vol cone extending ``horizon_days`` right of last_x.

    ``bars_per_day`` scales trading-day horizon onto the chart's bar axis so
    intraday charts do not treat +30 as +30 minute-bars.

    Returns ``(x_right, y_lo, y_hi)`` useful for axis padding, or None if
    inputs are not drawable.
    """
    from core.vol_models import probability_cone

    if p0 is None or sigma is None:
        return None
    try:
        p0 = float(p0)
        sigma = float(sigma)
        bars_per_day = float(bars_per_day) if bars_per_day else 1.0
    except (TypeError, ValueError):
        return None
    if p0 <= 0 or sigma < 0 or not np.isfinite(p0) or not np.isfinite(sigma):
        return None
    if horizon_days <= 0 or bars_per_day <= 0 or not np.isfinite(bars_per_day):
        return None

    days, upper, lower = probability_cone(p0, sigma, horizon_days=horizon_days)
    x_cone = last_x + days * bars_per_day
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

    ax.set_xticklabels(final_labels, rotation=45, ha="right", fontsize=10)


def _to_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        ts = pd.to_datetime(value)
        if pd.isna(ts):
            return None
        if getattr(ts, "tzinfo", None) is not None or getattr(ts, "tz", None) is not None:
            ts = pd.Timestamp(ts).tz_convert("UTC").tz_localize(None)
        return pd.Timestamp(ts).date()
    except (TypeError, ValueError, OverflowError):
        return None


def draw_earnings_markers(
    ax,
    times_for_labels,
    x_vals,
    earnings_dates: Optional[Sequence],
) -> None:
    """Vertical markers where an earnings date falls inside the visible window."""
    if not earnings_dates or len(times_for_labels) == 0:
        return
    bar_dates = []
    for ts in times_for_labels:
        d = _to_date(ts)
        bar_dates.append(d)
    labeled = False
    for ed in earnings_dates:
        ed_date = _to_date(ed)
        if ed_date is None:
            continue
        # nearest bar on/after earnings date (or exact match)
        hit = None
        for i, bd in enumerate(bar_dates):
            if bd is None:
                continue
            if bd == ed_date:
                hit = i
                break
            if bd > ed_date:
                hit = i
                break
        if hit is None:
            continue
        x = float(x_vals[hit])
        ax.axvline(
            x=x, color="#ffb74d", linestyle="--", linewidth=1.0, alpha=0.7,
            label="Earnings" if not labeled else None, zorder=2,
        )
        labeled = True


def draw_ichimoku(ax, x_vals, plot_df) -> None:
    """Plot Tenkan/Kijun/Chikou and Senkou cloud when columns are present."""
    tenkan = plot_df.get("Ichimoku_Tenkan")
    kijun = plot_df.get("Ichimoku_Kijun")
    sa = plot_df.get("Ichimoku_Senkou_A")
    sb = plot_df.get("Ichimoku_Senkou_B")
    chikou = plot_df.get("Ichimoku_Chikou")

    if tenkan is not None and tenkan.notna().any():
        ax.plot(x_vals, tenkan, label="Tenkan", color="#4fc3f7", linewidth=1.0, alpha=0.9)
    if kijun is not None and kijun.notna().any():
        ax.plot(x_vals, kijun, label="Kijun", color="#ef5350", linewidth=1.0, alpha=0.9)
    if chikou is not None and chikou.notna().any():
        ax.plot(
            x_vals, chikou, label="Chikou", color="#ce93d8",
            linewidth=0.9, alpha=0.75, linestyle=":",
        )
    if sa is not None and sb is not None and sa.notna().any() and sb.notna().any():
        ax.fill_between(
            x_vals, sa, sb,
            where=(sa >= sb), color="#26a69a", alpha=0.18, linewidth=0, label="Cloud+",
        )
        ax.fill_between(
            x_vals, sa, sb,
            where=(sa < sb), color="#ef5350", alpha=0.18, linewidth=0, label="Cloud-",
        )


def draw_fib_levels(ax, x_vals, plot_df) -> None:
    """Fibonacci retracement from latest swing anchors (not period max/min)."""
    from core.technicals import fib_retracement_levels, fib_swing_anchors

    if "High" not in plot_df.columns or "Low" not in plot_df.columns:
        return
    anchors = fib_swing_anchors(plot_df["High"].values, plot_df["Low"].values)
    if not anchors.get("ok"):
        return
    levels = fib_retracement_levels(anchors["fib_high"], anchors["fib_low"])
    if not levels:
        return
    fib_colors = {
        "23.6%": "#ff9800", "38.2%": "#e91e63",
        "50.0%": "#9c27b0", "61.8%": "#2196f3",
    }
    for level_name, level_val in levels.items():
        ax.axhline(
            y=level_val, color=fib_colors[level_name],
            linestyle=":", linewidth=0.7, alpha=0.55,
        )
        ax.text(
            x_vals[-1], level_val, f" {level_name}",
            color=fib_colors[level_name], fontsize=8, va="center", alpha=0.75,
        )


def draw_main_chart(
    ax,
    figure,
    plot_df,
    times_for_labels,
    x_vals,
    ticker: str,
    period: str,
    cone: Optional[dict] = None,
    show_fib: bool = False,
    show_ichimoku: bool = False,
    show_earnings: bool = False,
    earnings_dates: Optional[Sequence] = None,
    interval: Optional[str] = None,
) -> None:
    """Render price / daily EMAs / VWAP / optional Fib / Ichimoku / earnings / cone.

    ``cone`` keys (all optional unless noted):
      show (bool), p0 (float), sigma (float), horizon_days (int, default 30),
      bars_per_day (float, default from ``interval``)
    Overlay toggles default off except when callers pass True (Fib / Ichimoku /
    Earnings match the Fib checkbox pattern).
    EMA labels use daily spans (EMA 5d … 200d) — see ``attach_daily_emas``.
    """
    ax.clear()
    pal = chart_colors()

    ax.plot(x_vals, plot_df["Close"], label="Price", color=pal["price"], linewidth=1.6)

    # Daily-span EMAs (columns still named EMA_N; labels clarify "d")
    ema_style = {
        "EMA_5": ("EMA 5d", "#c084fc", 1.0),
        "EMA_21": ("EMA 21d", "#f0c14a", 1.0),
        "EMA_63": ("EMA 63d", "#7c6cf0", 1.0),
        "EMA_200": ("EMA 200d", "#f07178", 1.5),
    }
    for col, (label, color, lw) in ema_style.items():
        if col in plot_df.columns and plot_df[col].notna().sum() > 0:
            ax.plot(x_vals, plot_df[col], label=label, color=color, linewidth=lw, alpha=0.85)

    if "VWAP" in plot_df.columns and plot_df["VWAP"].notna().sum() > 0:
        ax.plot(
            x_vals, plot_df["VWAP"], label="VWAP",
            color="#e8c547", linewidth=1.3, linestyle="--", alpha=0.9,
        )

    if show_ichimoku:
        draw_ichimoku(ax, x_vals, plot_df)

    if show_fib:
        draw_fib_levels(ax, x_vals, plot_df)

    if show_earnings:
        draw_earnings_markers(ax, times_for_labels, x_vals, earnings_dates)

    x_right = float(x_vals[-1])
    y_lo = float(plot_df["Low"].min()) if "Low" in plot_df.columns else float(plot_df["Close"].min())
    y_hi = float(plot_df["High"].max()) if "High" in plot_df.columns else float(plot_df["Close"].max())

    if cone and cone.get("show"):
        from core.technicals import bars_per_trading_day

        bpd = cone.get("bars_per_day")
        if bpd is None:
            bpd = bars_per_trading_day(interval or cone.get("interval"))
        extent = draw_probability_cone(
            ax,
            last_x=float(x_vals[-1]),
            p0=cone.get("p0"),
            sigma=cone.get("sigma"),
            horizon_days=int(cone.get("horizon_days", 30)),
            bars_per_day=float(bpd),
            face_color=pal["cone"],
            edge_color=pal["cone"],
        )
        if extent is not None:
            x_right = max(x_right, extent[0])
            y_lo = min(y_lo, extent[1])
            y_hi = max(y_hi, extent[2])

    ax.set_xlim(left=float(x_vals[0]), right=x_right)
    if y_hi > y_lo:
        pad = 0.02 * (y_hi - y_lo)
        ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.set_title(
        f"{ticker}  ·  {period}",
        color=pal["title"], fontweight="semibold", fontsize=13, pad=10,
    )
    ax.legend(
        loc="upper right", fontsize=9, frameon=False,
        labelcolor=pal["muted"],
    )
    ax.grid(True, color=pal["grid"], linestyle="-", linewidth=0.6, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(pal["spine"])
    ax.spines["left"].set_color(pal["spine"])
    ax.tick_params(axis="x", colors=pal["tick"], labelsize=10)
    ax.tick_params(axis="y", colors=pal["tick"], labelsize=10)

    apply_tick_labels(ax, times_for_labels, period)

    ax.set_facecolor(pal["face"])
    figure.patch.set_facecolor(pal["figure"])
