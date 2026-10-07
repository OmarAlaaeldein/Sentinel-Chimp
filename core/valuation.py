"""Valuation rules (P/E, PEG, 5-year P/E percentile) without a window.

These are the rules ``MarketApp`` always applied to Yahoo's ``info`` and price /
earnings history, moved out of the Tk window so the CLI and other tools use the
same code: nothing here imports tkinter or anything from ``main/`` or ``ui/``.

* P/E TTM and forward come from ``info["trailingPE"]`` / ``info["forwardPE"]``;
  a missing or non-finite value stays missing (Yahoo leaves ``trailingPE`` out
  for a company with no positive trailing earnings, and for ETFs).
* PEG is provider-first: ``info["trailingPegRatio"]``. Without it, PEG is derived
  as (forward P/E, else TTM P/E) over ``earningsGrowth`` x 100. Zero growth gives
  +/-inf with reason ``ZERO_GROWTH``; negative growth gives a negative PEG, shown
  as such; missing inputs give ``MISSING_PEG_INPUTS``.
* The P/E percentile is the share of the last 5 years of daily closes whose P/E
  (close over the TTM EPS known on that day, ``merge_asof`` backward on report
  dates) sat strictly below today's TTM P/E. Positive P/E history only.

The functions that take frames never raise: a frame that does not fit gives the
window's reason (``NO_EARNINGS_HISTORY`` / ``NO_VALID_PE_HISTORY``) and, when a
``log`` callable is passed, the message the window used to write. The only
requests are the ones ``valuation`` makes through the provider it is given.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd

LogFn = Callable[[str], None]

# reason -> the text the window shows in its P/E percentile and PEG cells
REASONS = {
    "NO_CURRENT_PE_TTM": "N/A (No TTM P/E)",
    "NO_PRICE_HISTORY": "N/A (No 5Y Price)",
    "NO_EARNINGS_HISTORY": "N/A (No EPS History)",
    "INSUFFICIENT_EARNINGS_HISTORY": "N/A (EPS < 4 Qtrs)",
    "NO_VALID_PE_HISTORY": "N/A (No Valid P/E)",
    "MISSING_PEG_INPUTS": "Not Calculable",
    "ZERO_GROWTH": "Inf (Zero Growth)",
}


def to_finite_float(value: Any) -> Optional[float]:
    """Best-effort numeric normalization for provider fields."""
    try:
        f_val = float(value)
        if not math.isfinite(f_val):
            return None
        return f_val
    except (TypeError, ValueError):
        return None


def as_naive_datetime64_us(values):
    """Normalize timestamps for merge_asof (avoids s vs us unit mismatch)."""
    ts = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(ts, pd.Series):
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert(None)
        return ts.astype("datetime64[us]")
    if isinstance(ts, pd.DatetimeIndex):
        if ts.tz is not None:
            ts = ts.tz_convert(None)
        return ts.astype("datetime64[us]")
    # Fallback scalar/array path
    ts = pd.DatetimeIndex(ts)
    if ts.tz is not None:
        ts = ts.tz_convert(None)
    return pd.Series(ts.astype("datetime64[us]"))


def read_info(info: Optional[dict]) -> dict:
    """The four provider fields the rules read, each a finite float or None."""
    info = info or {}
    return {
        "pe_fwd": to_finite_float(info.get("forwardPE")),
        "pe_ttm": to_finite_float(info.get("trailingPE")),
        "provider_peg": to_finite_float(info.get("trailingPegRatio")),
        "earnings_growth": to_finite_float(info.get("earningsGrowth")),
    }


def peg_ratio(
    provider_peg: Any, pe_fwd: Any, pe_ttm: Any, earnings_growth: Any
) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """PEG with provider-first fallback to derived: ``(peg, source, reason)``.

    ``source`` is ``"provider"``, ``"derived"`` or None (not calculable).
    """
    provider = to_finite_float(provider_peg)
    if provider is not None:
        return provider, "provider", None

    pe_for_peg = to_finite_float(pe_fwd)
    if pe_for_peg is None:
        pe_for_peg = to_finite_float(pe_ttm)

    growth_dec = to_finite_float(earnings_growth)
    if pe_for_peg is None or growth_dec is None:
        return None, None, "MISSING_PEG_INPUTS"

    growth_pct = growth_dec * 100.0
    if abs(growth_pct) < 1e-9:
        return (math.inf if pe_for_peg >= 0 else -math.inf), "derived", "ZERO_GROWTH"

    return pe_for_peg / growth_pct, "derived", None


def from_info(info: Optional[dict]) -> dict:
    """P/E TTM, forward P/E and PEG from Yahoo's ``info``.

    ``{"pe_ttm", "pe_fwd", "peg", "peg_source", "earnings_growth", "peg_reason"}``
    """
    raw = read_info(info)
    peg, source, reason = peg_ratio(
        raw["provider_peg"], raw["pe_fwd"], raw["pe_ttm"], raw["earnings_growth"]
    )
    return {
        "pe_ttm": raw["pe_ttm"],
        "pe_fwd": raw["pe_fwd"],
        "peg": peg,
        "peg_source": source,
        "earnings_growth": raw["earnings_growth"],
        "peg_reason": reason,
    }


def ttm_eps_timeline(
    earnings_df, log: Optional[LogFn] = None
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Trailing-twelve-month EPS at each report date, from reported quarterly EPS.

    ``earnings_df`` is what ``get_earnings_dates`` returns (a date index and a
    "Reported EPS" column). Returns ``(frame with report_date and ttm_eps, None)``
    or ``(None, reason)``.
    """
    try:
        if earnings_df is None or earnings_df.empty:
            return None, "NO_EARNINGS_HISTORY"

        reported_col = None
        for col in earnings_df.columns:
            col_name = str(col).lower()
            if "reported" in col_name and "eps" in col_name:
                reported_col = col
                break

        if reported_col is None:
            return None, "NO_EARNINGS_HISTORY"

        eps_series = pd.to_numeric(earnings_df[reported_col], errors='coerce').dropna()
        if eps_series.empty:
            return None, "NO_EARNINGS_HISTORY"

        eps_df = pd.DataFrame({"reported_eps": eps_series})
        eps_df["report_date"] = as_naive_datetime64_us(eps_df.index)
        eps_df = eps_df.dropna(subset=["report_date"]).sort_values("report_date")
        eps_df = eps_df.drop_duplicates(subset=["report_date"], keep="last")
        eps_df["ttm_eps"] = eps_df["reported_eps"].rolling(4).sum()
        eps_df = eps_df.dropna(subset=["ttm_eps"])
        if eps_df.empty:
            return None, "INSUFFICIENT_EARNINGS_HISTORY"

        return eps_df[["report_date", "ttm_eps"]].copy(), None
    except Exception as e:
        if log:
            log(f"Historical EPS fetch error: {e}")
        return None, "NO_EARNINGS_HISTORY"


def has_price_history(history) -> bool:
    """True when ``history`` is a non-empty frame with a ``Close`` column."""
    return not (history is None or history.empty or "Close" not in history.columns)


def pe_percentile(
    history,
    eps_timeline,
    pe_ttm: Any,
    *,
    eps_reason: Optional[str] = None,
    log: Optional[LogFn] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Where today's TTM P/E sits in the P/E of the history: ``(percent, reason)``.

    ``history`` has a ``Close`` column and a date index (5 years of daily bars);
    ``eps_timeline`` is :func:`ttm_eps_timeline`'s frame, or None with its
    reason in ``eps_reason``. The percent is the share of days whose P/E was
    strictly below ``pe_ttm``, over days with a positive P/E only.
    """
    current_pe_ttm = to_finite_float(pe_ttm)
    if current_pe_ttm is None:
        return None, "NO_CURRENT_PE_TTM"

    try:
        if not has_price_history(history):
            return None, "NO_PRICE_HISTORY"

        if eps_timeline is None or eps_timeline.empty:
            return None, eps_reason or "NO_EARNINGS_HISTORY"

        hist_df = history[["Close"]].copy().dropna()
        hist_df["date"] = as_naive_datetime64_us(hist_df.index)
        hist_df = hist_df.dropna(subset=["date"]).sort_values("date")

        eps_timeline = eps_timeline.copy()
        eps_timeline["report_date"] = as_naive_datetime64_us(eps_timeline["report_date"])
        eps_timeline = eps_timeline.dropna(subset=["report_date"]).sort_values("report_date")

        merged = pd.merge_asof(
            hist_df[["date", "Close"]],
            eps_timeline,
            left_on="date",
            right_on="report_date",
            direction="backward"
        )

        pe_series = pd.to_numeric(merged["Close"], errors='coerce') / pd.to_numeric(merged["ttm_eps"], errors='coerce')
        pe_series = pe_series.replace([np.inf, -np.inf], np.nan).dropna()

        # For percentile comparability, use positive P/E history only.
        pe_series = pe_series[pe_series > 0]
        if pe_series.empty:
            return None, "NO_VALID_PE_HISTORY"

        return float((pe_series < current_pe_ttm).mean() * 100.0), None
    except Exception as e:
        if log:
            log(f"P/E Percentile error: {e}")
        return None, "NO_VALID_PE_HISTORY"


def valuation(
    provider,
    stock,
    *,
    history=None,
    earnings_df=None,
    percentile: bool = True,
    log: Optional[LogFn] = None,
) -> dict:
    """:func:`from_info` of the provider's ``info`` plus the 5-year P/E percentile.

    One ``provider.get_info(stock)`` call. With ``percentile`` true, and only
    when Yahoo gave a TTM P/E, it also reads ``history`` (else
    ``provider.fetch_history(stock, "5y", "1d")``) and, if that has prices,
    ``earnings_df`` (else ``stock.get_earnings_dates(limit=80)``): a name with no
    P/E, or no price history, costs no further request. A failed history or
    earnings request is a reason (and a ``log`` line), never an exception;
    ``get_info`` errors propagate to the caller.

    The result has ``pe_percentile`` and ``pe_percentile_reason``; both are None
    when ``percentile`` is false.
    """
    out = from_info(provider.get_info(stock))
    out["pe_percentile"] = None
    out["pe_percentile_reason"] = None
    if not percentile:
        return out

    if out["pe_ttm"] is None:
        out["pe_percentile_reason"] = "NO_CURRENT_PE_TTM"
        return out

    try:
        if history is None:
            kwargs = {"log": log} if log else {}
            history = provider.fetch_history(stock, "5y", "1d", **kwargs)
        eps_timeline, eps_reason = None, None
        if has_price_history(history):
            if earnings_df is None:
                try:
                    earnings_df = stock.get_earnings_dates(limit=80)
                except Exception as e:
                    if log:
                        log(f"Historical EPS fetch error: {e}")
                    eps_reason = "NO_EARNINGS_HISTORY"
            if eps_reason is None:
                eps_timeline, eps_reason = ttm_eps_timeline(earnings_df, log=log)
        out["pe_percentile"], out["pe_percentile_reason"] = pe_percentile(
            history, eps_timeline, out["pe_ttm"], eps_reason=eps_reason, log=log
        )
    except Exception as e:
        if log:
            log(f"P/E Percentile error: {e}")
        out["pe_percentile"], out["pe_percentile_reason"] = None, "NO_VALID_PE_HISTORY"
    return out


def _fmt(value: Any, decimals: int = 2, fallback: str = "N/A") -> str:
    try:
        f_val = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(f_val):
        return fallback
    if math.isinf(f_val):
        return "Inf" if f_val > 0 else "-Inf"
    return f"{f_val:.{decimals}f}"


def summary_line(v: dict) -> str:
    """One line for ``analyze``, e.g.
    ``Valuation: P/E 30.20 TTM | 25.10 fwd | PEG 1.30 (provider) | P/E percentile 82.0% (5y, TTM)``.
    """
    parts = [f"P/E {_fmt(v.get('pe_ttm'))} TTM", f"{_fmt(v.get('pe_fwd'))} fwd"]

    peg = v.get("peg")
    if peg is not None:
        how = v.get("peg_source") or "derived"
        if v.get("peg_reason") == "ZERO_GROWTH":
            how += ", zero growth"
        parts.append(f"PEG {_fmt(peg)} ({how})")
    else:
        parts.append(f"PEG {REASONS.get(v.get('peg_reason'), 'N/A')}")

    pct = v.get("pe_percentile")
    if pct is not None:
        parts.append(f"P/E percentile {_fmt(pct, 1)}% (5y, TTM)")
    elif v.get("pe_percentile_reason"):
        parts.append(f"P/E percentile {REASONS.get(v['pe_percentile_reason'], 'N/A')}")

    return "Valuation: " + " | ".join(parts)
