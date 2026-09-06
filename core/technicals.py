"""Technical indicator calculations.

Classic definitions used:
- RSI / ATR / ADX / +DI / -DI: Wilder (1978) smoothing (α = 1/N).
- MACD: Appel (1979) — 12/26 EMA difference with 9-period signal.
- Bollinger Bands: Bollinger (early 1980s) — SMA ± 2σ; %B / bandwidth derived.
- Stochastic RSI: Chande & Kroll (1994) — Stoch of RSI with 3/3 SMA %K/%D.
- VWAP: session typical-price × volume / cumulative volume (daily reset).
- OBV: Granville (1963).
- Williams %R: Williams (1973) — 14-period.
- CCI: Lambert (1980) — 20-period, 0.015 constant.
- Chart EMAs: span on **daily** closes (α=2/(N+1)); see attach_daily_emas.
- Ichimoku: Hosoda standard 9/26/52 with 26-bar displacement.
- Fib: latest confirmed swing high/low (not period max/min).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _rolling_mad(values: np.ndarray, window: int) -> np.ndarray:
    """Exact rolling mean absolute deviation via sliding windows (no Python apply)."""
    n = values.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    windows = sliding_window_view(values, window_shape=window)
    means = windows.mean(axis=1)
    mad = np.abs(windows - means[:, None]).mean(axis=1)
    out[window - 1 :] = mad
    return out


def calculate_technicals(df):
    """Add standard technical columns to an OHLCV DataFrame (in-place)."""
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    delta = close.diff()

    # 1. RSI (Wilder's Smoothing)
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. MACD with Histogram
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # 3. Bollinger Bands with %B and Bandwidth
    df['SMA_20'] = close.rolling(window=20).mean()
    df['STD_20'] = close.rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)
    bb_range = df['BB_Upper'] - df['BB_Lower']
    df['BB_PctB'] = (close - df['BB_Lower']) / bb_range.replace(0, np.nan)
    df['BB_Width'] = bb_range / df['SMA_20'].replace(0, np.nan)

    # 4. ATR (Wilder's Smoothing)
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.ewm(alpha=1 / 14, adjust=False).mean()

    # 5. StochRSI with %K/%D Smoothing
    min_rsi = df['RSI'].rolling(window=14).min()
    max_rsi = df['RSI'].rolling(window=14).max()
    rsi_range = (max_rsi - min_rsi).replace(0, np.nan)
    raw_stoch = (df['RSI'] - min_rsi) / rsi_range
    df['StochRSI_K'] = raw_stoch.rolling(3).mean()
    df['StochRSI_D'] = df['StochRSI_K'].rolling(3).mean()
    df['StochRSI'] = df['StochRSI_K']

    # 6. VWAP with Daily Reset — reuse typical price for CCI below
    tp = (high + low + close) / 3.0
    trade_date = df.index.normalize()
    tp_vol = tp * volume
    # Groupby cumsums without mutating helper columns onto df long-term
    vwap_num = tp_vol.groupby(trade_date).cumsum()
    vwap_den = volume.groupby(trade_date).cumsum()
    df['VWAP'] = vwap_num / vwap_den.replace(0, np.nan)

    # 7. OBV (On-Balance Volume)
    df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df['OBV_SMA'] = df['OBV'].rolling(20).mean()

    # 8. ADX with Directional Info
    up_move = high.diff()
    down_move = -low.diff()  # Low.shift(1) - Low  == -diff
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm_s = pd.Series(plus_dm, index=df.index)
    minus_dm_s = pd.Series(minus_dm, index=df.index)

    atr = df['ATR']
    df['+DI'] = 100 * (plus_dm_s.ewm(alpha=1 / 14, adjust=False).mean() / atr)
    df['-DI'] = 100 * (minus_dm_s.ewm(alpha=1 / 14, adjust=False).mean() / atr)
    di_sum = (df['+DI'] + df['-DI']).replace(0, np.nan)
    dx = 100 * (df['+DI'] - df['-DI']).abs() / di_sum
    df['ADX'] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # 9. Williams %R
    highest_14 = high.rolling(14).max()
    lowest_14 = low.rolling(14).min()
    df['Williams_R'] = -100 * (highest_14 - close) / (highest_14 - lowest_14).replace(0, np.nan)

    # 10. CCI — vectorized MAD (Lambert 1980); reuse tp
    tp_sma = tp.rolling(20).mean()
    tp_mad = pd.Series(_rolling_mad(tp.to_numpy(dtype=np.float64, copy=False), 20), index=df.index)
    df['CCI'] = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

    return df


# ---------------------------------------------------------------------------
# Chart overlays: daily EMAs, Ichimoku, Fib swing anchors
# ---------------------------------------------------------------------------

EMA_SPANS = (5, 21, 63, 200)


def compute_ema_series(close: pd.Series, span: int) -> pd.Series:
    """Classic chart EMA: pandas ``ewm(span=N, adjust=False)`` ⇒ α = 2/(N+1).

    This is *not* RiskMetrics EWMA variance (see ``VegaChimpCore.ewma_vol_forecast``).
    """
    close = pd.Series(close, dtype=np.float64)
    return close.ewm(span=int(span), adjust=False).mean()


def compute_daily_emas(daily_close: pd.Series, spans=EMA_SPANS) -> pd.DataFrame:
    """EMAs on a **daily** close series (spans = trading days)."""
    daily_close = pd.Series(daily_close, dtype=np.float64).dropna()
    out = pd.DataFrame(index=daily_close.index)
    for span in spans:
        out[f"EMA_{int(span)}"] = compute_ema_series(daily_close, int(span))
    return out


def _normalize_index_to_naive_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def _index_looks_intraday(index: pd.DatetimeIndex) -> bool:
    """True for intraday charts (tight bar spacing or non-midnight timestamps)."""
    idx = pd.DatetimeIndex(index)
    if len(idx) >= 2:
        ordered = idx.sort_values()
        med = pd.Series(ordered).diff().median()
        if pd.notna(med) and med < pd.Timedelta(hours=12):
            return True
    # Sparse / single bar: any non-midnight clock time ⇒ intraday print
    for ts in idx:
        t = pd.Timestamp(ts)
        if getattr(t, "tzinfo", None) is not None or getattr(t, "tz", None) is not None:
            t = t.tz_convert("America/New_York")
        if t.hour != 0 or t.minute != 0 or t.second != 0 or t.microsecond != 0:
            return True
    return False


def _stamp_daily_at_rth_close(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Mark each daily bar as available at 16:00 America/New_York (naive UTC).

    Morning intraday bars must not see the same session's close/EMA (look-ahead).
    """
    idx = pd.DatetimeIndex(index)
    if getattr(idx, "tz", None) is not None:
        et = idx.tz_convert("America/New_York")
    else:
        # Session date at midnight — interpret as US equity calendar date in ET.
        et = idx.tz_localize(
            "America/New_York", ambiguous="infer", nonexistent="shift_forward"
        )
    stamped = et.normalize() + pd.Timedelta(hours=16)
    return stamped.tz_convert("UTC").tz_localize(None)


def map_daily_columns_to_bars(
    bar_index: pd.DatetimeIndex,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    """Forward-fill daily columns onto an arbitrary bar timeline (as-of merge).

    Each bar receives the last **completed** daily value as of that bar's
    timestamp. On intraday charts, daily rows are stamped at RTH close
    (16:00 America/New_York) so morning bars do not see the same-day close.
    Daily→daily mapping leaves timestamps unchanged so the close bar includes
    that day in the EMA. Prevents treating ``EMA_200`` as 200 *minute* bars.
    """
    if daily_df is None or daily_df.empty or len(bar_index) == 0:
        return pd.DataFrame(index=bar_index)

    bar_index = pd.DatetimeIndex(bar_index)
    bar_ts = _normalize_index_to_naive_utc(bar_index)
    bars = pd.DataFrame({"_ts": bar_ts, "_orig": np.arange(len(bar_ts))})
    daily = daily_df.copy()
    daily_idx = pd.DatetimeIndex(daily.index)
    if _index_looks_intraday(bar_index):
        daily.index = _stamp_daily_at_rth_close(daily_idx)
    else:
        daily.index = _normalize_index_to_naive_utc(daily_idx)
    daily = daily.sort_index()
    daily = daily[~daily.index.duplicated(keep="last")]
    daily_reset = daily.reset_index()
    ts_col = daily_reset.columns[0]
    daily_reset = daily_reset.rename(columns={ts_col: "_ts"})
    merged = pd.merge_asof(
        bars.sort_values("_ts"),
        daily_reset.sort_values("_ts"),
        on="_ts",
        direction="backward",
    )
    merged = merged.sort_values("_orig")
    cols = [c for c in daily_df.columns]
    out = merged[cols].copy()
    out.index = bar_index
    return out


def attach_daily_emas(
    bar_df: pd.DataFrame,
    daily_close: pd.Series,
    spans=EMA_SPANS,
) -> pd.DataFrame:
    """Overwrite ``EMA_*`` on ``bar_df`` with daily-span EMAs mapped to bars."""
    emas = compute_daily_emas(daily_close, spans=spans)
    mapped = map_daily_columns_to_bars(bar_df.index, emas)
    for col in mapped.columns:
        bar_df[col] = mapped[col].values
    return bar_df


def calculate_ichimoku(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """Standard Ichimoku Kinko Hyo (Goichi Hosoda).

    Tenkan / Kijun / Senkou B = midpoint of rolling high/low.
    Senkou A = (Tenkan + Kijun) / 2.
    Senkou A/B are shifted **forward** by ``displacement``; Chikou is Close
    shifted **backward** by ``displacement``.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2.0
    kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2.0
    senkou_a = ((tenkan + kijun) / 2.0).shift(displacement)
    senkou_b = (
        (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2.0
    ).shift(displacement)
    chikou = close.shift(-displacement)

    df["Ichimoku_Tenkan"] = tenkan
    df["Ichimoku_Kijun"] = kijun
    df["Ichimoku_Senkou_A"] = senkou_a
    df["Ichimoku_Senkou_B"] = senkou_b
    df["Ichimoku_Chikou"] = chikou
    return df


def find_pivot_indices(
    high: np.ndarray,
    low: np.ndarray,
    left: int = 5,
    right: int = 5,
) -> tuple:
    """Return ``(swing_high_idxs, swing_low_idxs)`` for fractal pivots.

    A swing high at ``i`` is the max of ``high[i-left:i+right+1]`` (inclusive);
    likewise for swing lows. Requires ``left`` bars on each side (no look-ahead
    past ``right`` confirmation bars at the series end).
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    n = high.size
    sh, sl = [], []
    if n < left + right + 1:
        return np.array([], dtype=int), np.array([], dtype=int)
    for i in range(left, n - right):
        window_h = high[i - left : i + right + 1]
        window_l = low[i - left : i + right + 1]
        if np.isfinite(high[i]) and high[i] == np.nanmax(window_h):
            # unique peak: first occurrence of the max wins
            if np.nanargmax(window_h) == left:
                sh.append(i)
        if np.isfinite(low[i]) and low[i] == np.nanmin(window_l):
            if np.nanargmin(window_l) == left:
                sl.append(i)
    return np.asarray(sh, dtype=int), np.asarray(sl, dtype=int)


def fib_swing_anchors(
    high,
    low,
    close=None,
    pivot_left: int = 5,
    pivot_right: int = 5,
) -> dict:
    """Fibonacci retracement anchored to the **latest confirmed swing**.

    Rule (see also ``docs/LOGIC_REVIEW.md``):
    - Detect fractal swing highs/lows with ``pivot_left`` / ``pivot_right``
      confirmation bars (default 5/5).
    - Take the most recent swing (by index) and the nearest **opposite** swing
      that precedes it — that pair is the latest impulse.
    - If the latest swing is a **high**, Fib retraces from that high down to the
      preceding swing low (bearish impulse). If latest is a **low**, Fib
      retraces from the preceding swing high down to that low (bullish impulse).
    - Levels are ``high - pct * (high - low)`` for classic 23.6 / 38.2 / 50 / 61.8.
    - Fallback: if fewer than two opposite pivots exist, use the visible-window
      max high / min low (legacy behaviour).

    Returns dict with keys: ``fib_high``, ``fib_low``, ``high_idx``, ``low_idx``,
    ``rule``, ``ok``.
    """
    high_a = np.asarray(high, dtype=np.float64).reshape(-1)
    low_a = np.asarray(low, dtype=np.float64).reshape(-1)
    n = high_a.size
    result = {
        "fib_high": float("nan"),
        "fib_low": float("nan"),
        "high_idx": -1,
        "low_idx": -1,
        "rule": "fallback_window",
        "ok": False,
    }
    if n == 0:
        return result

    sh, sl = find_pivot_indices(high_a, low_a, left=pivot_left, right=pivot_right)

    # Merge pivots as (idx, kind) with kind 1=high, -1=low
    events = [(int(i), 1, float(high_a[i])) for i in sh] + [
        (int(i), -1, float(low_a[i])) for i in sl
    ]
    events.sort(key=lambda t: t[0])

    fib_high = fib_low = None
    hi_idx = lo_idx = -1
    rule = "fallback_window"

    if len(events) >= 2:
        last_idx, last_kind, last_val = events[-1]
        # walk backward for opposite kind
        prior = None
        for e in reversed(events[:-1]):
            if e[1] != last_kind:
                prior = e
                break
        if prior is not None:
            p_idx, p_kind, p_val = prior
            if last_kind == 1:  # latest high → retrace from high to prior low
                fib_high, fib_low = last_val, p_val
                hi_idx, lo_idx = last_idx, p_idx
                rule = "latest_swing_high_to_prior_low"
            else:  # latest low → retrace from prior high to this low
                fib_high, fib_low = p_val, last_val
                hi_idx, lo_idx = p_idx, last_idx
                rule = "prior_swing_high_to_latest_low"

    if fib_high is None or fib_low is None or not (
        math.isfinite(fib_high) and math.isfinite(fib_low)
    ) or fib_high <= fib_low:
        # Legacy fallback: period extreme
        hi_idx = int(np.nanargmax(high_a)) if np.isfinite(high_a).any() else -1
        lo_idx = int(np.nanargmin(low_a)) if np.isfinite(low_a).any() else -1
        fib_high = float(high_a[hi_idx]) if hi_idx >= 0 else float("nan")
        fib_low = float(low_a[lo_idx]) if lo_idx >= 0 else float("nan")
        rule = "fallback_window"

    ok = (
        math.isfinite(fib_high)
        and math.isfinite(fib_low)
        and fib_high > fib_low
    )
    result.update(
        {
            "fib_high": float(fib_high) if ok else float("nan"),
            "fib_low": float(fib_low) if ok else float("nan"),
            "high_idx": int(hi_idx),
            "low_idx": int(lo_idx),
            "rule": rule,
            "ok": bool(ok),
        }
    )
    return result


def fib_retracement_levels(fib_high: float, fib_low: float) -> dict:
    """Classic retracement prices from a swing high/low pair."""
    rng = float(fib_high) - float(fib_low)
    if rng <= 0 or not math.isfinite(rng):
        return {}
    return {
        "23.6%": fib_high - 0.236 * rng,
        "38.2%": fib_high - 0.382 * rng,
        "50.0%": fib_high - 0.500 * rng,
        "61.8%": fib_high - 0.618 * rng,
    }


def bars_per_trading_day(interval: str | None) -> float:
    """Approximate RTH bars per trading day for cone x-scaling."""
    if not interval:
        return 1.0
    key = str(interval).lower().strip()
    table = {
        "1m": 390.0,
        "2m": 195.0,
        "5m": 78.0,
        "15m": 26.0,
        "30m": 13.0,
        "60m": 6.5,
        "90m": 390.0 / 90.0,
        "1h": 6.5,
        "1d": 1.0,
        "1wk": 1.0 / 5.0,
        "1mo": 1.0 / 21.0,
    }
    return float(table.get(key, 1.0))
