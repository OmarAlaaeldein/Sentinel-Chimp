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
"""
from __future__ import annotations

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
