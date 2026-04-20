"""
Cross-validation of calculate_technicals() against independent reference
implementations.  Each indicator is recomputed from scratch using the
canonical textbook formula and compared to the output of
calculate_technicals().

Reference formulas follow:
  - RSI: Wilder (1978) smoothed RS  (EWM alpha=1/14)
  - MACD: Appel — EMA-12 minus EMA-26, signal = EMA-9 of MACD
  - Bollinger Bands: Bollinger — SMA-20 ± 2×σ-20
  - ATR: Wilder — EWM alpha=1/14 of True Range
  - StochRSI: Chande & Kroll — (RSI - min14) / (max14 - min14), K=SMA-3, D=SMA-3 of K
  - VWAP: cumulative TP×Vol / cumulative Vol  (daily reset)
  - OBV: Granville — cumulative sign(ΔClose) × Volume
  - ADX: Wilder — smoothed DX from +DI/-DI
  - Williams %R: Williams — -100 × (H14 - C) / (H14 - L14)
  - CCI: Lambert — (TP - SMA-20(TP)) / (0.015 × MAD-20(TP))
"""
import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sentinel import calculate_technicals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_ohlcv(seed=42, n=250):
    """Realistic synthetic OHLCV with intraday range and varying volume."""
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.normal(0.02, 1.0, n))
    high = close + rng.uniform(0.2, 2.0, n)
    low = close - rng.uniform(0.2, 2.0, n)
    volume = rng.randint(500_000, 3_000_000, n).astype(float)
    dates = pd.date_range('2024-01-02', periods=n, freq='h')
    return pd.DataFrame({
        'Open': close,
        'High': high,
        'Low':  low,
        'Close': close,
        'Volume': volume,
    }, index=dates)


def ref_rsi(close: pd.Series, period=14) -> pd.Series:
    """Reference RSI using Wilder smoothing (EWM alpha=1/period)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def ref_macd(close: pd.Series):
    """Reference MACD, Signal, Histogram."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def ref_bollinger(close: pd.Series, window=20, num_std=2):
    """Reference Bollinger Bands: SMA, Upper, Lower, %B, Bandwidth."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bw = upper - lower
    pctb = (close - lower) / bw.replace(0, np.nan)
    width = bw / sma.replace(0, np.nan)
    return sma, upper, lower, pctb, width


def ref_atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    """Reference ATR using Wilder smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def ref_stochrsi(rsi: pd.Series, rsi_period=14, k_period=3, d_period=3):
    """Reference Stochastic RSI with K/D smoothing."""
    min_rsi = rsi.rolling(rsi_period).min()
    max_rsi = rsi.rolling(rsi_period).max()
    stoch = (rsi - min_rsi) / (max_rsi - min_rsi)
    k = stoch.rolling(k_period).mean()
    d = k.rolling(d_period).mean()
    return stoch, k, d


def ref_vwap(df: pd.DataFrame):
    """Reference VWAP with daily reset."""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    trade_date = df.index.normalize()
    tp_vol = tp * df['Volume']
    cum_tpv = tp_vol.groupby(trade_date).cumsum()
    cum_vol = df['Volume'].groupby(trade_date).cumsum()
    return cum_tpv / cum_vol


def ref_obv(close: pd.Series, volume: pd.Series):
    """Reference OBV — Granville."""
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()


def ref_adx(df: pd.DataFrame, period=14):
    """Reference ADX from +DI/-DI."""
    up_move = df['High'] - df['High'].shift(1)
    down_move = df['Low'].shift(1) - df['Low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    atr = ref_atr(df['High'], df['Low'], df['Close'], period)
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return plus_di, minus_di, dx, adx


def ref_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    """Reference Williams %R."""
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def ref_cci(high: pd.Series, low: pd.Series, close: pd.Series, period=20):
    """Reference CCI — Lambert."""
    tp = (high + low + close) / 3
    tp_sma = tp.rolling(period).mean()
    tp_mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - tp_sma) / (0.015 * tp_mad)


# ---------------------------------------------------------------------------
# Build data once, compute both versions
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data():
    df = build_ohlcv(seed=42, n=250)
    result = calculate_technicals(df.copy())
    return df, result


# ---------------------------------------------------------------------------
# Tests — each compares sentinel's output vs an independent computation
# ---------------------------------------------------------------------------

class TestRSICrossCheck:
    def test_rsi_matches_reference(self, data):
        df, out = data
        expected = ref_rsi(df['Close'])
        mask = expected.notna() & out['RSI'].notna()
        diff = (out['RSI'][mask] - expected[mask]).abs()
        assert diff.max() < 1e-10, f"RSI max diff = {diff.max()}"


class TestMACDCrossCheck:
    def test_macd_matches(self, data):
        df, out = data
        macd, signal, hist = ref_macd(df['Close'])
        mask = macd.notna() & out['MACD'].notna()
        assert (out['MACD'][mask] - macd[mask]).abs().max() < 1e-10

    def test_signal_matches(self, data):
        df, out = data
        _, signal, _ = ref_macd(df['Close'])
        mask = signal.notna() & out['Signal'].notna()
        assert (out['Signal'][mask] - signal[mask]).abs().max() < 1e-10

    def test_histogram_matches(self, data):
        df, out = data
        _, _, hist = ref_macd(df['Close'])
        mask = hist.notna() & out['MACD_Hist'].notna()
        assert (out['MACD_Hist'][mask] - hist[mask]).abs().max() < 1e-10


class TestBollingerCrossCheck:
    def test_sma20_matches(self, data):
        df, out = data
        sma, _, _, _, _ = ref_bollinger(df['Close'])
        mask = sma.notna() & out['SMA_20'].notna()
        assert (out['SMA_20'][mask] - sma[mask]).abs().max() < 1e-10

    def test_upper_matches(self, data):
        df, out = data
        _, upper, _, _, _ = ref_bollinger(df['Close'])
        mask = upper.notna() & out['BB_Upper'].notna()
        assert (out['BB_Upper'][mask] - upper[mask]).abs().max() < 1e-10

    def test_lower_matches(self, data):
        df, out = data
        _, _, lower, _, _ = ref_bollinger(df['Close'])
        mask = lower.notna() & out['BB_Lower'].notna()
        assert (out['BB_Lower'][mask] - lower[mask]).abs().max() < 1e-10

    def test_pctb_matches(self, data):
        df, out = data
        _, _, _, pctb, _ = ref_bollinger(df['Close'])
        mask = pctb.notna() & out['BB_PctB'].notna()
        assert (out['BB_PctB'][mask] - pctb[mask]).abs().max() < 1e-10

    def test_bandwidth_matches(self, data):
        df, out = data
        _, _, _, _, width = ref_bollinger(df['Close'])
        mask = width.notna() & out['BB_Width'].notna()
        assert (out['BB_Width'][mask] - width[mask]).abs().max() < 1e-10


class TestATRCrossCheck:
    def test_atr_matches(self, data):
        df, out = data
        expected = ref_atr(df['High'], df['Low'], df['Close'])
        mask = expected.notna() & out['ATR'].notna()
        assert (out['ATR'][mask] - expected[mask]).abs().max() < 1e-10


class TestStochRSICrossCheck:
    def test_stochrsi_k_matches(self, data):
        df, out = data
        rsi = ref_rsi(df['Close'])
        _, k, _ = ref_stochrsi(rsi)
        mask = k.notna() & out['StochRSI_K'].notna()
        assert (out['StochRSI_K'][mask] - k[mask]).abs().max() < 1e-10

    def test_stochrsi_d_matches(self, data):
        df, out = data
        rsi = ref_rsi(df['Close'])
        _, _, d = ref_stochrsi(rsi)
        mask = d.notna() & out['StochRSI_D'].notna()
        assert (out['StochRSI_D'][mask] - d[mask]).abs().max() < 1e-10

    def test_stochrsi_equals_k(self, data):
        """StochRSI column should equal StochRSI_K in the implementation."""
        _, out = data
        mask = out['StochRSI'].notna() & out['StochRSI_K'].notna()
        assert (out['StochRSI'][mask] - out['StochRSI_K'][mask]).abs().max() < 1e-10


class TestVWAPCrossCheck:
    def test_vwap_matches(self, data):
        df, out = data
        expected = ref_vwap(df)
        mask = expected.notna() & out['VWAP'].notna()
        assert (out['VWAP'][mask] - expected[mask]).abs().max() < 1e-10


class TestOBVCrossCheck:
    def test_obv_matches(self, data):
        df, out = data
        expected = ref_obv(df['Close'], df['Volume'])
        mask = expected.notna() & out['OBV'].notna()
        assert (out['OBV'][mask] - expected[mask]).abs().max() < 1e-10

    def test_obv_sma_matches(self, data):
        df, out = data
        expected_obv = ref_obv(df['Close'], df['Volume'])
        expected_sma = expected_obv.rolling(20).mean()
        mask = expected_sma.notna() & out['OBV_SMA'].notna()
        assert (out['OBV_SMA'][mask] - expected_sma[mask]).abs().max() < 1e-10


class TestADXCrossCheck:
    def test_plus_di_matches(self, data):
        df, out = data
        plus_di, _, _, _ = ref_adx(df)
        mask = plus_di.notna() & out['+DI'].notna()
        assert (out['+DI'][mask] - plus_di[mask]).abs().max() < 1e-10

    def test_minus_di_matches(self, data):
        df, out = data
        _, minus_di, _, _ = ref_adx(df)
        mask = minus_di.notna() & out['-DI'].notna()
        assert (out['-DI'][mask] - minus_di[mask]).abs().max() < 1e-10

    def test_adx_matches(self, data):
        df, out = data
        _, _, _, adx = ref_adx(df)
        mask = adx.notna() & out['ADX'].notna()
        assert (out['ADX'][mask] - adx[mask]).abs().max() < 1e-10


class TestWilliamsRCrossCheck:
    def test_williams_r_matches(self, data):
        df, out = data
        expected = ref_williams_r(df['High'], df['Low'], df['Close'])
        mask = expected.notna() & out['Williams_R'].notna()
        assert (out['Williams_R'][mask] - expected[mask]).abs().max() < 1e-10


class TestCCICrossCheck:
    def test_cci_matches(self, data):
        df, out = data
        expected = ref_cci(df['High'], df['Low'], df['Close'])
        mask = expected.notna() & out['CCI'].notna()
        assert (out['CCI'][mask] - expected[mask]).abs().max() < 1e-10


# ---------------------------------------------------------------------------
# Bonus: sanity checks on value ranges (catch formula-level bugs)
# ---------------------------------------------------------------------------

class TestSanityRanges:
    def test_rsi_bounded(self, data):
        _, out = data
        valid = out['RSI'].dropna()
        assert valid.min() >= 0 and valid.max() <= 100

    def test_adx_bounded(self, data):
        _, out = data
        valid = out['ADX'].dropna()
        assert valid.min() >= 0 and valid.max() <= 100

    def test_williams_r_bounded(self, data):
        _, out = data
        valid = out['Williams_R'].dropna()
        assert valid.min() >= -100.01 and valid.max() <= 0.01

    def test_stochrsi_bounded(self, data):
        _, out = data
        valid = out['StochRSI'].dropna()
        assert valid.min() >= -0.01 and valid.max() <= 1.01

    def test_atr_positive(self, data):
        _, out = data
        valid = out['ATR'].dropna()
        assert (valid > 0).all()

    def test_bb_order(self, data):
        _, out = data
        valid = out.dropna(subset=['BB_Lower', 'SMA_20', 'BB_Upper'])
        assert (valid['BB_Lower'] <= valid['SMA_20']).all()
        assert (valid['SMA_20'] <= valid['BB_Upper']).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
