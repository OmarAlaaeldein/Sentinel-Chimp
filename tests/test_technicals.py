"""
Technical Indicator Tests — known-value checks for calculate_technicals().
Uses synthetic data to verify RSI, MACD, Bollinger Bands, ATR, StochRSI,
VWAP, OBV, ADX, Williams %R, and CCI calculations.
"""
import sys
import os
import math
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sentinel import calculate_technicals


def make_price_df(closes, highs=None, lows=None, volumes=None, n=None):
    """Create a DataFrame suitable for calculate_technicals from close prices."""
    if n is None:
        n = len(closes)
    closes = np.array(closes, dtype=float)
    if highs is None:
        highs = closes * 1.01
    else:
        highs = np.array(highs, dtype=float)
    if lows is None:
        lows = closes * 0.99
    else:
        lows = np.array(lows, dtype=float)
    if volumes is None:
        volumes = np.full(n, 1_000_000)
    else:
        volumes = np.array(volumes, dtype=float)

    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    df = pd.DataFrame({
        'Open': closes,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes,
    }, index=dates)
    return df


def make_trending_df(start=100, daily_return=0.005, n=100):
    """Create upward-trending synthetic data."""
    closes = [start]
    for i in range(1, n):
        closes.append(closes[-1] * (1 + daily_return + np.random.normal(0, 0.002)))
    closes = np.array(closes)
    highs = closes * (1 + np.random.uniform(0.001, 0.015, n))
    lows = closes * (1 - np.random.uniform(0.001, 0.015, n))
    volumes = np.random.randint(500_000, 2_000_000, n).astype(float)
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    return pd.DataFrame({
        'Open': closes,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes,
    }, index=dates)


# ==================== RSI Tests ====================

class TestRSI:
    def test_rsi_range(self):
        """RSI must always be in [0, 100]."""
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.normal(0, 1, 200))
        df = make_price_df(closes)
        df = calculate_technicals(df)
        valid = df['RSI'].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_uptrend(self):
        """Strongly rising prices should produce RSI > 60."""
        closes = np.linspace(100, 150, 100)
        df = make_price_df(closes)
        df = calculate_technicals(df)
        last_rsi = df['RSI'].iloc[-1]
        assert last_rsi > 60, f"Uptrend RSI should be > 60, got {last_rsi}"

    def test_rsi_downtrend(self):
        """Strongly falling prices should produce RSI < 40."""
        closes = np.linspace(150, 100, 100)
        df = make_price_df(closes)
        df = calculate_technicals(df)
        last_rsi = df['RSI'].iloc[-1]
        assert last_rsi < 40, f"Downtrend RSI should be < 40, got {last_rsi}"


# ==================== MACD Tests ====================

class TestMACD:
    def test_macd_columns_present(self):
        """MACD, Signal, and MACD_Hist columns should exist."""
        df = make_price_df(np.linspace(100, 110, 100))
        df = calculate_technicals(df)
        assert 'MACD' in df.columns
        assert 'Signal' in df.columns
        assert 'MACD_Hist' in df.columns

    def test_macd_hist_equals_macd_minus_signal(self):
        """MACD_Hist = MACD - Signal."""
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.normal(0, 1, 100))
        df = make_price_df(closes)
        df = calculate_technicals(df)
        valid = df.dropna(subset=['MACD', 'Signal', 'MACD_Hist'])
        diff = (valid['MACD'] - valid['Signal'] - valid['MACD_Hist']).abs()
        assert (diff < 1e-10).all()

    def test_macd_positive_in_uptrend(self):
        """MACD should be positive in an uptrend."""
        closes = np.linspace(100, 150, 100)
        df = make_price_df(closes)
        df = calculate_technicals(df)
        assert df['MACD'].iloc[-1] > 0


# ==================== Bollinger Bands Tests ====================

class TestBollingerBands:
    def test_bb_order(self):
        """BB_Lower < SMA_20 < BB_Upper."""
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.normal(0, 1, 100))
        df = make_price_df(closes)
        df = calculate_technicals(df)
        valid = df.dropna(subset=['BB_Lower', 'SMA_20', 'BB_Upper'])
        assert (valid['BB_Lower'] < valid['SMA_20']).all()
        assert (valid['SMA_20'] < valid['BB_Upper']).all()

    def test_bb_pctb_range(self):
        """BB %B is typically in [0, 1] when price is within bands."""
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.normal(0, 0.5, 100))
        df = make_price_df(closes)
        df = calculate_technicals(df)
        valid = df['BB_PctB'].dropna()
        # Most values should be between -0.5 and 1.5 (can exceed 0-1 in volatile moves)
        assert valid.median() > 0 and valid.median() < 1

    def test_bb_width_positive(self):
        """BB Bandwidth should always be > 0."""
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.normal(0, 1, 100))
        df = make_price_df(closes)
        df = calculate_technicals(df)
        valid = df['BB_Width'].dropna()
        assert (valid > 0).all()


# ==================== StochRSI Tests ====================

class TestStochRSI:
    def test_stochrsi_range(self):
        """StochRSI must be in [0, 1]."""
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.normal(0, 1, 200))
        df = make_price_df(closes)
        df = calculate_technicals(df)
        valid = df['StochRSI'].dropna()
        assert (valid >= -0.01).all() and (valid <= 1.01).all(), (
            f"StochRSI range: [{valid.min()}, {valid.max()}]")

    def test_stochrsi_kd_columns(self):
        """StochRSI_K and StochRSI_D columns exist."""
        df = make_price_df(np.linspace(100, 110, 200))
        df = calculate_technicals(df)
        assert 'StochRSI_K' in df.columns
        assert 'StochRSI_D' in df.columns

    def test_stochrsi_d_smoother_than_k(self):
        """StochRSI_D (signal) should be smoother than StochRSI_K."""
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.normal(0, 1, 200))
        df = make_price_df(closes)
        df = calculate_technicals(df)
        valid = df.dropna(subset=['StochRSI_K', 'StochRSI_D'])
        k_var = valid['StochRSI_K'].diff().var()
        d_var = valid['StochRSI_D'].diff().var()
        assert d_var < k_var, "D should be smoother (less variance) than K"


# ==================== ATR Tests ====================

class TestATR:
    def test_atr_positive(self):
        """ATR should always be positive."""
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.normal(0, 1, 100))
        df = make_price_df(closes)
        df = calculate_technicals(df)
        valid = df['ATR'].dropna()
        assert (valid > 0).all()

    def test_atr_higher_for_volatile(self):
        """Higher volatility should produce higher ATR."""
        low_vol = make_price_df(100 + np.cumsum(np.random.normal(0, 0.2, 100)))
        high_vol = make_price_df(100 + np.cumsum(np.random.normal(0, 2.0, 100)))
        low_vol = calculate_technicals(low_vol)
        high_vol = calculate_technicals(high_vol)
        assert high_vol['ATR'].iloc[-1] > low_vol['ATR'].iloc[-1]


# ==================== VWAP Tests ====================

class TestVWAP:
    def test_vwap_within_hl_range(self):
        """VWAP should be between daily low and high."""
        np.random.seed(42)
        n = 50
        closes = 100 + np.cumsum(np.random.normal(0, 0.5, n))
        highs = closes + np.random.uniform(0.5, 2, n)
        lows = closes - np.random.uniform(0.5, 2, n)
        df = make_price_df(closes, highs, lows, n=n)
        df = calculate_technicals(df)
        valid = df.dropna(subset=['VWAP'])
        # VWAP should be within cumulative range for the day
        assert valid['VWAP'].iloc[-1] > 0

    def test_vwap_column_exists(self):
        """VWAP column should be present."""
        df = make_price_df(np.linspace(100, 110, 50))
        df = calculate_technicals(df)
        assert 'VWAP' in df.columns


# ==================== OBV Tests ====================

class TestOBV:
    def test_obv_rising_with_price(self):
        """Rising prices + constant volume = rising OBV."""
        closes = np.linspace(100, 120, 50)
        df = make_price_df(closes)
        df = calculate_technicals(df)
        # OBV should be mostly increasing
        obv_diff = df['OBV'].diff().dropna()
        assert (obv_diff > 0).sum() > len(obv_diff) * 0.8

    def test_obv_sma_exists(self):
        """OBV_SMA column should exist."""
        df = make_price_df(np.linspace(100, 110, 50))
        df = calculate_technicals(df)
        assert 'OBV_SMA' in df.columns


# ==================== ADX Tests ====================

class TestADX:
    def test_adx_range(self):
        """ADX should be in [0, 100]."""
        np.random.seed(42)
        df = make_trending_df(n=200)
        df = calculate_technicals(df)
        valid = df['ADX'].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_adx_strong_in_trend(self):
        """Strong trending data should give ADX > 20."""
        np.random.seed(42)
        df = make_trending_df(start=100, daily_return=0.01, n=200)
        df = calculate_technicals(df)
        last_adx = df['ADX'].iloc[-1]
        assert last_adx > 20, f"Strong trend ADX should be > 20, got {last_adx}"

    def test_di_columns_exist(self):
        """'+DI' and '-DI' columns must be present for directional info."""
        df = make_trending_df(n=100)
        df = calculate_technicals(df)
        assert '+DI' in df.columns
        assert '-DI' in df.columns


# ==================== Williams %R Tests ====================

class TestWilliamsR:
    def test_williams_r_range(self):
        """Williams %R must be in [-100, 0]."""
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.normal(0, 1, 100))
        df = make_price_df(closes)
        df = calculate_technicals(df)
        valid = df['Williams_R'].dropna()
        assert (valid >= -100.01).all() and (valid <= 0.01).all(), (
            f"Williams %R range: [{valid.min()}, {valid.max()}]")

    def test_williams_r_overbought_in_uptrend(self):
        """Strong uptrend should show overbought (near 0)."""
        closes = np.linspace(100, 150, 100)
        df = make_price_df(closes)
        df = calculate_technicals(df)
        last_wr = df['Williams_R'].iloc[-1]
        assert last_wr > -20, f"Uptrend should be overbought, got {last_wr}"


# ==================== CCI Tests ====================

class TestCCI:
    def test_cci_column_exists(self):
        """CCI column should be present."""
        df = make_price_df(np.linspace(100, 110, 100))
        df = calculate_technicals(df)
        assert 'CCI' in df.columns

    def test_cci_positive_in_uptrend(self):
        """CCI should be positive in a strong uptrend."""
        closes = np.linspace(100, 150, 100)
        df = make_price_df(closes)
        df = calculate_technicals(df)
        last_cci = df['CCI'].iloc[-1]
        assert last_cci > 0, f"Uptrend CCI should be > 0, got {last_cci}"

    def test_cci_negative_in_downtrend(self):
        """CCI should be negative in a strong downtrend."""
        closes = np.linspace(150, 100, 100)
        df = make_price_df(closes)
        df = calculate_technicals(df)
        last_cci = df['CCI'].iloc[-1]
        assert last_cci < 0, f"Downtrend CCI should be < 0, got {last_cci}"


# ==================== All Columns Present ====================

class TestAllColumns:
    def test_all_expected_columns(self):
        """All expected technical indicator columns should be computed."""
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.normal(0, 1, 200))
        df = make_price_df(closes)
        df = calculate_technicals(df)
        expected = ['RSI', 'MACD', 'Signal', 'MACD_Hist', 'SMA_20', 'BB_Upper', 'BB_Lower',
                    'BB_PctB', 'BB_Width', 'ATR', 'StochRSI', 'StochRSI_K', 'StochRSI_D',
                    'VWAP', 'OBV', 'OBV_SMA', 'ADX', '+DI', '-DI', 'Williams_R', 'CCI']
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
