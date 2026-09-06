"""Tests for watchlist, Ichimoku, Fib swing anchors, and daily EMA mapping."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.technicals import (
    attach_daily_emas,
    bars_per_trading_day,
    calculate_ichimoku,
    compute_daily_emas,
    compute_ema_series,
    fib_retracement_levels,
    fib_swing_anchors,
    find_pivot_indices,
    map_daily_columns_to_bars,
)
from core.pricing import VegaChimpCore
from ui.watchlist import (
    DEFAULT_WATCHLIST,
    add_ticker,
    load_watchlist,
    remove_ticker,
    save_watchlist,
)
from ui.prefs import DEFAULT_PREFS, load_prefs, save_prefs
from ui.chart import draw_probability_cone, prepare_plot_frame
from matplotlib.figure import Figure


# -------------------- Watchlist --------------------

class TestWatchlist:
    def test_default_when_missing(self, tmp_path):
        assert load_watchlist(str(tmp_path)) == DEFAULT_WATCHLIST

    def test_round_trip_add_remove(self, tmp_path):
        saved = save_watchlist(str(tmp_path), ["amd", "aapl", "AMD", "bad symbol!"])
        assert saved == ["AMD", "AAPL"]
        assert load_watchlist(str(tmp_path)) == ["AMD", "AAPL"]
        assert add_ticker(str(tmp_path), "msft") == ["AMD", "AAPL", "MSFT"]
        assert remove_ticker(str(tmp_path), "AAPL") == ["AMD", "MSFT"]

    def test_prefs_new_overlay_defaults(self, tmp_path):
        data = load_prefs(str(tmp_path))
        assert data["show_ichimoku"] is False
        assert data["show_earnings"] is False
        save_prefs(str(tmp_path), show_ichimoku=True, show_earnings=True, show_fib=True)
        data = load_prefs(str(tmp_path))
        assert data["show_ichimoku"] is True
        assert data["show_earnings"] is True
        assert data["show_fib"] is True
        assert set(DEFAULT_PREFS) <= set(data)


# -------------------- EMA daily mapping --------------------

class TestDailyEMA:
    def test_ema_span_matches_pandas(self):
        close = pd.Series(np.linspace(100, 120, 50), dtype=float)
        got = compute_ema_series(close, 21)
        exp = close.ewm(span=21, adjust=False).mean()
        np.testing.assert_allclose(got.values, exp.values)

    def test_intraday_bars_do_not_use_bar_span_200(self):
        """Regression: EMA_200 on intraday bars must equal daily EMA mapped as-of, not 200 bars."""
        rng = np.random.default_rng(0)
        bars_per_day = 78
        n_days = 30
        daily_close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
        daily_idx = pd.bdate_range("2024-01-01", periods=n_days)
        daily = pd.Series(daily_close, index=daily_idx)

        # RTH-ish 5m bars in America/New_York (09:30 + 5m*k)
        bar_times = []
        bar_closes = []
        for i, d in enumerate(daily_idx):
            for m in range(bars_per_day):
                ts = pd.Timestamp(d, tz="America/New_York") + pd.Timedelta(
                    hours=9, minutes=30 + 5 * m
                )
                bar_times.append(ts)
                bar_closes.append(daily_close[i] * (1 + 0.0001 * math.sin(m)))
        bar_df = pd.DataFrame({"Close": bar_closes}, index=pd.DatetimeIndex(bar_times))

        wrong = bar_df["Close"].ewm(span=200, adjust=False).mean()
        attach_daily_emas(bar_df, daily)
        right = bar_df["EMA_200"]

        daily_ema200 = compute_ema_series(daily, 200)
        # Post-close bar on last day sees that day's completed daily EMA
        after_close = pd.Timestamp(daily_idx[-1], tz="America/New_York") + pd.Timedelta(
            hours=16, minutes=30
        )
        # append a synthetic after-close print for the assert
        bar_df2 = bar_df.copy()
        bar_df2.loc[after_close, "Close"] = daily_close[-1]
        attach_daily_emas(bar_df2, daily)
        assert bar_df2["EMA_200"].loc[after_close] == pytest.approx(
            float(daily_ema200.iloc[-1]), rel=1e-9
        )
        # bar-span EMA_200 differs materially on this sample
        assert abs(float(wrong.iloc[-1]) - float(right.iloc[-1])) > 0.01

    def test_intraday_morning_no_same_day_ema_lookahead(self):
        """Morning bars must not see the same session's daily EMA (completed-close rule)."""
        daily_idx = pd.DatetimeIndex(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
        )
        daily = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0], index=daily_idx)
        emas = compute_daily_emas(daily, spans=(5,))
        morning = pd.DatetimeIndex(
            [pd.Timestamp("2024-01-08 10:00:00", tz="America/New_York")]
        )
        after = pd.DatetimeIndex(
            [pd.Timestamp("2024-01-08 16:30:00", tz="America/New_York")]
        )
        mapped_am = map_daily_columns_to_bars(morning, emas)
        mapped_pm = map_daily_columns_to_bars(after, emas)
        assert float(mapped_am["EMA_5"].iloc[0]) == pytest.approx(
            float(emas["EMA_5"].iloc[-2]), rel=1e-12
        )
        assert float(mapped_pm["EMA_5"].iloc[0]) == pytest.approx(
            float(emas["EMA_5"].iloc[-1]), rel=1e-12
        )

    def test_daily_chart_ema_includes_same_day_close(self):
        """Daily→daily as-of must still include that day's close in EMA_*."""
        daily_idx = pd.bdate_range("2024-01-02", periods=10)
        daily = pd.Series(np.linspace(100, 120, 10), index=daily_idx)
        emas = compute_daily_emas(daily, spans=(5,))
        mapped = map_daily_columns_to_bars(daily_idx, emas)
        assert float(mapped["EMA_5"].iloc[-1]) == pytest.approx(
            float(emas["EMA_5"].iloc[-1]), rel=1e-12
        )

    def test_map_preserves_bar_order(self):
        daily = pd.DataFrame(
            {"EMA_5": [1.0, 2.0, 3.0]},
            index=pd.bdate_range("2024-01-01", periods=3),
        )
        # bars intentionally not chronological in construction order
        idx = pd.DatetimeIndex([
            pd.Timestamp("2024-01-03 10:00", tz="America/New_York"),
            pd.Timestamp("2024-01-01 10:00", tz="America/New_York"),
            pd.Timestamp("2024-01-02 15:00", tz="America/New_York"),
        ])
        mapped = map_daily_columns_to_bars(idx, daily)
        assert list(mapped.index) == list(idx)
        # Morning / before RTH close → prior completed daily only
        assert mapped["EMA_5"].iloc[0] == pytest.approx(2.0)  # Jan 3 10:00 → Jan 2
        assert pd.isna(mapped["EMA_5"].iloc[1])  # Jan 1 10:00 → nothing prior completed
        assert mapped["EMA_5"].iloc[2] == pytest.approx(1.0)  # Jan 2 15:00 → Jan 1


# -------------------- EWMA vol (math OK) --------------------

class TestEWMAVolNotBroken:
    def test_closed_form_matches_sequential(self):
        rng = np.random.default_rng(1)
        rets = rng.normal(0, 0.015, 300)
        closed = VegaChimpCore.ewma_vol_forecast(rets)
        lam = 0.94
        r2 = rets * rets
        v = float(np.var(rets))
        for x in r2:
            v = lam * v + (1 - lam) * x
        sequential = math.sqrt(v * 252)
        assert closed == pytest.approx(sequential, rel=1e-12)

    def test_ewma_is_on_returns_not_prices(self):
        prices = np.linspace(100, 150, 100)
        # treating prices as "returns" would give nonsense huge vol
        bad = VegaChimpCore.ewma_vol_forecast(prices)
        log_rets = np.diff(np.log(prices))
        good = VegaChimpCore.ewma_vol_forecast(log_rets)
        assert good < 1.0  # ~reasonable annualized for smooth trend
        assert bad > good

    def test_cone_scales_with_bars_per_day(self):
        fig = Figure()
        ax = fig.add_subplot(111)
        e1 = draw_probability_cone(ax, last_x=10.0, p0=100.0, sigma=0.2, horizon_days=30, bars_per_day=1.0)
        fig2 = Figure()
        ax2 = fig2.add_subplot(111)
        e2 = draw_probability_cone(ax2, last_x=10.0, p0=100.0, sigma=0.2, horizon_days=30, bars_per_day=78.0)
        assert e1 is not None and e2 is not None
        assert e1[0] == pytest.approx(40.0)
        assert e2[0] == pytest.approx(10.0 + 30 * 78.0)
        assert bars_per_trading_day("5m") == 78.0
        assert bars_per_trading_day("1d") == 1.0


# -------------------- Ichimoku --------------------

class TestIchimoku:
    def test_formulas_on_flat_then_ramp(self):
        n = 80
        closes = np.concatenate([np.full(40, 100.0), np.linspace(100, 140, 40)])
        highs = closes + 1.0
        lows = closes - 1.0
        idx = pd.bdate_range("2024-01-01", periods=n)
        df = pd.DataFrame(
            {"High": highs, "Low": lows, "Close": closes, "Open": closes, "Volume": 1e6},
            index=idx,
        )
        calculate_ichimoku(df)
        # Tenkan at bar 20: midpoint of last 9 highs/lows
        i = 20
        exp_tenkan = (highs[i - 8 : i + 1].max() + lows[i - 8 : i + 1].min()) / 2.0
        assert df["Ichimoku_Tenkan"].iloc[i] == pytest.approx(exp_tenkan)
        # Kijun needs 26 bars — use index 30
        i = 30
        exp_kijun = (highs[i - 25 : i + 1].max() + lows[i - 25 : i + 1].min()) / 2.0
        assert df["Ichimoku_Kijun"].iloc[i] == pytest.approx(exp_kijun)
        # Senkou A at j equals (tenkan+kijun)/2 computed at j-26, then shifted forward
        j = 55
        src = j - 26  # 29 — needs >=25 for kijun window
        tenkan_src = (highs[src - 8 : src + 1].max() + lows[src - 8 : src + 1].min()) / 2.0
        kijun_src = (highs[src - 25 : src + 1].max() + lows[src - 25 : src + 1].min()) / 2.0
        exp_sa = (tenkan_src + kijun_src) / 2.0
        assert df["Ichimoku_Senkou_A"].iloc[j] == pytest.approx(exp_sa)
        # Chikou at i is close[i+26]
        assert df["Ichimoku_Chikou"].iloc[10] == pytest.approx(closes[36])
        # Senkou B period 52 midpoint, shifted +26 (src must be >= 51)
        j = 77
        src = j - 26  # 51
        exp_sb = (highs[src - 51 : src + 1].max() + lows[src - 51 : src + 1].min()) / 2.0
        assert df["Ichimoku_Senkou_B"].iloc[j] == pytest.approx(exp_sb)

    def test_ichimoku_windows_use_rth_not_extended_hours(self):
        """Premarket spikes must not enter Tenkan/Kijun once RTH-filtered."""
        from ui.chart import prepare_plot_frame

        times = []
        for d in pd.bdate_range("2024-01-02", periods=5):
            for hour in (4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15):
                times.append(pd.Timestamp(d, tz="America/New_York") + pd.Timedelta(hours=hour))
        idx = pd.DatetimeIndex(times)
        n = len(idx)
        close = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        for i, ts in enumerate(idx):
            if ts.hour < 9:
                high[i], low[i], close[i] = 200.0, 50.0, 125.0
        df = pd.DataFrame(
            {"Open": close, "High": high, "Low": low, "Close": close, "Volume": 1e6},
            index=idx,
        )
        # Wrong path: compute on eth+rth then filter (historical bug)
        polluted = df.copy()
        calculate_ichimoku(polluted)
        plot_polluted, _, _ = prepare_plot_frame(polluted, "60m")
        # Right path: filter first, then Ichimoku
        plot_clean, _, _ = prepare_plot_frame(df.copy(), "60m")
        calculate_ichimoku(plot_clean)
        assert float(plot_clean["Ichimoku_Tenkan"].iloc[-1]) == pytest.approx(100.0)
        assert float(plot_polluted["Ichimoku_Tenkan"].iloc[-1]) != pytest.approx(100.0)


# -------------------- Fib swing --------------------

class TestFibSwing:
    def test_latest_swing_not_period_extreme(self):
        # Construct: early huge high 200, then later swing high 120 / low 100
        n = 80
        close = np.full(n, 110.0)
        high = np.full(n, 111.0)
        low = np.full(n, 109.0)
        # early spike high (period max) — should NOT anchor if later swings exist
        high[5] = 200.0
        low[5] = 108.0
        # confirmed swing low around 40
        for k in range(35, 46):
            low[k] = 100.0 + abs(k - 40) * 0.5
            high[k] = 105.0 + abs(k - 40) * 0.3
        low[40] = 100.0
        # confirmed swing high around 60
        for k in range(55, 66):
            high[k] = 120.0 - abs(k - 60) * 0.5
            low[k] = 110.0 - abs(k - 60) * 0.3
        high[60] = 120.0

        anchors = fib_swing_anchors(high, low, pivot_left=3, pivot_right=3)
        assert anchors["ok"]
        # Must not use the early 200 spike as fib_high when later swings dominate
        assert anchors["fib_high"] < 150.0
        assert anchors["rule"] != "fallback_window" or anchors["fib_high"] <= 120.0 + 1e-9
        levels = fib_retracement_levels(anchors["fib_high"], anchors["fib_low"])
        assert "61.8%" in levels
        # level between high and low
        assert anchors["fib_low"] < levels["50.0%"] < anchors["fib_high"]

    def test_fallback_when_too_short(self):
        high = np.array([10.0, 12.0, 11.0])
        low = np.array([9.0, 8.0, 8.5])
        anchors = fib_swing_anchors(high, low, pivot_left=5, pivot_right=5)
        assert anchors["ok"]
        assert anchors["rule"] == "fallback_window"
        assert anchors["fib_high"] == pytest.approx(12.0)
        assert anchors["fib_low"] == pytest.approx(8.0)

    def test_pivot_finder_basic(self):
        high = np.array([1, 2, 5, 2, 1, 2, 6, 2, 1], dtype=float)
        low = np.array([1, 1, 1, 1, 0.5, 1, 1, 1, 1], dtype=float)
        sh, sl = find_pivot_indices(high, low, left=2, right=2)
        assert 2 in sh and 6 in sh
        assert 4 in sl
