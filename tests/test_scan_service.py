"""Scan service unit tests with mocked provider / no network."""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.scan_service import (
    analyze_ticker,
    earnings_contract_set,
    interpolate_rfr,
    normalize_div_yield,
    resolve_forecast_vol,
    scan_option_chains,
    OptionScanRow,
)
from core.pricing import VegaChimpCore
from datetime import date, timedelta


class TestHelpers:
    def test_normalize_div_yield(self):
        assert normalize_div_yield(None) is None
        assert normalize_div_yield(0.0294) == pytest.approx(0.0294)
        assert normalize_div_yield(2.94) == pytest.approx(0.0294)
        assert normalize_div_yield(-1) is None

    def test_resolve_forecast_vol_ewma_and_blend(self):
        assert resolve_forecast_vol(0.2, 0.4, False) == pytest.approx(0.2)
        assert resolve_forecast_vol(0.2, 0.4, True) == pytest.approx(0.3)
        assert resolve_forecast_vol(0.0, 0.0, False, hv_30=0.18) == pytest.approx(0.18)
        assert resolve_forecast_vol(0.0, 0.0, False, hv_30=0.0) == pytest.approx(0.25)

    def test_interpolate_rfr(self):
        assert interpolate_rfr(0.05, 0.05, 0.1) == pytest.approx(0.05)
        mid = interpolate_rfr(0.04, 0.06, 5.125)
        assert 0.04 < mid < 0.06

    def test_earnings_contract_set(self):
        exps = ["2026-09-12", "2026-09-19", "2026-10-17"]
        got = earnings_contract_set([date(2026, 9, 15)], exps)
        assert got == {"2026-09-19"}


def _fake_chain(spot=100.0):
    """Build a liquid near-ATM call/put chain with room for Under edge."""
    strikes = [95.0, 100.0, 105.0]
    # Wide mispricing: ask well below fair under high forecast vol
    calls = pd.DataFrame({
        "strike": strikes,
        "bid": [8.0, 5.0, 3.0],
        "ask": [8.2, 5.2, 3.2],
        "volume": [100, 200, 80],
        "openInterest": [500, 800, 400],
        "impliedVolatility": [0.25, 0.25, 0.25],
    })
    puts = pd.DataFrame({
        "strike": strikes,
        "bid": [3.0, 5.0, 8.0],
        "ask": [3.2, 5.2, 8.2],
        "volume": [90, 180, 70],
        "openInterest": [400, 700, 350],
        "impliedVolatility": [0.25, 0.25, 0.25],
    })
    return SimpleNamespace(calls=calls, puts=puts)


class FakeProvider:
    def __init__(self, spot=100.0):
        self.spot = spot
        self.chains = {}

    def create_ticker(self, symbol):
        return SimpleNamespace(ticker=symbol, symbol=symbol)

    def fetch_history(self, ticker, period, interval, retries=2, delay=1.0, log=None):
        n = 260
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.01, n)
        close = 100 * np.exp(np.cumsum(rets))
        idx = pd.bdate_range("2025-01-02", periods=n)
        return pd.DataFrame({
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(n, 1_000_000),
        }, index=idx)

    def get_fast_last_price(self, ticker):
        return self.spot

    def get_info(self, ticker):
        return {"dividendYield": 0.01}

    def get_fast_info(self, ticker):
        return {"dividend_yield": 0.01, "last_price": self.spot}

    def get_option_expirations(self, ticker):
        d0 = date.today() + timedelta(days=30)
        # snap to Friday-ish string
        return (d0.strftime("%Y-%m-%d"),)

    def get_option_chain(self, ticker, expiration):
        return _fake_chain(self.spot)

    def fetch_rate_curve(self):
        return 0.045, 0.04

    def get_calendar(self, ticker):
        return {}


class TestAnalyzeTicker:
    def test_analyze_mocked(self):
        provider = FakeProvider(spot=101.5)
        analysis = analyze_ticker(provider, "TEST")
        assert analysis.ticker == "TEST"
        assert analysis.spot == pytest.approx(101.5)
        assert analysis.ewma_vol > 0
        assert analysis.hv_30 > 0
        assert analysis.summary_lines
        assert "RSI" in analysis.technicals


class TestScanOptionChains:
    def test_scan_returns_rows_and_under_filter(self):
        provider = FakeProvider(spot=100.0)
        stock = provider.create_ticker("TEST")
        exp = provider.get_option_expirations(stock)[0]
        # High forecast vol → fair >> ask for ATM options → Under
        result = scan_option_chains(
            data_provider=provider,
            stock=stock,
            spot=100.0,
            dates=[exp],
            all_exps=[exp],
            ewma_vol=0.55,
            garch_vol=0.0,
            use_garch_blend=False,
            under_only=False,
            dividend_yield=0.01,
            short_rate=0.045,
            long_rate=0.04,
            use_american_greeks=True,
        )
        assert result.forecast_vol == pytest.approx(0.55)
        assert len(result.rows) >= 1
        assert all(isinstance(r, OptionScanRow) for r in result.rows)
        # under-only should be subset
        under = scan_option_chains(
            data_provider=provider,
            stock=stock,
            spot=100.0,
            dates=[exp],
            all_exps=[exp],
            ewma_vol=0.55,
            under_only=True,
            dividend_yield=0.01,
            short_rate=0.045,
            long_rate=0.04,
        )
        assert all("Under" in r.verdict for r in under.rows)
        if len(under.rows) >= 2:
            edges = [r.edge_pct for r in under.rows]
            assert edges == sorted(edges, reverse=True)

    def test_option_type_filter_call_only(self):
        provider = FakeProvider()
        stock = provider.create_ticker("TEST")
        exp = provider.get_option_expirations(stock)[0]
        result = scan_option_chains(
            data_provider=provider,
            stock=stock,
            spot=100.0,
            dates=[exp],
            ewma_vol=0.55,
            option_type="call",
            dividend_yield=0.0,
            short_rate=0.045,
            long_rate=0.04,
        )
        assert all(r.type == "CALL" for r in result.rows)

    def test_ui_batch_callback(self):
        provider = FakeProvider()
        stock = provider.create_ticker("TEST")
        exp = provider.get_option_expirations(stock)[0]
        seen = []

        def on_batch(items):
            seen.extend(items)

        scan_option_chains(
            data_provider=provider,
            stock=stock,
            spot=100.0,
            dates=[exp],
            ewma_vol=0.55,
            dividend_yield=0.0,
            short_rate=0.045,
            long_rate=0.04,
            on_ui_batch=on_batch,
        )
        assert seen  # at least one flush
        vals, tag = seen[0]
        assert len(vals) == 17

    def test_tree_vals_format(self):
        row = OptionScanRow(
            date="2026-10-17", type="CALL", strike=100.0, volume=10, oi=50,
            mid=5.0, spread_pct=4.0, breakeven=105.0, iv=0.25, fair=5.5,
            ev_at_ask=0.3, edge_pct=0.1, delta=0.5, gamma=0.01, theta=-0.05,
            vega=0.2, pop=55.0, verdict="Under", tag="green",
        )
        vals = row.tree_vals()
        assert vals[0] == "2026-10-17"
        assert vals[2] == "100.00"
        assert vals[10] == "+0.30"
        assert vals[16] == "Under"
