"""Unit tests for tradeable-edge / scan verdict helpers."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.options_scan import (
    quote_passes_liquidity,
    near_atm_strike,
    delta_in_band,
    tradeable_edge,
    half_spread,
    min_abs_edge,
    scan_verdict,
)


class TestLiquidity:
    def test_rejects_last_only(self):
        assert quote_passes_liquidity(0, 0, 100, 50) is False
        assert quote_passes_liquidity(1.0, 0, 100, 50) is False

    def test_rejects_crossed_or_wide(self):
        assert quote_passes_liquidity(2.0, 1.0, 100, 50) is False  # crossed
        # mid=1.0, spread 0.30 → 30% > 20%
        assert quote_passes_liquidity(0.85, 1.15, 100, 50) is False

    def test_rejects_junk_mid_and_no_interest(self):
        assert quote_passes_liquidity(0.01, 0.02, 100, 50) is False  # mid < 0.05
        assert quote_passes_liquidity(1.0, 1.1, 0, 0) is False  # no OI/vol
        assert quote_passes_liquidity(1.0, 1.1, 10, 0) is True
        assert quote_passes_liquidity(1.0, 1.1, 0, 5) is True

    def test_accepts_tight_quote(self):
        assert quote_passes_liquidity(1.00, 1.10, 50, 10) is True


class TestMoneyness:
    def test_strike_band(self):
        assert near_atm_strike(100, 100)
        assert near_atm_strike(110, 100)
        assert not near_atm_strike(113, 100)

    def test_delta_band(self):
        assert delta_in_band(0.45)
        assert delta_in_band(-0.30)
        assert not delta_in_band(0.10)
        assert not delta_in_band(0.80)


class TestTradeableEdge:
    def test_edge_signs(self):
        # fair 2.00, bid 1.40 ask 1.60 → long +0.40, short -0.60, mid 1.50
        el, es, mid = tradeable_edge(2.00, 1.40, 1.60)
        assert el == pytest.approx(0.40)
        assert es == pytest.approx(-0.60)
        assert mid == pytest.approx(1.50)
        assert half_spread(1.40, 1.60) == pytest.approx(0.10)

    def test_under_requires_beat_ask_and_pct(self):
        # mid=1.50, half_spread=0.10 → hurdle max(0.10, 0.15)=0.15
        # fair-ask must > 0.15 and >= 8% of mid (0.12)
        v, ev, pct = scan_verdict(1.80, 1.40, 1.60)  # edge_long=0.20
        assert v == "Under"
        assert ev == pytest.approx(0.20)
        assert pct == pytest.approx(0.20 / 1.50)

        # edge_long=0.12 — fails absolute hurdle 0.15
        v2, _, _ = scan_verdict(1.72, 1.40, 1.60)
        assert v2 == "Fair"

    def test_over_vs_bid(self):
        # fair cheap vs market: bid 2.00 ask 2.20, fair 1.50
        # edge_short = 2.00-1.50=0.50; mid=2.10; half=0.10; hurdle=0.15
        v, ev, pct = scan_verdict(1.50, 2.00, 2.20)
        assert v == "Over"
        assert ev == pytest.approx(1.50 - 2.20)  # EV@Ask still fair-ask
        assert pct == pytest.approx(0.50 / 2.10)

    def test_earnings_bumps_hurdle(self):
        # Without earnings: edge_long=0.16 > max(0.10, 0.15)=0.15 and pct=0.16/1.5≈0.107
        v, _, _ = scan_verdict(1.76, 1.40, 1.60, is_earnings=False)
        assert v == "Under"
        # With earnings: hurdle max(0.15, 0.20)=0.20 → 0.16 fails
        v2, _, _ = scan_verdict(1.76, 1.40, 1.60, is_earnings=True)
        assert v2 == "Fair"

    def test_min_abs_edge_formula(self):
        assert min_abs_edge(1.0, 1.2) == pytest.approx(max(0.10, 0.10 + 0.05))
        assert min_abs_edge(1.0, 1.2, is_earnings=True) == pytest.approx(
            max(0.15, 0.10 + 0.05 + 0.05)
        )
