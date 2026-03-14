"""
Option Pricing Tests — analytical benchmarks for VegaChimpCore.
Tests Black-Scholes pricing, Greeks, implied vol solver, EWMA, and Bjerksund-Stensland.
"""
import sys
import os
import math
import numpy as np
import pytest

# Add parent directory so we can import sentinel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sentinel import VegaChimpCore


# ==================== Black-Scholes Known Values ====================

class TestBSPricing:
    """Test European Black-Scholes pricing against known analytical values."""

    def test_bs_call_known_value(self):
        """S=100, K=100, r=0.05, q=0, sig=0.2, T=1 -> C≈10.4506"""
        price = VegaChimpCore.bs_price(100, 100, 0.05, 0.0, 0.20, 1.0, "call")
        assert abs(price - 10.4506) < 0.01, f"Expected ~10.4506, got {price}"

    def test_bs_put_known_value(self):
        """S=100, K=100, r=0.05, q=0, sig=0.2, T=1 -> P≈5.5735"""
        price = VegaChimpCore.bs_price(100, 100, 0.05, 0.0, 0.20, 1.0, "put")
        assert abs(price - 5.5735) < 0.01, f"Expected ~5.5735, got {price}"

    def test_bs_call_put_parity(self):
        """C - P = S*e^(-qT) - K*e^(-rT) for any set of params."""
        S, K, r, q, sig, T = 110, 100, 0.05, 0.02, 0.25, 0.5
        C = VegaChimpCore.bs_price(S, K, r, q, sig, T, "call")
        P = VegaChimpCore.bs_price(S, K, r, q, sig, T, "put")
        parity = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert abs((C - P) - parity) < 1e-6, f"Parity violated: C-P={C-P}, expected {parity}"

    def test_bs_call_put_parity_high_div(self):
        """Parity with high dividend yield."""
        S, K, r, q, sig, T = 50, 55, 0.08, 0.06, 0.35, 2.0
        C = VegaChimpCore.bs_price(S, K, r, q, sig, T, "call")
        P = VegaChimpCore.bs_price(S, K, r, q, sig, T, "put")
        parity = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert abs((C - P) - parity) < 1e-6

    def test_bs_deep_itm_call(self):
        """Deep ITM call ≈ S*e^(-qT) - K*e^(-rT)."""
        S, K, r, q, sig, T = 200, 50, 0.05, 0.0, 0.20, 1.0
        price = VegaChimpCore.bs_price(S, K, r, q, sig, T, "call")
        intrinsic = S - K * math.exp(-r * T)
        assert abs(price - intrinsic) < 0.50, f"Deep ITM call should ≈ {intrinsic}, got {price}"

    def test_bs_deep_otm_call(self):
        """Deep OTM call ≈ 0."""
        price = VegaChimpCore.bs_price(50, 200, 0.05, 0.0, 0.20, 0.25, "call")
        assert price < 0.01, f"Deep OTM call should ≈ 0, got {price}"

    def test_bs_deep_otm_put(self):
        """Deep OTM put ≈ 0."""
        price = VegaChimpCore.bs_price(200, 50, 0.05, 0.0, 0.20, 0.25, "put")
        assert price < 0.01, f"Deep OTM put should ≈ 0, got {price}"

    def test_bs_zero_time(self):
        """At expiry, price = intrinsic value."""
        call = VegaChimpCore.bs_price(105, 100, 0.05, 0.0, 0.20, 0.0, "call")
        assert abs(call - 5.0) < 1e-6
        put = VegaChimpCore.bs_price(95, 100, 0.05, 0.0, 0.20, 0.0, "put")
        assert abs(put - 5.0) < 1e-6

    def test_bs_zero_vol(self):
        """Zero vol: call = max(0, S*e^(-qT) - K*e^(-rT))."""
        call = VegaChimpCore.bs_price(105, 100, 0.05, 0.0, 0.0, 1.0, "call")
        assert abs(call - 5.0) < 1e-6  # Intrinsic

    def test_bs_atm_call_positive(self):
        """ATM call always has positive time value."""
        price = VegaChimpCore.bs_price(100, 100, 0.05, 0.0, 0.30, 0.5, "call")
        assert price > 0


# ==================== Greeks Tests ====================

class TestGreeks:
    """Test Black-Scholes Greeks for correctness."""

    def test_call_delta_range(self):
        """Call delta is between 0 and 1."""
        for S in [80, 100, 120]:
            greeks = VegaChimpCore.bs_greeks(S, 100, 0.05, 0.0, 0.25, 1.0, "call")
            assert 0 <= greeks['delta'] <= 1, f"Call delta out of range: {greeks['delta']}"

    def test_put_delta_range(self):
        """Put delta is between -1 and 0."""
        for S in [80, 100, 120]:
            greeks = VegaChimpCore.bs_greeks(S, 100, 0.05, 0.0, 0.25, 1.0, "put")
            assert -1 <= greeks['delta'] <= 0, f"Put delta out of range: {greeks['delta']}"

    def test_call_put_delta_relation(self):
        """Call delta - Put delta = e^(-qT) (for same S, K, etc.)."""
        S, K, r, q, sig, T = 100, 100, 0.05, 0.02, 0.25, 1.0
        call_g = VegaChimpCore.bs_greeks(S, K, r, q, sig, T, "call")
        put_g = VegaChimpCore.bs_greeks(S, K, r, q, sig, T, "put")
        expected = math.exp(-q * T)
        assert abs((call_g['delta'] - put_g['delta']) - expected) < 1e-6

    def test_gamma_positive(self):
        """Gamma is always positive for both calls and puts."""
        for kind in ["call", "put"]:
            greeks = VegaChimpCore.bs_greeks(100, 100, 0.05, 0.0, 0.25, 1.0, kind)
            assert greeks['gamma'] > 0, f"Gamma should be positive for {kind}"

    def test_gamma_same_for_call_put(self):
        """Gamma is the same for calls and puts at the same strike."""
        call_g = VegaChimpCore.bs_greeks(100, 100, 0.05, 0.0, 0.25, 1.0, "call")
        put_g = VegaChimpCore.bs_greeks(100, 100, 0.05, 0.0, 0.25, 1.0, "put")
        assert abs(call_g['gamma'] - put_g['gamma']) < 1e-10

    def test_vega_positive(self):
        """Vega is always positive."""
        for kind in ["call", "put"]:
            greeks = VegaChimpCore.bs_greeks(100, 100, 0.05, 0.0, 0.25, 1.0, kind)
            assert greeks['vega'] > 0

    def test_vega_same_for_call_put(self):
        """Vega is the same for calls and puts."""
        call_g = VegaChimpCore.bs_greeks(100, 100, 0.05, 0.0, 0.25, 1.0, "call")
        put_g = VegaChimpCore.bs_greeks(100, 100, 0.05, 0.0, 0.25, 1.0, "put")
        assert abs(call_g['vega'] - put_g['vega']) < 1e-10

    def test_theta_negative_for_long(self):
        """Theta is typically negative (time decay hurts long positions)."""
        for kind in ["call", "put"]:
            greeks = VegaChimpCore.bs_greeks(100, 100, 0.05, 0.0, 0.25, 1.0, kind)
            assert greeks['theta'] < 0, f"Theta should be negative for ATM {kind}"

    def test_deep_itm_call_delta_near_one(self):
        """Deep ITM call has delta near 1."""
        greeks = VegaChimpCore.bs_greeks(200, 100, 0.05, 0.0, 0.20, 1.0, "call")
        assert greeks['delta'] > 0.95

    def test_deep_otm_call_delta_near_zero(self):
        """Deep OTM call has delta near 0."""
        greeks = VegaChimpCore.bs_greeks(50, 100, 0.05, 0.0, 0.20, 1.0, "call")
        assert greeks['delta'] < 0.05

    def test_numerical_delta(self):
        """Delta ≈ (P(S+h) - P(S-h)) / (2h) for small h."""
        S, K, r, q, sig, T = 100, 100, 0.05, 0.0, 0.25, 1.0
        h = 0.01
        for kind in ["call", "put"]:
            p_up = VegaChimpCore.bs_price(S + h, K, r, q, sig, T, kind)
            p_dn = VegaChimpCore.bs_price(S - h, K, r, q, sig, T, kind)
            num_delta = (p_up - p_dn) / (2 * h)
            greeks = VegaChimpCore.bs_greeks(S, K, r, q, sig, T, kind)
            assert abs(greeks['delta'] - num_delta) < 0.001, (
                f"Analytical delta {greeks['delta']:.6f} != numerical {num_delta:.6f} for {kind}")

    def test_numerical_gamma(self):
        """Gamma ≈ (delta(S+h) - delta(S-h)) / (2h) for small h."""
        S, K, r, q, sig, T = 100, 100, 0.05, 0.0, 0.25, 1.0
        h = 0.01
        g_up = VegaChimpCore.bs_greeks(S + h, K, r, q, sig, T, "call")
        g_dn = VegaChimpCore.bs_greeks(S - h, K, r, q, sig, T, "call")
        num_gamma = (g_up['delta'] - g_dn['delta']) / (2 * h)
        greeks = VegaChimpCore.bs_greeks(S, K, r, q, sig, T, "call")
        assert abs(greeks['gamma'] - num_gamma) < 0.001

    def test_numerical_vega(self):
        """Vega ≈ (P(sig+h) - P(sig-h)) / (2h) / 100 for small h."""
        S, K, r, q, sig, T = 100, 100, 0.05, 0.0, 0.25, 1.0
        h = 0.001
        p_up = VegaChimpCore.bs_price(S, K, r, q, sig + h, T, "call")
        p_dn = VegaChimpCore.bs_price(S, K, r, q, sig - h, T, "call")
        num_vega = (p_up - p_dn) / (2 * h) / 100  # Per 1% move
        greeks = VegaChimpCore.bs_greeks(S, K, r, q, sig, T, "call")
        assert abs(greeks['vega'] - num_vega) < 0.01

    def test_edge_case_zero_time(self):
        """Greeks at T=0."""
        greeks = VegaChimpCore.bs_greeks(105, 100, 0.05, 0.0, 0.25, 0.0, "call")
        assert greeks['delta'] == 1.0  # ITM
        assert greeks['gamma'] == 0.0


# ==================== Implied Volatility Solver ====================

class TestImpliedVol:
    """Test Newton-Raphson IV solver."""

    def test_iv_roundtrip_call(self):
        """price -> IV -> price should roundtrip for calls."""
        S, K, r, q, T = 100, 100, 0.05, 0.0, 1.0
        true_sig = 0.25
        price = VegaChimpCore.bs_price(S, K, r, q, true_sig, T, "call")
        recovered_sig = VegaChimpCore.implied_vol(price, S, K, r, q, T, "call")
        assert abs(recovered_sig - true_sig) < 0.001, (
            f"IV roundtrip failed: true={true_sig}, recovered={recovered_sig}")

    def test_iv_roundtrip_put(self):
        """price -> IV -> price should roundtrip for puts."""
        S, K, r, q, T = 100, 100, 0.05, 0.0, 1.0
        true_sig = 0.30
        price = VegaChimpCore.bs_price(S, K, r, q, true_sig, T, "put")
        recovered_sig = VegaChimpCore.implied_vol(price, S, K, r, q, T, "put")
        assert abs(recovered_sig - true_sig) < 0.001

    def test_iv_roundtrip_otm(self):
        """IV roundtrip on OTM option."""
        S, K, r, q, T = 100, 120, 0.05, 0.0, 0.5
        true_sig = 0.40
        price = VegaChimpCore.bs_price(S, K, r, q, true_sig, T, "call")
        recovered_sig = VegaChimpCore.implied_vol(price, S, K, r, q, T, "call")
        assert abs(recovered_sig - true_sig) < 0.005

    def test_iv_roundtrip_itm(self):
        """IV roundtrip on ITM option."""
        S, K, r, q, T = 120, 100, 0.05, 0.0, 0.5
        true_sig = 0.20
        price = VegaChimpCore.bs_price(S, K, r, q, true_sig, T, "call")
        recovered_sig = VegaChimpCore.implied_vol(price, S, K, r, q, T, "call")
        assert abs(recovered_sig - true_sig) < 0.005

    def test_iv_zero_price(self):
        """IV for zero-priced option returns 0."""
        result = VegaChimpCore.implied_vol(0, 100, 100, 0.05, 0.0, 1.0, "call")
        assert result == 0.0

    def test_iv_positive_output(self):
        """IV solver always returns positive value."""
        result = VegaChimpCore.implied_vol(5.0, 100, 100, 0.05, 0.0, 1.0, "call")
        assert result > 0


# ==================== American Option Pricing ====================

class TestAmericanPricing:
    """Test Bjerksund-Stensland American option pricing."""

    def test_american_call_no_div_equals_european(self):
        """American call = European call when q=0 (no early exercise benefit)."""
        S, K, r, q, sig, T = 100, 100, 0.05, 0.0, 0.25, 1.0
        american = VegaChimpCore.bjerksund_stensland(S, K, T, r, q, sig, 'call')
        european = VegaChimpCore.bs_price(S, K, r, q, sig, T, "call")
        assert abs(american - european) < 0.01, (
            f"American call ({american:.4f}) should equal European ({european:.4f}) with q=0")

    def test_american_put_geq_european(self):
        """American put >= European put (early exercise premium)."""
        S, K, r, q, sig, T = 100, 100, 0.05, 0.02, 0.25, 1.0
        american = VegaChimpCore.bjerksund_stensland(S, K, T, r, q, sig, 'put')
        european = VegaChimpCore.bs_price(S, K, r, q, sig, T, "put")
        assert american >= european - 0.01, (
            f"American put ({american:.4f}) should be >= European put ({european:.4f})")

    def test_american_geq_intrinsic_call(self):
        """American call >= intrinsic value."""
        S, K, r, q, sig, T = 120, 100, 0.05, 0.03, 0.25, 1.0
        american = VegaChimpCore.bjerksund_stensland(S, K, T, r, q, sig, 'call')
        intrinsic = max(S - K, 0)
        assert american >= intrinsic - 0.01

    def test_american_geq_intrinsic_put(self):
        """American put >= intrinsic value."""
        S, K, r, q, sig, T = 80, 100, 0.05, 0.03, 0.25, 1.0
        american = VegaChimpCore.bjerksund_stensland(S, K, T, r, q, sig, 'put')
        intrinsic = max(K - S, 0)
        assert american >= intrinsic - 0.01

    def test_american_zero_inputs(self):
        """Edge cases: zero S, K, or T return 0."""
        assert VegaChimpCore.bjerksund_stensland(0, 100, 1, 0.05, 0, 0.2, 'call') == 0.0
        assert VegaChimpCore.bjerksund_stensland(100, 0, 1, 0.05, 0, 0.2, 'call') == 0.0
        assert VegaChimpCore.bjerksund_stensland(100, 100, 0, 0.05, 0, 0.2, 'call') == 0.0

    def test_american_put_deep_itm(self):
        """Deep ITM American put ≈ intrinsic value."""
        S, K = 50, 150
        american = VegaChimpCore.bjerksund_stensland(S, K, 1.0, 0.05, 0.0, 0.20, 'put')
        assert american >= K - S - 1, f"Deep ITM put should be near {K-S}, got {american}"

    def test_american_positive_for_reasonable_inputs(self):
        """American option prices are positive for reasonable inputs."""
        for kind in ['call', 'put']:
            price = VegaChimpCore.bjerksund_stensland(100, 100, 1.0, 0.05, 0.02, 0.25, kind)
            assert price > 0, f"American {kind} price should be positive, got {price}"


# ==================== EWMA Vol Forecast ====================

class TestEWMAVol:
    """Test EWMA volatility forecast."""

    def test_ewma_positive_output(self):
        """EWMA forecast should always be > 0 for sufficient data."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 100)
        vol = VegaChimpCore.ewma_vol_forecast(returns)
        assert vol > 0, f"EWMA vol should be positive, got {vol}"

    def test_ewma_short_data_returns_zero(self):
        """EWMA with < 30 data points returns 0."""
        returns = np.random.normal(0, 0.01, 10)
        vol = VegaChimpCore.ewma_vol_forecast(returns)
        assert vol == 0.0

    def test_ewma_reasonable_range(self):
        """EWMA on typical stock returns should give 10-60% annualized vol."""
        np.random.seed(42)
        returns = np.random.normal(0.0003, 0.015, 252)  # Typical daily returns
        vol = VegaChimpCore.ewma_vol_forecast(returns)
        assert 0.05 < vol < 1.0, f"EWMA vol {vol:.4f} outside reasonable range"

    def test_ewma_higher_vol_for_volatile_series(self):
        """More volatile series should produce higher EWMA forecast."""
        np.random.seed(42)
        low_vol = np.random.normal(0, 0.005, 100)
        high_vol = np.random.normal(0, 0.02, 100)
        assert VegaChimpCore.ewma_vol_forecast(high_vol) > VegaChimpCore.ewma_vol_forecast(low_vol)


# ==================== Normal Distribution ====================

class TestNormalDist:
    """Test N(x) and n(x) implementations."""

    def test_N_at_zero(self):
        assert abs(VegaChimpCore.N(0) - 0.5) < 1e-10

    def test_N_large_positive(self):
        assert abs(VegaChimpCore.N(10) - 1.0) < 1e-10

    def test_N_large_negative(self):
        assert abs(VegaChimpCore.N(-10) - 0.0) < 1e-10

    def test_N_symmetry(self):
        """N(x) + N(-x) = 1."""
        for x in [0.5, 1.0, 2.0, 3.0]:
            assert abs(VegaChimpCore.N(x) + VegaChimpCore.N(-x) - 1.0) < 1e-10

    def test_n_at_zero(self):
        """n(0) = 1/sqrt(2*pi)."""
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        assert abs(VegaChimpCore.n(0) - expected) < 1e-10

    def test_n_symmetry(self):
        """n(x) = n(-x)."""
        for x in [0.5, 1.0, 2.0]:
            assert abs(VegaChimpCore.n(x) - VegaChimpCore.n(-x)) < 1e-10

    def test_n_positive(self):
        """PDF is always positive."""
        for x in [-3, -1, 0, 1, 3]:
            assert VegaChimpCore.n(x) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
