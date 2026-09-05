"""Tests for batch BS2002, American FD Greeks, GARCH(1,1), and smile fit."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sentinel import VegaChimpCore
from core.vol_models import (
    garch11_vol_forecast, fit_quadratic_smile, smile_vol, smile_vol_arr,
)


class TestBatchBS2002:
    def test_batch_matches_scalar_calls_puts(self):
        S, T, r, q = 100.0, 0.5, 0.08, 0.12
        K = np.linspace(80, 120, 41)
        sig = np.linspace(0.15, 0.35, 41)
        kinds = np.array(['call' if i % 2 == 0 else 'put' for i in range(41)])
        batch = VegaChimpCore.bjerksund_stensland_batch(S, K, T, r, q, sig, kinds)
        for i in range(41):
            ref = VegaChimpCore.bjerksund_stensland(
                S, float(K[i]), T, r, q, float(sig[i]), kinds[i],
            )
            # Hastings erf in batch ⇒ ~1e-5; still tick-accurate
            assert abs(batch[i] - ref) < 5e-5, (i, batch[i], ref)

    def test_batch_paper_table1(self):
        price = float(VegaChimpCore.bjerksund_stensland_batch(
            100, 100, 0.5, 0.08, 0.12, 0.20, 'call',
        ))
        assert abs(price - 4.69) < 0.01

    def test_batch_scalar_string_broadcast(self):
        prices = VegaChimpCore.bjerksund_stensland_batch(
            100, [90, 100, 110], 1.0, 0.05, 0.02, [0.25, 0.25, 0.25], 'put',
        )
        assert prices.shape == (3,)
        assert np.all(prices > 0)

    def test_batch_q0_call_equals_european(self):
        S, K, T, r, q, sig = 100.0, np.array([95.0, 100.0, 105.0]), 1.0, 0.05, 0.0, 0.25
        am = VegaChimpCore.bjerksund_stensland_batch(S, K, T, r, q, sig, 'call')
        for i, k in enumerate(K):
            euro = VegaChimpCore.bs_price(S, float(k), r, q, sig, T, 'call')
            assert abs(am[i] - euro) < 1e-4


class TestAmericanGreeks:
    def test_fd_matches_euro_when_q0_call(self):
        """With q=0, American call = European ⇒ FD Greeks ≈ analytic BS."""
        S, K, r, q, sig, T = 100.0, 100.0, 0.05, 0.0, 0.25, 1.0
        am = VegaChimpCore.american_greeks(S, K, r, q, sig, T, 'call')
        eu = VegaChimpCore.bs_greeks(S, K, r, q, sig, T, 'call')
        assert abs(am['delta'] - eu['delta']) < 0.01
        assert abs(am['gamma'] - eu['gamma']) < 0.01
        assert abs(am['vega'] - eu['vega']) < 0.05
        # theta/rho: FD vs analytic — looser
        assert abs(am['theta'] - eu['theta']) < 0.05

    def test_put_delta_negative(self):
        g = VegaChimpCore.american_greeks(100, 100, 0.05, 0.02, 0.25, 1.0, 'put')
        assert -1.0 <= g['delta'] <= 0.0
        assert g['gamma'] > 0
        assert g['vega'] > 0

    def test_batch_greeks_match_scalar(self):
        S, T, r, q = 100.0, 0.75, 0.05, 0.03
        K = np.array([90.0, 100.0, 110.0])
        sig = np.array([0.2, 0.25, 0.3])
        kinds = np.array(['call', 'put', 'call'])
        batch = VegaChimpCore.american_greeks_batch(S, K, r, q, sig, T, kinds)
        for i in range(3):
            ref = VegaChimpCore.american_greeks(
                S, float(K[i]), r, q, float(sig[i]), T, kinds[i],
            )
            assert abs(batch['delta'][i] - ref['delta']) < 5e-4
            assert abs(batch['gamma'][i] - ref['gamma']) < 5e-4
            assert abs(batch['vega'][i] - ref['vega']) < 5e-4

    def test_american_put_delta_fd_sanity(self):
        S, K, r, q, sig, T = 100.0, 100.0, 0.05, 0.02, 0.25, 1.0
        h = 0.05
        p_up = VegaChimpCore.bjerksund_stensland(S + h, K, T, r, q, sig, 'put')
        p_dn = VegaChimpCore.bjerksund_stensland(S - h, K, T, r, q, sig, 'put')
        num = (p_up - p_dn) / (2 * h)
        g = VegaChimpCore.american_greeks(S, K, r, q, sig, T, 'put', dS=h)
        assert abs(g['delta'] - num) < 1e-6


class TestGARCH:
    def test_garch_positive_on_sufficient_data(self):
        rng = np.random.default_rng(0)
        # Simulate true GARCH(1,1) with clustering so α>0 is identifiable
        n, omega, alpha0, beta0 = 400, 1e-6, 0.08, 0.90
        rets = np.empty(n)
        var = omega / (1.0 - alpha0 - beta0)
        for i in range(n):
            prev = rets[i - 1] ** 2 if i else var
            var = omega + alpha0 * prev + beta0 * var
            rets[i] = rng.normal(0.0, np.sqrt(var))
        vol, info = garch11_vol_forecast(rets)
        assert info['ok']
        assert vol > 0
        assert 0 < info['alpha'] < 1
        assert 0 < info['beta'] < 1
        assert info['alpha'] + info['beta'] < 1
        # Recover roughly the DGP (loose — coarse QMLE grid)
        assert abs(info['alpha'] - alpha0) < 0.08
        assert abs(info['beta'] - beta0) < 0.10

    def test_garch_short_series_returns_zero(self):
        vol, info = garch11_vol_forecast(np.random.randn(20))
        assert vol == 0.0
        assert not info['ok']

    def test_garch_higher_for_volatile_series(self):
        rng = np.random.default_rng(1)
        low, _ = garch11_vol_forecast(rng.normal(0, 0.005, 250))
        high, _ = garch11_vol_forecast(rng.normal(0, 0.025, 250))
        assert high > low

    def test_garch_ignores_non_finite(self):
        rng = np.random.default_rng(2)
        rets = rng.normal(0, 0.01, 200)
        rets[10] = np.nan
        rets[50] = np.inf
        vol, info = garch11_vol_forecast(rets)
        assert info['ok'] and math.isfinite(vol) and vol > 0


class TestSmile:
    def test_quadratic_smile_recovers_coeffs(self):
        forward = 100.0
        strikes = np.linspace(70, 130, 25)
        a, b, c = 0.22, -0.15, 0.80  # typical skew + smile
        k = np.log(strikes / forward)
        ivs = a + b * k + c * k * k
        fit = fit_quadratic_smile(strikes, ivs, forward)
        assert fit is not None
        assert abs(fit[0] - a) < 1e-6
        assert abs(fit[1] - b) < 1e-6
        assert abs(fit[2] - c) < 1e-6

    def test_smile_vol_clamps(self):
        coef = (0.2, 0.0, 0.0)
        assert smile_vol(100, 100, coef) == pytest.approx(0.2)
        assert smile_vol(1e-12, 100, coef) == pytest.approx(0.2)

    def test_smile_arr_matches_scalar(self):
        coef = (0.25, -0.1, 0.5)
        strikes = np.array([80.0, 100.0, 120.0])
        arr = smile_vol_arr(strikes, 100.0, coef)
        for i, k in enumerate(strikes):
            assert abs(arr[i] - smile_vol(float(k), 100.0, coef)) < 1e-12

    def test_smile_rejects_too_few_points(self):
        assert fit_quadratic_smile([100, 105], [0.2, 0.21], 100.0) is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
