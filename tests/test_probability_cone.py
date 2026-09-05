"""Tests for Phase II probability-cone math helpers."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.vol_models import probability_cone, blend_forecast_vol


class TestProbabilityCone:
    def test_t0_collapses_to_spot(self):
        days, upper, lower = probability_cone(100.0, 0.25, horizon_days=30)
        assert days[0] == 0.0
        assert abs(upper[0] - 100.0) < 1e-12
        assert abs(lower[0] - 100.0) < 1e-12

    def test_formula_at_day_t(self):
        p0, sigma, t, n = 100.0, 0.30, 21, 252
        days, upper, lower = probability_cone(p0, sigma, horizon_days=30)
        idx = int(np.where(days == t)[0][0])
        expected_u = p0 * math.exp(+sigma * math.sqrt(t / n))
        expected_l = p0 * math.exp(-sigma * math.sqrt(t / n))
        assert abs(upper[idx] - expected_u) < 1e-10
        assert abs(lower[idx] - expected_l) < 1e-10

    def test_symmetry_in_log_space(self):
        p0, sigma = 55.5, 0.42
        _, upper, lower = probability_cone(p0, sigma, horizon_days=10)
        # log(U/P0) == -log(L/P0) == σ√(t/N)
        mid = np.sqrt(upper * lower)
        assert np.allclose(mid, p0, rtol=0, atol=1e-9)

    def test_zero_vol_is_flat(self):
        days, upper, lower = probability_cone(200.0, 0.0, horizon_days=5)
        assert np.allclose(upper, 200.0)
        assert np.allclose(lower, 200.0)
        assert len(days) == 6  # t=0..5

    def test_without_t0(self):
        days, upper, lower = probability_cone(
            100.0, 0.2, horizon_days=3, include_t0=False,
        )
        assert list(days) == [1.0, 2.0, 3.0]
        assert upper[0] > 100.0 > lower[0]

    def test_wider_with_higher_vol(self):
        _, u1, l1 = probability_cone(100.0, 0.10, horizon_days=30)
        _, u2, l2 = probability_cone(100.0, 0.40, horizon_days=30)
        assert (u2[-1] - l2[-1]) > (u1[-1] - l1[-1])

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            probability_cone(0.0, 0.2)
        with pytest.raises(ValueError):
            probability_cone(100.0, -0.1)
        with pytest.raises(ValueError):
            probability_cone(100.0, 0.2, horizon_days=-1)


class TestBlendForecastVol:
    def test_ewma_default(self):
        assert blend_forecast_vol(0.25, 0.40, False) == 0.25

    def test_garch_blend_half(self):
        assert abs(blend_forecast_vol(0.20, 0.40, True) - 0.30) < 1e-12

    def test_garch_zero_falls_back(self):
        assert blend_forecast_vol(0.22, 0.0, True) == 0.22

    def test_invalid_ewma(self):
        assert blend_forecast_vol(float('nan'), 0.3, False) == 0.0
