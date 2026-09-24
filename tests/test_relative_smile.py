"""Smile-shaped forecast vol. No network."""
import numpy as np
import pytest
from core.pricing import VegaChimpCore
from core.scan_service import relative_smile_vols


def test_flat_slice_stays_on_the_forecast():
    vols = relative_smile_vols(0.30, np.array([0.25, 0.25, 0.25]))
    assert np.allclose(vols, 0.30)


def test_shape_tracks_iv_when_the_forecast_matches_the_reference():
    ivs = np.array([0.20, 0.25, 0.40])
    weights = np.array([0.2, 1.0, 0.2])
    reference = np.average(ivs, weights=weights)
    vols = relative_smile_vols(reference, ivs, weights)
    assert np.allclose(vols, ivs, rtol=1e-6)


def test_a_tiny_iv_is_clipped_and_an_invalid_iv_stays_on_the_forecast():
    clipped = relative_smile_vols(0.30, np.array([0.01, 0.30]), np.array([1.0, 1.0]))
    assert clipped[0] == pytest.approx(0.15)
    assert clipped.max() <= 0.60 + 1e-9
    invalid = relative_smile_vols(0.30, np.array([1e-6, 0.30]), np.array([1.0, 1.0]))
    assert invalid[0] == pytest.approx(0.30)


def test_flat_forecast_misprices_a_skew_that_relative_smile_removes():
    spot, rate, years = 100.0, 0.0, 30 / 365
    forecast = 0.25
    chain = [(105.0, 0.20, True), (100.0, 0.25, True), (95.0, 0.35, False)]
    ivs = np.array([iv for _k, iv, _call in chain])
    shaped = relative_smile_vols(forecast, ivs, np.array([0.3, 1.0, 0.3]))
    for (strike, iv, is_call), fair_vol in zip(chain, shaped, strict=True):
        kind = "call" if is_call else "put"
        market = VegaChimpCore.bs_price(spot, strike, rate, 0.0, iv, years, kind)
        flat = VegaChimpCore.bs_price(spot, strike, rate, 0.0, forecast, years, kind)
        shaped_price = VegaChimpCore.bs_price(spot, strike, rate, 0.0, float(fair_vol), years, kind)
        if iv != forecast:
            assert abs(shaped_price - market) < abs(flat - market)
