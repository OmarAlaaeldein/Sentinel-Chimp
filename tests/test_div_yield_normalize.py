"""Regression: dividendYield /100 bug must stay fixed."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main.app import MarketApp


class TestNormalizeDivYield:
    def test_decimal_passthrough(self):
        assert MarketApp._normalize_div_yield(None, 0.0294) == pytest.approx(0.0294)

    def test_percent_form_divided_once(self):
        assert MarketApp._normalize_div_yield(None, 2.94) == pytest.approx(0.0294)

    def test_none_and_invalid(self):
        assert MarketApp._normalize_div_yield(None, None) is None
        assert MarketApp._normalize_div_yield(None, "n/a") is None
        assert MarketApp._normalize_div_yield(None, -0.1) is None

    def test_zero_ok(self):
        assert MarketApp._normalize_div_yield(None, 0.0) == 0.0

    def test_boundary_one_not_scaled(self):
        # 100% yield already in decimal form
        assert MarketApp._normalize_div_yield(None, 1.0) == pytest.approx(1.0)
        assert MarketApp._normalize_div_yield(None, 1.01) == pytest.approx(0.0101)
