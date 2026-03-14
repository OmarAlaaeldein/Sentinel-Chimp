import math
import unittest
from unittest import mock

import pandas as pd

from sentinel import MarketApp


class DummyTicker:
    def __init__(self, history_df=None, earnings_df=None, earnings_history=None):
        self._history_df = history_df
        self._earnings_df = earnings_df
        self._earnings_history = earnings_history

    def history(self, period="5y", interval="1d"):
        return self._history_df.copy()

    def get_earnings_dates(self, limit=80):
        if self._earnings_df is None:
            return pd.DataFrame()
        return self._earnings_df.copy()

    def get_earnings_history(self):
        return list(self._earnings_history or [])


class ValuationMetricsTests(unittest.TestCase):
    def make_app(self):
        app = MarketApp.__new__(MarketApp)
        app.valuation_cache = {}
        app.VALUATION_CACHE_DURATION = 3600
        app.current_ticker = "TEST"
        app.valuation_status = {}
        app.pe_fwd = None
        app.pe_ttm = None
        app.eps = None
        app.peg_ratio = None
        app.pe_percentile = None
        app.earnings_growth = None
        app.log = lambda *_args, **_kwargs: None
        return app

    def test_compute_peg_prefers_provider(self):
        app = self.make_app()
        app.pe_fwd = 18.0
        app.pe_ttm = 20.0
        app.peg_ratio = 1.25
        app.earnings_growth = 0.20

        app.compute_peg_ratio()

        self.assertAlmostEqual(app.peg_ratio, 1.25)
        self.assertEqual(app.valuation_status.get("peg_source"), "provider")
        self.assertIsNone(app.valuation_status.get("peg_reason"))

    def test_compute_peg_derives_from_forward_growth_percent(self):
        app = self.make_app()
        app.pe_fwd = 24.0
        app.pe_ttm = 30.0
        app.peg_ratio = None
        app.earnings_growth = 0.12  # 12%

        app.compute_peg_ratio()

        self.assertAlmostEqual(app.peg_ratio, 2.0)
        self.assertEqual(app.valuation_status.get("peg_source"), "derived_forward")

    def test_compute_peg_zero_growth_returns_inf(self):
        app = self.make_app()
        app.pe_fwd = 15.0
        app.peg_ratio = None
        app.earnings_growth = 0.0

        app.compute_peg_ratio()

        self.assertTrue(math.isinf(app.peg_ratio))
        self.assertEqual(app.valuation_status.get("peg_reason"), "ZERO_GROWTH")

    def test_compute_peg_negative_growth_keeps_raw_value(self):
        app = self.make_app()
        app.pe_fwd = 10.0
        app.peg_ratio = None
        app.earnings_growth = -0.05  # -5%

        app.compute_peg_ratio()

        self.assertAlmostEqual(app.peg_ratio, -2.0)
        self.assertEqual(app.valuation_status.get("peg_reason"), "NEGATIVE_GROWTH")

    def test_pe_percentile_uses_historical_ttm_eps_alignment(self):
        app = self.make_app()
        app.pe_ttm = 20.0

        idx = pd.to_datetime(["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15"], utc=True)
        history_df = pd.DataFrame({"Close": [10.0, 20.0, 30.0, 40.0]}, index=idx)
        ticker = DummyTicker(history_df=history_df)

        eps_timeline = pd.DataFrame(
            {
                "report_date": pd.to_datetime(["2023-12-31", "2024-03-01"], utc=True).tz_convert(None),
                "ttm_eps": [1.0, 2.0],
            }
        )

        with mock.patch.object(app, "_get_historical_ttm_eps", return_value=(eps_timeline, None)):
            app.calculate_valuation_multiplier(ticker)

        # Historical P/E series -> [10, 20, 15, 20], strictly lower than 20 are 2 of 4 -> 50%
        self.assertAlmostEqual(app.pe_percentile, 50.0)
        self.assertIsNone(app.valuation_status.get("pe_percentile_reason"))

    def test_historical_eps_falls_back_to_earnings_history(self):
        app = self.make_app()

        # Force primary source empty so fallback source is used.
        empty_dates_df = pd.DataFrame()
        earnings_history = [
            {"quarter": "2024-01-31", "epsActual": 1.0},
            {"quarter": "2024-04-30", "epsActual": 1.2},
            {"quarter": "2024-07-31", "epsActual": 1.1},
            {"quarter": "2024-10-31", "epsActual": 1.3},
            {"quarter": "2025-01-31", "epsActual": 1.4},
        ]
        ticker = DummyTicker(earnings_df=empty_dates_df, earnings_history=earnings_history)

        timeline, reason = app._get_historical_ttm_eps(ticker)

        self.assertIsNone(reason)
        self.assertIsNotNone(timeline)
        self.assertFalse(timeline.empty)
        self.assertIn("ttm_eps", timeline.columns)
        # At least two rolling-4 points exist for 5 quarterly records.
        self.assertGreaterEqual(len(timeline), 2)


if __name__ == "__main__":
    unittest.main()
