"""Prefs persistence + chart cone helper smoke tests (no GUI display)."""
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ui.prefs import load_prefs, save_prefs, DEFAULT_PREFS
from ui.chart import draw_probability_cone, prepare_plot_frame
from matplotlib.figure import Figure


class TestPrefs:
    def test_defaults_when_missing(self, tmp_path):
        data = load_prefs(str(tmp_path))
        assert data == DEFAULT_PREFS

    def test_round_trip(self, tmp_path):
        save_prefs(str(tmp_path), use_garch_blend=True, show_prob_cone=False)
        data = load_prefs(str(tmp_path))
        assert data["use_garch_blend"] is True
        assert data["show_prob_cone"] is False
        assert data["use_smile_vol"] is False


class TestChartConeHelper:
    def test_prepare_plot_frame_daily(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        df = pd.DataFrame(
            {"Close": np.linspace(100, 110, 10),
             "High": np.linspace(101, 111, 10),
             "Low": np.linspace(99, 109, 10),
             "Volume": 1000},
            index=idx,
        )
        plot_df, times, x = prepare_plot_frame(df, "1d")
        assert len(plot_df) == 10
        assert len(x) == 10

    def test_draw_cone_on_axes(self):
        fig = Figure()
        ax = fig.add_subplot(111)
        extent = draw_probability_cone(ax, last_x=9.0, p0=100.0, sigma=0.25, horizon_days=30)
        assert extent is not None
        assert extent[0] == pytest.approx(39.0)
        assert extent[2] > 100.0 > extent[1]
