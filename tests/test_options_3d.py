"""Headless tests for options 3D helpers (Plotly + binning)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.options_3d import (
    bin_ev_grid,
    build_plotly_figure,
    categorize_row,
    filter_rows_for_plot,
    CAT_EARN_UNDER,
    CAT_UNDER,
    CAT_OVER,
)


class TestCategorize:
    def test_categories(self):
        assert categorize_row(True, True) == CAT_EARN_UNDER
        assert categorize_row(False, True) == CAT_UNDER
        assert categorize_row(False, False) == CAT_OVER


class TestFilter:
    def test_filters(self):
        rows = [
            {"date": "2026-12-01", "strike": 100, "ev": 0.5, "vol": 10,
             "is_earnings": False, "is_good": True},
            {"date": "2026-12-01", "strike": 105, "ev": -0.4, "vol": 5,
             "is_earnings": False, "is_good": False},
            {"date": "2026-12-15", "strike": 100, "ev": 0.8, "vol": 20,
             "is_earnings": True, "is_good": True},
        ]
        out = filter_rows_for_plot(
            rows,
            show_earn_under=True,
            show_earn_over=False,
            show_under=True,
            show_over=False,
        )
        assert len(out) == 2
        assert all(r["category"] in (CAT_UNDER, CAT_EARN_UNDER) for r in out)


class TestBinGrid:
    def test_sparse_returns_none(self):
        assert bin_ev_grid([1, 2], [100, 101], [0.1, 0.2]) is None

    def test_grid_shape(self):
        rng = np.random.default_rng(0)
        days = rng.integers(7, 120, size=40).astype(float)
        strikes = rng.uniform(90, 110, size=40)
        evs = rng.normal(0, 0.5, size=40)
        grid = bin_ev_grid(days, strikes, evs)
        assert grid is not None
        d_c, s_c, Z = grid
        assert Z.shape == (len(s_c), len(d_c))
        assert np.isfinite(Z).sum() >= 4


class TestPlotlyFigure:
    def test_builds_with_colorbar_and_hover(self):
        pytest.importorskip("plotly")
        days = [30, 45, 60, 30, 45, 90, 60, 90]
        strikes = [100, 100, 105, 110, 95, 100, 110, 95]
        evs = [0.4, -0.2, 0.1, -0.5, 0.6, 0.0, -0.1, 0.3]
        labels = ["2026-10-01"] * len(days)
        vols = [10, 20, 5, 8, 40, 12, 3, 15]
        cats = [CAT_UNDER, CAT_OVER, CAT_UNDER, CAT_OVER,
                CAT_EARN_UNDER, CAT_UNDER, CAT_OVER, CAT_UNDER]
        fig = build_plotly_figure(
            "TEST", "CALL", days, strikes, evs, labels, vols, cats,
        )
        assert fig is not None
        # At least the scatter; heatmap may or may not appear depending on bins
        assert len(fig.data) >= 1
        scatter = fig.data[0]
        assert scatter.type == "scatter3d"
        assert "EV@Ask" in (scatter.marker.colorbar.title.text or "")
        assert scatter.hoverinfo == "text"
        # Camera set on scene
        scene = fig.layout.scene
        assert scene.camera.eye.x is not None
        assert "EV@Ask" in (scene.zaxis.title.text or "")
