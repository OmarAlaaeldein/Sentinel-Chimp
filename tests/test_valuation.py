"""core.valuation: Sentinel's P/E, PEG and 5-year P/E percentile rules, no window.

The ``_legacy_*`` functions below are verbatim copies of the method bodies that
lived in ``MarketApp`` (main/app.py at e58bacd) before the rules moved into
``core/valuation.py``. They are the oracle: the new functions must give the
same numbers and the same reasons on every case, so a refactor cannot drift.
"""
import ast
import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core import valuation as val  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Oracle: the old MarketApp bodies, copied without changes (self -> ``s``).
# --------------------------------------------------------------------------
def _legacy_to_finite_float(value):
    try:
        f_val = float(value)
        if not math.isfinite(f_val):
            return None
        return f_val
    except (TypeError, ValueError):
        return None


def _legacy_as_naive_datetime64_us(values):
    ts = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(ts, pd.Series):
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert(None)
        return ts.astype("datetime64[us]")
    if isinstance(ts, pd.DatetimeIndex):
        if ts.tz is not None:
            ts = ts.tz_convert(None)
        return ts.astype("datetime64[us]")
    ts = pd.DatetimeIndex(ts)
    if ts.tz is not None:
        ts = ts.tz_convert(None)
    return pd.Series(ts.astype("datetime64[us]"))


def _legacy_get_info(info):
    s = SimpleNamespace(valuation_status={})
    s.pe_fwd = _legacy_to_finite_float(info.get("forwardPE"))
    s.pe_ttm = _legacy_to_finite_float(info.get("trailingPE"))
    s.peg_ratio = _legacy_to_finite_float(info.get("trailingPegRatio"))
    s.earnings_growth = _legacy_to_finite_float(info.get("earningsGrowth"))
    return s


def _legacy_compute_peg_ratio(s):
    s.valuation_status["peg_reason"] = None

    provider_peg = _legacy_to_finite_float(s.peg_ratio)
    if provider_peg is not None:
        s.peg_ratio = provider_peg
        return

    pe_for_peg = _legacy_to_finite_float(s.pe_fwd)
    if pe_for_peg is None:
        pe_for_peg = _legacy_to_finite_float(s.pe_ttm)

    growth_dec = _legacy_to_finite_float(s.earnings_growth)
    if pe_for_peg is None or growth_dec is None:
        s.peg_ratio = None
        s.valuation_status["peg_reason"] = "MISSING_PEG_INPUTS"
        return

    growth_pct = growth_dec * 100.0
    if abs(growth_pct) < 1e-9:
        s.peg_ratio = math.inf if pe_for_peg >= 0 else -math.inf
        s.valuation_status["peg_reason"] = "ZERO_GROWTH"
        return

    s.peg_ratio = pe_for_peg / growth_pct


def _legacy_ttm_eps(earnings_df):
    """_get_historical_ttm_eps without the cache and without the provider call."""
    if earnings_df is None or earnings_df.empty:
        return None, "NO_EARNINGS_HISTORY"

    reported_col = None
    for col in earnings_df.columns:
        col_name = str(col).lower()
        if "reported" in col_name and "eps" in col_name:
            reported_col = col
            break

    if reported_col is None:
        return None, "NO_EARNINGS_HISTORY"

    eps_series = pd.to_numeric(earnings_df[reported_col], errors="coerce").dropna()
    if eps_series.empty:
        return None, "NO_EARNINGS_HISTORY"

    eps_df = pd.DataFrame({"reported_eps": eps_series})
    eps_df["report_date"] = _legacy_as_naive_datetime64_us(eps_df.index)
    eps_df = eps_df.dropna(subset=["report_date"]).sort_values("report_date")
    eps_df = eps_df.drop_duplicates(subset=["report_date"], keep="last")
    eps_df["ttm_eps"] = eps_df["reported_eps"].rolling(4).sum()
    eps_df = eps_df.dropna(subset=["ttm_eps"])
    if eps_df.empty:
        return None, "INSUFFICIENT_EARNINGS_HISTORY"

    return eps_df[["report_date", "ttm_eps"]].copy(), None


def _legacy_pe_percentile(hist, eps_timeline, eps_reason, pe_ttm):
    """calculate_pe_percentile without the provider calls; returns (value, reason)."""
    current_pe_ttm = _legacy_to_finite_float(pe_ttm)
    if current_pe_ttm is None:
        return None, "NO_CURRENT_PE_TTM"
    try:
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None, "NO_PRICE_HISTORY"

        if eps_timeline is None or eps_timeline.empty:
            return None, eps_reason or "NO_EARNINGS_HISTORY"

        hist_df = hist[["Close"]].copy().dropna()
        hist_df["date"] = _legacy_as_naive_datetime64_us(hist_df.index)
        hist_df = hist_df.dropna(subset=["date"]).sort_values("date")

        eps_timeline = eps_timeline.copy()
        eps_timeline["report_date"] = _legacy_as_naive_datetime64_us(eps_timeline["report_date"])
        eps_timeline = eps_timeline.dropna(subset=["report_date"]).sort_values("report_date")

        merged = pd.merge_asof(
            hist_df[["date", "Close"]],
            eps_timeline,
            left_on="date",
            right_on="report_date",
            direction="backward",
        )

        pe_series = pd.to_numeric(merged["Close"], errors="coerce") / pd.to_numeric(merged["ttm_eps"], errors="coerce")
        pe_series = pe_series.replace([np.inf, -np.inf], np.nan).dropna()
        pe_series = pe_series[pe_series > 0]
        if pe_series.empty:
            return None, "NO_VALID_PE_HISTORY"

        return float((pe_series < current_pe_ttm).mean() * 100.0), None
    except Exception:
        return None, "NO_VALID_PE_HISTORY"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _history(closes, start="2024-01-02", tz=None):
    idx = pd.bdate_range(start, periods=len(closes), tz=tz)
    return pd.DataFrame({"Close": [float(c) for c in closes]}, index=idx)


def _timeline(rows):
    return pd.DataFrame(
        {
            "report_date": pd.to_datetime([d for d, _ in rows]).astype("datetime64[us]"),
            "ttm_eps": [float(v) for _, v in rows],
        }
    )


def _earnings_df(dates, reported, tz="America/New_York"):
    """Shape of yfinance ``get_earnings_dates``: tz-aware index, three columns."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(tz)
    idx.name = "Earnings Date"
    return pd.DataFrame(
        {
            "EPS Estimate": [1.0] * len(dates),
            "Reported EPS": reported,
            "Surprise(%)": [0.0] * len(dates),
        },
        index=idx,
    )


# ten closes 50..140; P/E is close / 5 until 2024-01-09, close / 10 from then on
CLOSES = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
TWO_REPORTS = [("2023-12-01", 5.0), ("2024-01-09", 10.0)]


# --------------------------------------------------------------------------
# from_info / peg_ratio
# --------------------------------------------------------------------------
class TestFromInfo:
    def test_keys_are_exactly_the_documented_ones(self):
        out = val.from_info({})
        assert set(out) == {"pe_ttm", "pe_fwd", "peg", "peg_source", "earnings_growth", "peg_reason"}

    def test_provider_peg_wins_over_derived(self):
        out = val.from_info(
            {"trailingPE": 30.2, "forwardPE": 25.1, "trailingPegRatio": 1.3, "earningsGrowth": 0.5}
        )
        assert out["pe_ttm"] == 30.2
        assert out["pe_fwd"] == 25.1
        assert out["peg"] == 1.3
        assert out["peg_source"] == "provider"
        assert out["earnings_growth"] == 0.5
        assert out["peg_reason"] is None

    def test_provider_peg_of_zero_is_still_the_provider_peg(self):
        out = val.from_info({"trailingPegRatio": 0.0, "forwardPE": 20, "earningsGrowth": 0.1})
        assert out["peg"] == 0.0
        assert out["peg_source"] == "provider"

    def test_derived_peg_uses_forward_pe_first(self):
        out = val.from_info({"trailingPE": 30.0, "forwardPE": 24.0, "earningsGrowth": 0.2})
        assert out["peg"] == pytest.approx(24.0 / 20.0)
        assert out["peg_source"] == "derived"
        assert out["peg_reason"] is None

    def test_derived_peg_falls_back_to_ttm_when_no_forward(self):
        out = val.from_info({"trailingPE": 30.0, "earningsGrowth": 0.25})
        assert out["peg"] == pytest.approx(30.0 / 25.0)
        assert out["peg_source"] == "derived"

    def test_negative_growth_gives_a_negative_peg_not_a_hidden_one(self):
        out = val.from_info({"forwardPE": 20.0, "earningsGrowth": -0.1})
        assert out["peg"] == pytest.approx(-2.0)
        assert out["peg_source"] == "derived"
        assert out["peg_reason"] is None

    def test_zero_growth_is_infinite_with_a_reason(self):
        out = val.from_info({"forwardPE": 20.0, "earningsGrowth": 0.0})
        assert out["peg"] == math.inf
        assert out["peg_reason"] == "ZERO_GROWTH"
        assert out["peg_source"] == "derived"

    def test_zero_growth_with_negative_pe_is_minus_infinity(self):
        out = val.from_info({"forwardPE": -5.0, "earningsGrowth": 0.0})
        assert out["peg"] == -math.inf
        assert out["peg_reason"] == "ZERO_GROWTH"

    def test_growth_under_a_billionth_of_a_percent_counts_as_zero(self):
        out = val.from_info({"forwardPE": 20.0, "earningsGrowth": 1e-12})
        assert out["peg"] == math.inf
        assert out["peg_reason"] == "ZERO_GROWTH"

    def test_growth_just_above_that_is_a_real_growth_rate(self):
        out = val.from_info({"forwardPE": 20.0, "earningsGrowth": 1e-10})  # 1e-8 percent
        assert out["peg"] == pytest.approx(20.0 / 1e-8)
        assert out["peg_reason"] is None

    @pytest.mark.parametrize(
        "info",
        [
            {},
            {"forwardPE": 20.0},
            {"earningsGrowth": 0.2},
            {"trailingPE": None, "forwardPE": None, "earningsGrowth": 0.2},
        ],
    )
    def test_missing_inputs(self, info):
        out = val.from_info(info)
        assert out["peg"] is None
        assert out["peg_source"] is None
        assert out["peg_reason"] == "MISSING_PEG_INPUTS"

    def test_none_info_reads_as_empty(self):
        out = val.from_info(None)
        assert out["pe_ttm"] is None and out["pe_fwd"] is None and out["peg"] is None

    def test_non_finite_and_non_numeric_strings_read_as_missing(self):
        out = val.from_info(
            {
                "trailingPE": "Infinity",
                "forwardPE": float("nan"),
                "trailingPegRatio": "N/A",
                "earningsGrowth": float("inf"),
            }
        )
        assert out["pe_ttm"] is None
        assert out["pe_fwd"] is None
        assert out["earnings_growth"] is None
        assert out["peg"] is None
        assert out["peg_reason"] == "MISSING_PEG_INPUTS"

    def test_infinite_provider_peg_falls_through_to_derived(self):
        out = val.from_info({"trailingPegRatio": float("inf"), "forwardPE": 20.0, "earningsGrowth": 0.1})
        assert out["peg"] == pytest.approx(2.0)
        assert out["peg_source"] == "derived"

    def test_numeric_strings_still_parse_as_in_the_window(self):
        # float("12.5") has always worked in the window; the move keeps it.
        assert val.from_info({"trailingPE": "12.5"})["pe_ttm"] == 12.5

    def test_peg_ratio_function_matches_from_info(self):
        assert val.peg_ratio(None, 24.0, 30.0, 0.2) == (pytest.approx(1.2), "derived", None)
        assert val.peg_ratio(1.3, 24.0, 30.0, 0.2) == (1.3, "provider", None)
        assert val.peg_ratio(None, None, None, None) == (None, None, "MISSING_PEG_INPUTS")

    def test_read_info_returns_the_raw_provider_fields(self):
        raw = val.read_info(
            {"forwardPE": 25.1, "trailingPE": "bad", "trailingPegRatio": 1.3, "earningsGrowth": 0.2}
        )
        assert raw == {"pe_fwd": 25.1, "pe_ttm": None, "provider_peg": 1.3, "earnings_growth": 0.2}


class TestFromInfoMatchesTheOldWindowRules:
    """Every combination, run through the old MarketApp bodies and the new function."""

    PEGS = [None, 1.3, 0.0, -0.5, float("inf"), float("nan"), "N/A", "Infinity"]
    FWDS = [None, 25.1, -4.0, 0.0, float("nan")]
    TTMS = [None, 30.2, -8.0]
    GROWTHS = [None, 0.2, -0.1, 0.0, 1e-12, 1e-10, 1e-8, float("inf")]

    def test_grid(self):
        n = 0
        for peg in self.PEGS:
            for fwd in self.FWDS:
                for ttm in self.TTMS:
                    for growth in self.GROWTHS:
                        info = {
                            "trailingPegRatio": peg,
                            "forwardPE": fwd,
                            "trailingPE": ttm,
                            "earningsGrowth": growth,
                        }
                        old = _legacy_get_info(info)
                        _legacy_compute_peg_ratio(old)
                        new = val.from_info(info)
                        n += 1
                        for old_value, new_value in (
                            (old.pe_ttm, new["pe_ttm"]),
                            (old.pe_fwd, new["pe_fwd"]),
                            (old.peg_ratio, new["peg"]),
                            (old.earnings_growth, new["earnings_growth"]),
                        ):
                            assert old_value == new_value, info
                        assert old.valuation_status["peg_reason"] == new["peg_reason"], info
        assert n == len(self.PEGS) * len(self.FWDS) * len(self.TTMS) * len(self.GROWTHS)


# --------------------------------------------------------------------------
# ttm_eps_timeline
# --------------------------------------------------------------------------
class TestTtmEpsTimeline:
    def test_rolling_four_quarter_sum(self):
        df = _earnings_df(
            ["2024-04-25", "2023-10-26", "2024-01-25", "2023-07-27", "2023-04-27"],  # unsorted on purpose
            [5.0, 3.0, 4.0, 2.0, 1.0],
        )
        timeline, reason = val.ttm_eps_timeline(df)
        assert reason is None
        assert list(timeline.columns) == ["report_date", "ttm_eps"]
        assert list(timeline["ttm_eps"]) == [1 + 2 + 3 + 4, 2 + 3 + 4 + 5]
        assert str(timeline["report_date"].dtype) == "datetime64[us]"
        assert timeline["report_date"].is_monotonic_increasing

    def test_future_rows_without_a_reported_eps_are_ignored(self):
        df = _earnings_df(
            ["2026-01-29", "2025-10-30", "2025-07-31", "2025-04-30", "2025-01-30"],
            [np.nan, 4.0, 3.0, 2.0, 1.0],
        )
        timeline, reason = val.ttm_eps_timeline(df)
        assert reason is None
        assert list(timeline["ttm_eps"]) == [10.0]

    def test_two_rows_on_one_report_date_count_once(self):
        df = _earnings_df(
            ["2023-04-27", "2023-07-27", "2023-07-27", "2023-10-26", "2024-01-25"],
            [1.0, 7.0, 2.0, 3.0, 4.0],
        )
        timeline, _reason = val.ttm_eps_timeline(df)
        assert list(timeline["ttm_eps"]) == [1 + 2 + 3 + 4]

    def test_exactly_four_quarters_make_one_point(self):
        df = _earnings_df(["2023-07-27", "2023-10-26", "2024-01-25", "2024-04-25"], [1.0, 2.0, 3.0, 4.0])
        timeline, reason = val.ttm_eps_timeline(df)
        assert reason is None
        assert len(timeline) == 1

    def test_fewer_than_four_quarters(self):
        df = _earnings_df(["2023-10-26", "2024-01-25", "2024-04-25"], [2.0, 3.0, 4.0])
        assert val.ttm_eps_timeline(df) == (None, "INSUFFICIENT_EARNINGS_HISTORY")

    def test_no_reported_eps_column(self):
        df = _earnings_df(["2023-10-26", "2024-01-25"], [1.0, 2.0]).drop(columns=["Reported EPS"])
        assert val.ttm_eps_timeline(df) == (None, "NO_EARNINGS_HISTORY")

    def test_column_is_found_by_name_case_insensitively(self):
        df = _earnings_df(["2023-07-27", "2023-10-26", "2024-01-25", "2024-04-25"], [1.0, 2.0, 3.0, 4.0])
        df = df.rename(columns={"Reported EPS": "reported eps (usd)"})
        timeline, reason = val.ttm_eps_timeline(df)
        assert reason is None and list(timeline["ttm_eps"]) == [10.0]

    @pytest.mark.parametrize("empty", [None, pd.DataFrame()])
    def test_nothing_to_read(self, empty):
        assert val.ttm_eps_timeline(empty) == (None, "NO_EARNINGS_HISTORY")

    def test_all_reported_values_missing(self):
        df = _earnings_df(["2023-10-26", "2024-01-25"], [np.nan, np.nan])
        assert val.ttm_eps_timeline(df) == (None, "NO_EARNINGS_HISTORY")

    def test_garbage_does_not_raise_and_is_logged(self):
        seen = []
        assert val.ttm_eps_timeline("not a frame", log=seen.append) == (None, "NO_EARNINGS_HISTORY")
        assert seen and seen[0].startswith("Historical EPS fetch error:")

    def test_matches_the_old_window_rules(self):
        rng = np.random.default_rng(7)
        for case in range(25):
            n = int(rng.integers(0, 14))
            dates = pd.to_datetime("2020-01-15") + pd.to_timedelta(rng.integers(0, 1500, n), unit="D")
            eps = rng.normal(1.0, 1.0, n)
            eps[rng.random(n) < 0.2] = np.nan
            if n:
                df = _earnings_df(list(dates), list(eps))
            else:
                df = pd.DataFrame()
            old = _legacy_ttm_eps(df)
            new = val.ttm_eps_timeline(df)
            assert old[1] == new[1], case
            if old[0] is None:
                assert new[0] is None
            else:
                pd.testing.assert_frame_equal(old[0], new[0])


# --------------------------------------------------------------------------
# pe_percentile
# --------------------------------------------------------------------------
class TestPePercentile:
    def test_known_answer_is_strictly_below(self):
        # P/E: 10 12 14 16 18 | 10 11 12 13 14 ; 16.0 equals one value and does not count
        assert val.pe_percentile(_history(CLOSES), _timeline(TWO_REPORTS), 16.0) == (80.0, None)

    def test_low_and_high_ends(self):
        assert val.pe_percentile(_history(CLOSES), _timeline(TWO_REPORTS), 9.0) == (0.0, None)
        assert val.pe_percentile(_history(CLOSES), _timeline(TWO_REPORTS), 100.0) == (100.0, None)

    def test_days_before_the_first_report_are_left_out(self):
        timeline = _timeline([("2024-01-04", 5.0), ("2024-01-09", 10.0)])
        # eight priced days: 14 16 18 | 10 11 12 13 14 ; below 15 -> 14,10,11,12,13,14
        assert val.pe_percentile(_history(CLOSES), timeline, 15.0) == (75.0, None)

    def test_negative_pe_history_is_dropped(self):
        timeline = _timeline([("2023-12-01", -5.0), ("2024-01-09", 10.0)])
        # only 10 11 12 13 14 count ; below 12.5 -> 10, 11, 12
        assert val.pe_percentile(_history(CLOSES), timeline, 12.5) == (60.0, None)

    def test_only_negative_or_zero_eps_leaves_no_valid_history(self):
        for eps in (-5.0, 0.0):
            timeline = _timeline([("2023-12-01", eps)])
            assert val.pe_percentile(_history(CLOSES), timeline, 15.0) == (None, "NO_VALID_PE_HISTORY")

    def test_a_report_on_the_same_day_applies_that_day(self):
        # 2024-01-09 is the sixth close (100): 100 / 10 = 10, not 100 / 5 = 20
        history = _history(CLOSES)
        timeline = _timeline(TWO_REPORTS)
        value, _ = val.pe_percentile(history, timeline, 10.5)
        assert value == pytest.approx(2 / 10 * 100)  # 10 (day 0) and 10 (day 5)

    def test_close_gaps_are_dropped_not_zero(self):
        closes = [50, np.nan, 70, 80, 90, 100, 110, 120, 130, 140]
        value, reason = val.pe_percentile(_history(closes), _timeline(TWO_REPORTS), 16.0)
        assert reason is None
        assert value == pytest.approx(7 / 9 * 100)

    def test_works_on_tz_aware_history_and_tz_aware_earnings(self):
        history = _history(CLOSES, start="2024-01-02", tz="America/New_York")
        df = _earnings_df(
            ["2023-01-26", "2023-04-27", "2023-07-27", "2023-10-26", "2024-01-09"],
            [1.0, 1.1, 1.2, 1.3, 1.4],
        )
        timeline, _ = val.ttm_eps_timeline(df)
        value, reason = val.pe_percentile(history, timeline, 30.0)
        assert reason is None
        assert 0.0 <= value <= 100.0
        old = _legacy_pe_percentile(history, timeline, None, 30.0)
        assert (value, reason) == old

    # every reason
    def test_reason_no_current_pe(self):
        for pe in (None, float("nan"), float("inf"), "n/a"):
            assert val.pe_percentile(_history(CLOSES), _timeline(TWO_REPORTS), pe) == (None, "NO_CURRENT_PE_TTM")

    def test_reason_no_current_pe_wins_over_missing_history(self):
        assert val.pe_percentile(None, None, None) == (None, "NO_CURRENT_PE_TTM")

    @pytest.mark.parametrize(
        "history",
        [None, pd.DataFrame(), pd.DataFrame({"Open": [1.0]}, index=pd.bdate_range("2024-01-02", periods=1))],
    )
    def test_reason_no_price_history(self, history):
        assert val.pe_percentile(history, _timeline(TWO_REPORTS), 20.0) == (None, "NO_PRICE_HISTORY")

    def test_reason_no_earnings_history(self):
        assert val.pe_percentile(_history(CLOSES), None, 20.0) == (None, "NO_EARNINGS_HISTORY")
        assert val.pe_percentile(_history(CLOSES), _timeline([]), 20.0) == (None, "NO_EARNINGS_HISTORY")

    def test_the_reason_the_timeline_gave_is_kept(self):
        assert val.pe_percentile(
            _history(CLOSES), None, 20.0, eps_reason="INSUFFICIENT_EARNINGS_HISTORY"
        ) == (None, "INSUFFICIENT_EARNINGS_HISTORY")

    def test_reason_no_valid_pe_history_when_the_frames_do_not_fit(self):
        seen = []
        bad = pd.DataFrame({"report_date": pd.to_datetime(["2023-12-01"]), "eps": [5.0]})
        assert val.pe_percentile(_history(CLOSES), bad, 20.0, log=seen.append) == (None, "NO_VALID_PE_HISTORY")
        assert seen and seen[0].startswith("P/E Percentile error:")

    def test_every_reason_has_a_window_label(self):
        for reason in (
            "NO_CURRENT_PE_TTM",
            "NO_PRICE_HISTORY",
            "NO_EARNINGS_HISTORY",
            "INSUFFICIENT_EARNINGS_HISTORY",
            "NO_VALID_PE_HISTORY",
            "MISSING_PEG_INPUTS",
            "ZERO_GROWTH",
        ):
            assert reason in val.REASONS

    def test_reason_labels_are_the_ones_the_window_showed(self):
        assert val.REASONS == {
            "NO_CURRENT_PE_TTM": "N/A (No TTM P/E)",
            "NO_PRICE_HISTORY": "N/A (No 5Y Price)",
            "NO_EARNINGS_HISTORY": "N/A (No EPS History)",
            "INSUFFICIENT_EARNINGS_HISTORY": "N/A (EPS < 4 Qtrs)",
            "NO_VALID_PE_HISTORY": "N/A (No Valid P/E)",
            "MISSING_PEG_INPUTS": "Not Calculable",
            "ZERO_GROWTH": "Inf (Zero Growth)",
        }

    def test_matches_the_old_window_rules_on_random_cases(self):
        rng = np.random.default_rng(11)
        for case in range(30):
            n = int(rng.integers(5, 400))
            closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
            closes[rng.random(n) < 0.05] = np.nan
            tz = "America/New_York" if case % 2 else None
            history = _history(closes, start="2021-03-01", tz=tz)
            k = int(rng.integers(1, 9))
            first = pd.Timestamp("2021-01-15") + pd.to_timedelta(np.sort(rng.integers(0, 3 * n, k)), unit="D")
            eps = rng.normal(1.0, 1.5, k)  # negative and near-zero values included
            timeline = _timeline(list(zip(first, eps)))
            pe = float(rng.uniform(5, 60))
            assert val.pe_percentile(history, timeline, pe) == _legacy_pe_percentile(
                history, timeline, None, pe
            ), case


# --------------------------------------------------------------------------
# valuation(): the provider-facing composition
# --------------------------------------------------------------------------
class FakeStock:
    def __init__(self, earnings=None, boom=False):
        self.earnings = earnings
        self.boom = boom
        self.earnings_calls = []

    def get_earnings_dates(self, limit=None):
        self.earnings_calls.append(limit)
        if self.boom:
            raise RuntimeError("earnings down")
        return self.earnings


class FakeProvider:
    def __init__(self, info=None, history=None, boom_info=False, boom_history=False):
        self.info = info if info is not None else {}
        self.history = history
        self.boom_info = boom_info
        self.boom_history = boom_history
        self.info_calls = 0
        self.history_calls = []

    def get_info(self, stock):
        self.info_calls += 1
        if self.boom_info:
            raise RuntimeError("info down")
        return self.info

    def fetch_history(self, ticker, period, interval, retries=2, delay=1.0, log=None):
        self.history_calls.append((period, interval))
        if self.boom_history:
            raise RuntimeError("history down")
        return self.history


INFO = {"trailingPE": 16.0, "forwardPE": 14.0, "trailingPegRatio": 1.3, "earningsGrowth": 0.1}


def _five_quarters():
    # ttm = 4 from 2023-01-26 on; closes /4 -> 12.5 .. 35
    return _earnings_df(
        ["2022-04-28", "2022-07-28", "2022-10-27", "2023-01-26"],
        [1.0, 1.0, 1.0, 1.0],
    )


class TestValuation:
    def test_percentile_false_makes_exactly_one_get_info_call(self):
        provider = FakeProvider(INFO)
        stock = FakeStock()
        out = val.valuation(provider, stock, percentile=False)
        assert provider.info_calls == 1
        assert provider.history_calls == []
        assert stock.earnings_calls == []
        assert out["peg"] == 1.3 and out["peg_source"] == "provider"
        assert out["pe_percentile"] is None and out["pe_percentile_reason"] is None

    def test_full_valuation_fetches_five_years_daily_and_eighty_earnings_rows(self):
        provider = FakeProvider(INFO, history=_history(CLOSES, start="2023-02-01"))
        stock = FakeStock(_five_quarters())
        out = val.valuation(provider, stock)
        assert provider.info_calls == 1
        assert provider.history_calls == [("5y", "1d")]
        assert stock.earnings_calls == [80]
        # ttm eps 4.0 for every day: P/E 12.5 .. 35 ; 16.0 sits above 12.5, 15, 17.5? -> 2 of 10
        assert out["pe_percentile"] == pytest.approx(20.0)
        assert out["pe_percentile_reason"] is None
        assert out["pe_ttm"] == 16.0

    def test_given_history_and_earnings_are_not_fetched_again(self):
        provider = FakeProvider(INFO)
        stock = FakeStock()
        out = val.valuation(
            provider, stock, history=_history(CLOSES, start="2023-02-01"), earnings_df=_five_quarters()
        )
        assert provider.history_calls == []
        assert stock.earnings_calls == []
        assert out["pe_percentile"] == pytest.approx(20.0)

    def test_a_name_without_a_trailing_pe_asks_for_nothing_more(self):
        provider = FakeProvider({"forwardPE": 14.0})
        stock = FakeStock()
        out = val.valuation(provider, stock)
        assert provider.info_calls == 1
        assert provider.history_calls == []
        assert stock.earnings_calls == []
        assert out["pe_percentile"] is None
        assert out["pe_percentile_reason"] == "NO_CURRENT_PE_TTM"

    def test_an_etf_info_gives_nothing_and_asks_for_nothing_more(self):
        provider = FakeProvider({"quoteType": "ETF", "totalAssets": 1e11})
        stock = FakeStock()
        out = val.valuation(provider, stock)
        assert (out["pe_ttm"], out["pe_fwd"], out["peg"]) == (None, None, None)
        assert provider.history_calls == [] and stock.earnings_calls == []

    def test_no_price_history_skips_the_earnings_request(self):
        provider = FakeProvider(INFO, history=pd.DataFrame())
        stock = FakeStock(_five_quarters())
        out = val.valuation(provider, stock)
        assert out["pe_percentile_reason"] == "NO_PRICE_HISTORY"
        assert stock.earnings_calls == []

    def test_history_failure_is_a_reason_and_is_logged(self):
        seen = []
        provider = FakeProvider(INFO, boom_history=True)
        stock = FakeStock(_five_quarters())
        out = val.valuation(provider, stock, log=seen.append)
        assert out["pe_percentile"] is None
        assert out["pe_percentile_reason"] == "NO_VALID_PE_HISTORY"
        assert any("P/E Percentile error: history down" in line for line in seen)
        assert stock.earnings_calls == []

    def test_earnings_failure_is_a_reason_and_is_logged(self):
        seen = []
        provider = FakeProvider(INFO, history=_history(CLOSES, start="2023-02-01"))
        out = val.valuation(provider, FakeStock(boom=True), log=seen.append)
        assert out["pe_percentile_reason"] == "NO_EARNINGS_HISTORY"
        assert any("Historical EPS fetch error: earnings down" in line for line in seen)

    def test_too_few_quarters_reason_is_kept(self):
        provider = FakeProvider(INFO, history=_history(CLOSES, start="2023-02-01"))
        df = _earnings_df(["2023-10-26", "2024-01-25"], [1.0, 2.0])
        out = val.valuation(provider, FakeStock(df))
        assert out["pe_percentile_reason"] == "INSUFFICIENT_EARNINGS_HISTORY"

    def test_info_failure_propagates_to_the_caller(self):
        with pytest.raises(RuntimeError, match="info down"):
            val.valuation(FakeProvider(boom_info=True), FakeStock())

    def test_result_carries_every_from_info_key(self):
        out = val.valuation(FakeProvider(INFO), FakeStock(), percentile=False)
        assert set(val.from_info({})) <= set(out)
        assert {"pe_percentile", "pe_percentile_reason"} <= set(out)


# --------------------------------------------------------------------------
# summary_line
# --------------------------------------------------------------------------
class TestSummaryLine:
    def test_the_documented_line(self):
        line = val.summary_line(
            {
                "pe_ttm": 30.2,
                "pe_fwd": 25.1,
                "peg": 1.3,
                "peg_source": "provider",
                "peg_reason": None,
                "pe_percentile": 82.0,
                "pe_percentile_reason": None,
            }
        )
        assert line == "Valuation: P/E 30.20 TTM | 25.10 fwd | PEG 1.30 (provider) | P/E percentile 82.0% (5y, TTM)"

    def test_derived_negative_and_infinite_peg(self):
        base = {"pe_ttm": 10.0, "pe_fwd": None, "peg_reason": None}
        assert "PEG -2.00 (derived)" in val.summary_line({**base, "peg": -2.0, "peg_source": "derived"})
        zero = val.summary_line({**base, "peg": math.inf, "peg_source": "derived", "peg_reason": "ZERO_GROWTH"})
        assert "PEG Inf (derived, zero growth)" in zero

    def test_missing_values_say_why(self):
        line = val.summary_line(
            {
                "pe_ttm": None,
                "pe_fwd": None,
                "peg": None,
                "peg_source": None,
                "peg_reason": "MISSING_PEG_INPUTS",
                "pe_percentile": None,
                "pe_percentile_reason": "NO_CURRENT_PE_TTM",
            }
        )
        assert line == "Valuation: P/E N/A TTM | N/A fwd | PEG Not Calculable | P/E percentile N/A (No TTM P/E)"

    def test_percentile_not_asked_is_left_out(self):
        line = val.summary_line(val.valuation(FakeProvider(INFO), FakeStock(), percentile=False))
        assert line == "Valuation: P/E 16.00 TTM | 14.00 fwd | PEG 1.30 (provider)"


# --------------------------------------------------------------------------
# Module boundary
# --------------------------------------------------------------------------
class TestNoWindow:
    def test_source_imports_no_gui_and_nothing_from_main_or_ui(self):
        tree = ast.parse((ROOT / "core" / "valuation.py").read_text(encoding="utf-8"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules.append(("." * node.level) + (node.module or ""))
        for name in modules:
            top = name.split(".")[0]
            assert top not in {"tkinter", "main", "ui", "matplotlib"}, name
            assert not name.startswith("."), name

    def test_importing_it_loads_no_tkinter_and_no_main_package(self):
        code = (
            "import sys; import core.valuation; "
            "bad = [m for m in sys.modules if m == 'tkinter' or m == '_tkinter' "
            "or m == 'main' or m.startswith('main.') or m == 'ui' or m.startswith('ui.')]; "
            "print(bad)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120, check=True
        )
        assert out.stdout.strip() == "[]"


# --------------------------------------------------------------------------
# analyze_ticker and the CLI
# --------------------------------------------------------------------------
class AnalyzeProvider:
    """Enough of the provider for analyze_ticker; counts every request kind."""

    def __init__(self, info=None, boom_info=False):
        self.info = info if info is not None else dict(INFO)
        self.boom_info = boom_info
        self.info_calls = 0
        self.history_calls = []
        self.stock = FakeStock(_five_quarters())

    def create_ticker(self, symbol):
        return self.stock

    def fetch_history(self, ticker, period, interval, retries=2, delay=1.0, log=None):
        self.history_calls.append((period, interval))
        n = 300 if period == "1y" else 600
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        idx = pd.bdate_range("2023-02-01", periods=n)
        return pd.DataFrame(
            {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1_000_000.0},
            index=idx,
        )

    def get_fast_last_price(self, ticker):
        return 100.0

    def get_info(self, ticker):
        self.info_calls += 1
        if self.boom_info:
            raise RuntimeError("info down")
        return self.info


class TestAnalyzeTicker:
    def test_default_makes_no_valuation_request_and_no_valuation_key(self):
        from core.scan_service import analyze_ticker

        provider = AnalyzeProvider()
        analysis = analyze_ticker(provider, "TEST")
        assert provider.info_calls == 0
        assert provider.stock.earnings_calls == []
        assert provider.history_calls == [("1y", "1d")]
        assert analysis.valuation == {}
        assert "valuation" not in analysis.to_dict()
        assert not any(line.startswith("Valuation") for line in analysis.summary_lines)
        assert len(analysis.summary_lines) == 3

    def test_valuation_true_adds_the_dict_and_one_summary_line(self):
        from core.scan_service import analyze_ticker

        provider = AnalyzeProvider()
        analysis = analyze_ticker(provider, "TEST", valuation=True)
        assert provider.info_calls == 1
        assert ("5y", "1d") in provider.history_calls
        assert provider.stock.earnings_calls == [80]
        assert analysis.valuation["peg"] == 1.3
        assert analysis.valuation["pe_ttm"] == 16.0
        assert analysis.to_dict()["valuation"] == analysis.valuation
        lines = [line for line in analysis.summary_lines if line.startswith("Valuation:")]
        assert lines == [val.summary_line(analysis.valuation)]
        assert analysis.summary_lines[-1] == lines[0]
        assert len(analysis.summary_lines) == 4

    def test_a_failing_info_request_does_not_lose_the_analysis(self):
        from core.scan_service import analyze_ticker

        provider = AnalyzeProvider(boom_info=True)
        seen = []
        analysis = analyze_ticker(provider, "TEST", valuation=True, log=seen.append)
        assert analysis.spot == pytest.approx(100.0)
        assert analysis.valuation == {"error": "info down"}
        assert analysis.summary_lines[-1] == "Valuation: unavailable (info down)"
        assert any("Valuation fetch error: info down" in line for line in seen)

    def test_the_scan_path_never_asks_for_valuation(self, monkeypatch):
        import core.scan_service as scan_mod

        def boom(*args, **kwargs):
            raise AssertionError("valuation must not run inside scan / batch")

        monkeypatch.setattr(val, "valuation", boom)
        monkeypatch.setattr(scan_mod, "scan_option_chains", lambda **kw: scan_mod.ScanResult())
        provider = AnalyzeProvider()
        provider.get_option_expirations = lambda stock: ()
        provider.get_calendar = lambda stock: {}
        analysis, _result = scan_mod.run_ticker_scan(provider, "TEST")
        assert analysis.valuation == {}
        assert provider.history_calls == [("1y", "1d")]
        assert provider.stock.earnings_calls == []


class TestCli:
    def test_analyze_has_the_opt_in_flag_and_it_defaults_off(self):
        from main.cli import build_parser

        parser = build_parser()
        assert parser.parse_args(["analyze", "NVDA"]).valuation is False
        assert parser.parse_args(["analyze", "NVDA", "--valuation"]).valuation is True
        assert parser.parse_args(["analyze", "NVDA", "--valuation", "--json"]).json is True

    @pytest.mark.parametrize("command", ["scan", "batch"])
    def test_scan_and_batch_do_not_take_the_flag(self, command):
        from main.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args([command, "NVDA", "--valuation"])

    def _run(self, monkeypatch, argv, provider):
        import core.data as data_mod
        from main import cli

        monkeypatch.setattr(data_mod, "YFinanceProvider", lambda: provider)
        return cli.main(argv)

    def test_analyze_prints_the_valuation_line(self, monkeypatch, capsys):
        provider = AnalyzeProvider()
        assert self._run(monkeypatch, ["analyze", "NVDA", "--valuation"], provider) == 0
        lines = capsys.readouterr().out.strip().splitlines()
        assert lines[-1].startswith("Valuation: P/E 16.00 TTM | 14.00 fwd | PEG 1.30 (provider)")
        assert provider.info_calls == 1

    def test_analyze_without_the_flag_is_unchanged(self, monkeypatch, capsys):
        provider = AnalyzeProvider()
        assert self._run(monkeypatch, ["analyze", "NVDA"], provider) == 0
        out = capsys.readouterr().out
        assert "Valuation" not in out
        assert len(out.strip().splitlines()) == 3
        assert provider.info_calls == 0

    def test_analyze_json_carries_the_valuation_and_survives_infinity(self, monkeypatch, capsys):
        provider = AnalyzeProvider(info={"trailingPE": 16.0, "forwardPE": 14.0, "earningsGrowth": 0.0})
        assert self._run(monkeypatch, ["analyze", "NVDA", "--valuation", "--json"], provider) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["valuation"]["peg"] is None  # +inf is written as null, as everywhere in this CLI
        assert payload["valuation"]["peg_reason"] == "ZERO_GROWTH"
        assert payload["valuation"]["pe_ttm"] == 16.0

    def test_analyze_json_without_the_flag_has_no_valuation_key(self, monkeypatch, capsys):
        assert self._run(monkeypatch, ["analyze", "NVDA", "--json"], AnalyzeProvider()) == 0
        assert "valuation" not in json.loads(capsys.readouterr().out)


# --------------------------------------------------------------------------
# MarketApp keeps its method names and gives the same answers (needs tkinter)
# --------------------------------------------------------------------------
@pytest.fixture()
def market_app_class():
    pytest.importorskip("tkinter")
    pytest.importorskip("_tkinter")
    from main.app import MarketApp

    return MarketApp


def _bare_app(MarketApp, provider=None, ticker="A"):
    app = object.__new__(MarketApp)
    app.current_ticker = ticker
    app.log = lambda message: app.logged.append(message)
    app.logged = []
    app.valuation_cache = {}
    app._valuation_cache_lock = threading.Lock()
    app.VALUATION_CACHE_DURATION = 3600
    app.VALUATION_CACHE_MAX_ENTRIES = 16
    app.pe_fwd = app.pe_ttm = app.peg_ratio = app.pe_percentile = app.earnings_growth = None
    app.valuation_status = {}
    app.data_provider = provider
    return app


class TestMarketAppDelegates:
    def test_compute_peg_ratio_matches_the_oracle_on_the_grid(self, market_app_class):
        for peg in TestFromInfoMatchesTheOldWindowRules.PEGS:
            for fwd in TestFromInfoMatchesTheOldWindowRules.FWDS:
                for ttm in TestFromInfoMatchesTheOldWindowRules.TTMS:
                    for growth in TestFromInfoMatchesTheOldWindowRules.GROWTHS:
                        app = _bare_app(market_app_class)
                        app.peg_ratio, app.pe_fwd, app.pe_ttm, app.earnings_growth = peg, fwd, ttm, growth
                        app.compute_peg_ratio()
                        old = SimpleNamespace(
                            peg_ratio=peg, pe_fwd=fwd, pe_ttm=ttm, earnings_growth=growth, valuation_status={}
                        )
                        _legacy_compute_peg_ratio(old)
                        assert app.peg_ratio == old.peg_ratio
                        assert app.valuation_status["peg_reason"] == old.valuation_status["peg_reason"]

    def test_status_text_uses_the_shared_labels(self, market_app_class):
        app = _bare_app(market_app_class)
        for reason, label in val.REASONS.items():
            app.valuation_status = {"k": reason}
            assert app._valuation_status_text("k") == label
        app.valuation_status = {"k": "SOMETHING_NEW"}
        assert app._valuation_status_text("k", default="dflt") == "dflt"
        app.valuation_status = {}
        assert app._valuation_status_text("k", default="Loading...") == "Loading..."

    def test_historical_eps_is_cached_for_the_ticker(self, market_app_class):
        app = _bare_app(market_app_class)
        stock = FakeStock(_five_quarters())
        first = app._get_historical_ttm_eps(stock)
        second = app._get_historical_ttm_eps(stock)
        assert first[1] is None and second[1] is None
        assert stock.earnings_calls == [80]
        assert list(first[0]["ttm_eps"]) == [4.0]

    def test_historical_eps_failure_logs_and_gives_no_history(self, market_app_class):
        app = _bare_app(market_app_class)
        assert app._get_historical_ttm_eps(FakeStock(boom=True)) == (None, "NO_EARNINGS_HISTORY")
        assert any("Historical EPS fetch error: earnings down" in m for m in app.logged)

    def test_calculate_pe_percentile_matches_the_module(self, market_app_class):
        provider = FakeProvider(INFO, history=_history(CLOSES, start="2023-02-01"))
        app = _bare_app(market_app_class, provider)
        app.pe_ttm = 16.0
        app.calculate_pe_percentile(FakeStock(_five_quarters()))
        assert app.pe_percentile == pytest.approx(20.0)
        assert app.valuation_status["pe_percentile_reason"] is None

    def test_calculate_pe_percentile_reasons_and_requests(self, market_app_class):
        stock = FakeStock(_five_quarters())
        app = _bare_app(market_app_class, FakeProvider(INFO, history=pd.DataFrame()))
        app.calculate_pe_percentile(stock)  # no P/E: nothing is fetched
        assert app.valuation_status["pe_percentile_reason"] == "NO_CURRENT_PE_TTM"
        assert app.data_provider.history_calls == []
        app.pe_ttm = 16.0
        app.calculate_pe_percentile(stock)  # empty history: no earnings request
        assert app.valuation_status["pe_percentile_reason"] == "NO_PRICE_HISTORY"
        assert stock.earnings_calls == []
        app.data_provider = FakeProvider(INFO, boom_history=True)
        app.calculate_pe_percentile(stock)
        assert app.pe_percentile is None
        assert app.valuation_status["pe_percentile_reason"] == "NO_VALID_PE_HISTORY"
        assert any("P/E Percentile error: history down" in m for m in app.logged)

    def test_get_info_publishes_the_same_values_as_the_module(self, market_app_class):
        info = {"trailingPE": 16.0, "forwardPE": 14.0, "earningsGrowth": 0.2}
        provider = FakeProvider(info, history=_history(CLOSES, start="2023-02-01"))
        app = _bare_app(market_app_class, provider)
        app._fundamental_request_id = 1
        app.stock = FakeStock(_five_quarters())
        app.update_pe_display = lambda: None
        pending = []
        app.root = SimpleNamespace(after=lambda delay, callback: pending.append(callback))
        app.get_info(app.stock, "A", 1)
        pending.pop()()
        expected = val.from_info(info)
        assert app.pe_ttm == expected["pe_ttm"]
        assert app.pe_fwd == expected["pe_fwd"]
        assert app.peg_ratio == pytest.approx(expected["peg"])
        assert app.earnings_growth == expected["earnings_growth"]
        assert app.pe_percentile == pytest.approx(20.0)
        assert app.valuation_status["peg_reason"] is None

    def test_as_naive_helper_is_still_on_the_class(self, market_app_class):
        s = pd.Series(pd.date_range("2024-01-01", periods=3, freq="D").astype("datetime64[s]"))
        assert str(market_app_class._as_naive_datetime64_us(s).dtype) == "datetime64[us]"
