"""P/E percentile merge_asof must tolerate mismatched datetime units."""
import numpy as np
import pandas as pd

from main.app import MarketApp


def test_as_naive_datetime64_us_unifies_units():
    s = pd.Series(pd.date_range("2024-01-01", periods=3, freq="D").astype("datetime64[s]"))
    u = pd.Series(pd.date_range("2024-01-01", periods=3, freq="D").astype("datetime64[us]"))
    sn = MarketApp._as_naive_datetime64_us(s)
    un = MarketApp._as_naive_datetime64_us(u)
    assert str(sn.dtype) == "datetime64[us]"
    assert str(un.dtype) == "datetime64[us]"


def test_merge_asof_s_vs_us_does_not_raise():
    hist = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D").astype("datetime64[s]"),
            "Close": np.linspace(100, 110, 10),
        }
    )
    eps = pd.DataFrame(
        {
            "report_date": pd.to_datetime(["2023-12-15", "2024-01-05"]).astype("datetime64[us]"),
            "ttm_eps": [4.0, 4.2],
        }
    )
    hist["date"] = MarketApp._as_naive_datetime64_us(hist["date"])
    eps["report_date"] = MarketApp._as_naive_datetime64_us(eps["report_date"])
    merged = pd.merge_asof(
        hist.sort_values("date"),
        eps.sort_values("report_date"),
        left_on="date",
        right_on="report_date",
        direction="backward",
    )
    assert len(merged) == 10
    assert merged["ttm_eps"].notna().any()
