"""Offline regression coverage for the September reliability review."""
from datetime import datetime, timezone
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
from matplotlib.figure import Figure

from core.expiry_time import remaining_years
from core.graph_service import categorize_peers
from core.scan_service import resolve_dividend_yield, scan_option_chains
from core.stock_graph import StockGraph, GraphEdge, build_default_graph
from main.app import MarketApp
from ui.chart import apply_tick_labels, draw_earnings_markers
from ui.options_3d import filter_rows_for_plot, build_plotly_figure
from ui.prefs import load_prefs, save_prefs
from ui.storage import config_dir
from ui.watchlist import load_watchlist, save_watchlist


def test_dividend_source_failure_continues_to_info():
    provider = SimpleNamespace(get_fast_info=Mock(side_effect=RuntimeError("offline")),
                               get_info=Mock(return_value={"dividendYield": 2}))
    status = {}
    assert resolve_dividend_yield(provider, object(), status=status) == .02
    assert status["source"] == "info.dividendYield"
    assert status["errors"][0]["source"] == "fast_info"


def test_missing_dividend_is_distinguishable_from_confirmed_zero():
    provider = SimpleNamespace(get_fast_info=Mock(return_value={}), get_info=Mock(return_value={}))
    status = {}
    assert resolve_dividend_yield(provider, object(), status=status) == 0
    assert status["source"] == "missing"
    provider.get_info.return_value = {"dividendYield": 0}
    assert resolve_dividend_yield(provider, object(), status=status) == 0
    assert status["source"] == "info.dividendYield"


@pytest.mark.parametrize("stamp,expected_sessions", [
    ("2026-09-11T09:30:00-04:00", 1),
    ("2026-09-11T15:59:00-04:00", 1 / 390),
    ("2026-09-11T16:00:00-04:00", 0),
    ("2026-09-11T18:00:00-04:00", 0),
])
def test_expiry_clock_tracks_remaining_session(stamp, expected_sessions):
    assert remaining_years("2026-09-11", pd.Timestamp(stamp)) == pytest.approx(expected_sessions / 252)


def test_expiry_calendar_excludes_holidays_and_counts_early_close():
    # July 3 is the observed Independence Day holiday in 2026.
    assert remaining_years("2026-07-03", pd.Timestamp("2026-07-02T16:00:00-04:00")) == 0
    # Black Friday closes at 13:00 ET: 3.5 hours, not the regular 6.5.
    assert remaining_years("2026-11-27", pd.Timestamp("2026-11-27T09:30:00-05:00")) == pytest.approx(3.5 / 6.5 / 252)
    assert remaining_years("2026-11-27", pd.Timestamp("2026-11-27T13:00:00-05:00")) == 0


def test_expiry_clock_is_timezone_independent_and_requires_aware_input():
    utc = pd.Timestamp("2026-09-11T18:00:00Z")
    assert remaining_years("2026-09-11", utc) == remaining_years("2026-09-11", utc.tz_convert("Asia/Tokyo"))
    with pytest.raises(ValueError, match="timezone-aware"):
        remaining_years("2026-09-11", datetime(2026, 9, 11))


def test_ask_based_breakeven_and_pop_respond_to_spread_at_same_mid():
    def scan(bid, ask):
        calls = pd.DataFrame({"strike": [100.0], "bid": [bid], "ask": [ask],
                              "volume": [100], "openInterest": [500], "impliedVolatility": [.3]})
        provider = SimpleNamespace(get_option_chain=lambda *_: SimpleNamespace(calls=calls, puts=calls.iloc[:0]))
        return scan_option_chains(data_provider=provider, stock=object(), spot=100,
                                  dates=["2026-10-16"], valuation_time=pd.Timestamp("2026-09-11T14:00:00Z"),
                                  dividend_yield=0, short_rate=.04, long_rate=.04,
                                  ewma_vol=.3, use_american_greeks=False).rows[0]
    narrow, wide = scan(4.9, 5.1), scan(4.6, 5.4)
    assert narrow.mid == wide.mid == 5
    assert narrow.breakeven == 105.1
    assert wide.breakeven == 105.4
    assert narrow.pop > wide.pop
    assert narrow.to_dict()["bid"] == 4.9
    assert narrow.to_dict()["category"] in ("Under", "Over", "Fair")
    assert "risk-neutral" in narrow.to_dict()["probability_model"]


def test_fair_plot_rows_are_independent_of_over_filter():
    rows = [{"verdict": verdict, "is_good": False, "is_earnings": earnings}
            for verdict in ("Fair", "Over") for earnings in (False, True)]
    result = filter_rows_for_plot(rows, show_under=False, show_over=False,
                                  show_earn_under=False, show_earn_over=False, show_fair=True)
    assert [r["category"] for r in result] == ["fair", "earnings_fair"]
    assert filter_rows_for_plot(rows, show_under=False, show_over=False,
                                show_earn_under=False, show_earn_over=False, show_fair=False) == []
    fig = build_plotly_figure("TEST", "CALL", [1, 2], [100, 105], [-.1, -.2],
                              ["2026-09-11"] * 2, [10, 20], ["fair", "earnings_fair"])
    assert all("no tradeable edge" in text and "sell-side edge" not in text for text in fig.data[0].text)


def test_peer_payload_preserves_multiple_edges_and_direction():
    graph = StockGraph()
    graph.add_edge(GraphEdge("A", "B", "SUPPLIER_TO"))
    graph.add_edge(GraphEdge("A", "B", "COMPETITOR", bidirectional=True))
    peers = categorize_peers(graph, "B")
    assert set(peers["peer_categories"]) == {"SUPPLIER_TO", "COMPETITOR"}
    supplier = peers["peer_categories"]["SUPPLIER_TO"][0]
    assert (supplier["source"], supplier["target"], supplier["direction"]) == ("A", "B", "incoming")
    for depth in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            graph.get_neighbors("B", depth=depth)


def test_cached_graph_import_status_is_exposed():
    graph = build_default_graph()
    assert graph.import_status["skipped"] is False
    assert graph.import_status["edges_added"] > 0
    assert "Sectivia" in graph.import_status["attribution"]


def test_preferences_use_user_storage_and_preserve_empty_watchlist(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_CONFIG_DIR", str(tmp_path / "config"))
    root = config_dir()
    assert save_prefs(root, show_fib=True)
    assert load_prefs(root)["show_fib"] is True
    assert save_watchlist(root, []) == []
    assert load_watchlist(root) == []


def test_failed_atomic_replace_keeps_previous_settings(tmp_path, monkeypatch):
    assert save_prefs(str(tmp_path), show_fib=True)
    monkeypatch.setattr("ui.storage.os.replace", Mock(side_effect=OSError("read-only")))
    with pytest.warns(RuntimeWarning, match="Could not save"):
        assert save_prefs(str(tmp_path), show_fib=False) is False
    assert load_prefs(str(tmp_path))["show_fib"] is True
    with pytest.raises(OSError):
        save_watchlist(str(tmp_path), ["TEST"])
    assert list(tmp_path.glob(".settings-*")) == []


def test_chart_keeps_friday_date_and_skips_old_earnings():
    ax = Figure().subplots()
    times = pd.DatetimeIndex(["2026-09-11T15:55:00-04:00"])
    apply_tick_labels(ax, times, "5d")
    assert [x.get_text() for x in ax.get_xticklabels()] == ["2026-09-11"]
    draw_earnings_markers(ax, times, [0], ["2026-01-01", "2026-09-12"])
    assert len(ax.lines) == 0
    draw_earnings_markers(ax, times, [0], ["2026-09-11"])
    assert len(ax.lines) == 1


def _app():
    app = object.__new__(MarketApp)
    app.current_ticker = "A"
    app.stock = SimpleNamespace(ticker="A")
    app._fundamental_request_id = 1
    app._options_request_id = 1
    app._scan_lock = threading.Lock()
    app.log = Mock()
    app.pending = []
    app.root = SimpleNamespace(after=lambda delay, callback: app.pending.append(callback))
    app.update_pe_display = Mock()
    app.pe_fwd = app.pe_ttm = app.peg_ratio = app.pe_percentile = app.earnings_growth = None
    app.valuation_status = {}
    return app


def test_old_fundamentals_cannot_publish_after_new_request(monkeypatch):
    app = _app()
    app.data_provider = SimpleNamespace(get_info=lambda stock: {"forwardPE": 12 if stock.ticker == "A" else 24})
    monkeypatch.setattr(MarketApp, "calculate_pe_percentile", lambda self, stock: None)
    monkeypatch.setattr(MarketApp, "compute_peg_ratio", lambda self: None)
    app.get_info(app.stock, "A", 1)
    app.current_ticker = "B"
    app.stock = SimpleNamespace(ticker="B")
    app._fundamental_request_id = 2
    app.get_info(app.stock, "B", 2)
    # Complete B before A; neither worker changed shared valuation fields.
    assert app.pe_fwd is None
    app.pending.pop()()
    app.pending.pop()()
    assert app.pe_fwd == 24
    app.update_pe_display.assert_called_once()


@pytest.mark.parametrize("invalidate", ["ticker", "request", "closed", "window"])
def test_stale_scan_callbacks_do_not_touch_new_window(monkeypatch, invalidate):
    app = _app()
    app.tree = SimpleNamespace(winfo_exists=lambda: True, insert=Mock())
    old_tree = app.tree
    app.scan_data = []
    def scan(**kwargs):
        kwargs["on_ui_batch"]([(("A",), "green")])
        return SimpleNamespace(scan_buf=[{"ticker": "A"}])
    monkeypatch.setattr("main.app.scan_option_chains", scan)
    app.fetch_options_batch({}, "A", old_tree, 1)
    if invalidate == "ticker":
        app.current_ticker = "B"
    elif invalidate == "request":
        app._options_request_id += 1
    elif invalidate == "closed":
        old_tree.winfo_exists = lambda: False
    else:
        app.tree = SimpleNamespace(winfo_exists=lambda: True, insert=Mock())
    for callback in app.pending:
        callback()
    old_tree.insert.assert_not_called()
    assert app.scan_data == []


def test_current_scan_publishes_rows_and_buffer(monkeypatch):
    app = _app()
    app.tree = SimpleNamespace(winfo_exists=lambda: True, insert=Mock())
    def scan(**kwargs):
        kwargs["on_ui_batch"]([(("A",), "green")])
        return SimpleNamespace(scan_buf=[{"ticker": "A"}])
    monkeypatch.setattr("main.app.scan_option_chains", scan)
    app.fetch_options_batch({}, "A", app.tree, 1)
    for callback in app.pending:
        callback()
    app.tree.insert.assert_called_once()
    assert app.scan_data == [{"ticker": "A"}]


def test_mixed_news_timestamps_are_aware_and_sorted():
    app = _app()
    app._sent_cache_lock = threading.Lock()
    app.sent_cache = {}
    app.SENT_CACHE_DURATION = 60
    app.SENT_CACHE_MAX_TICKERS = 5
    app.headline_limit = 10
    app.use_sentiment = False
    app.get_google_news_rss = lambda ticker: [{"title": "RSS", "published": datetime(2026, 9, 12)}]
    yahoo = SimpleNamespace(news=[{"title": "Yahoo", "providerPublishTime": 1789171200}])
    _, items = app.calculate_sentiment("A", yahoo)
    assert len(items) == 2
    assert all(item["published"].tzinfo is not None for item in items)
    assert items[0]["published"] >= items[1]["published"]


def test_rss_certificate_failure_is_isolated_and_verification_enabled(monkeypatch):
    app = _app()
    app.headline_limit = 10
    get = Mock(side_effect=requests.exceptions.SSLError("invalid certificate"))
    monkeypatch.setattr("main.app.requests.get", get)
    assert app.get_google_news_rss("TEST") == []
    assert all(call.kwargs.get("verify", True) is True for call in get.call_args_list)


def test_scan_inputs_are_captured_before_worker_starts(monkeypatch):
    app = _app()
    app.tree = SimpleNamespace(winfo_exists=lambda: True)
    app.current_price = 100
    app.projected_earnings = [pd.Timestamp("2026-10-01")]
    app.all_exps = ["2026-10-16"]
    app.data_provider = object()
    thread = Mock()
    monkeypatch.setattr("main.app.threading.Thread", thread)
    dates = ["2026-10-16"]
    app._start_options_scan(dates)
    inputs, ticker, tree, token = thread.call_args.kwargs["args"]
    app.current_ticker, app.current_price = "B", 200
    app.all_exps.clear()
    dates.clear()
    assert (ticker, inputs["spot"], inputs["dates"], inputs["all_exps"]) == ("A", 100, ("2026-10-16",), ("2026-10-16",))


def test_closed_expiration_window_ignores_worker_results():
    app = _app()
    app.exp_list = SimpleNamespace(winfo_exists=lambda: False)
    app.all_exps = []
    app.update_exp_list = Mock()
    app.data_provider = SimpleNamespace(get_option_expirations=lambda stock: ["2026-10-16"])
    app.load_expirations(app.stock, "A", app.exp_list)
    app.pending.pop()()
    assert app.all_exps == []
    app.update_exp_list.assert_not_called()


@pytest.mark.parametrize("version,success", [
    ("v2.4.0", True), ("1.8", True), ("v2.4.0-rc.1", True),
    ('2.4.0"; touch injected; #', False), ("$(touch injected)", False),
    ("2.4.0\ntag=evil", False), ("../../bad", False), ("", False),
])
def test_release_version_is_validated_as_data(tmp_path, version, success):
    import os
    from pathlib import Path
    import subprocess
    script = Path(__file__).resolve().parents[1] / "scripts/resolve_release_tag.sh"
    output = tmp_path / "output"
    process = subprocess.run(["bash", str(script)], cwd=tmp_path,
                             env={**os.environ, "GITHUB_EVENT_NAME": "workflow_dispatch",
                                  "DISPATCH_VERSION": version, "GITHUB_OUTPUT": str(output)},
                             capture_output=True, text=True)
    assert (process.returncode == 0) == success
    assert not (tmp_path / "injected").exists()
    if success:
        assert output.read_text() == f"tag={version}\n"
    else:
        assert not output.exists()


def test_graph_cli_rejects_nonpositive_depth():
    import argparse
    from main.graph_cli import add_commands
    parser = argparse.ArgumentParser()
    add_commands(parser.add_subparsers())
    for command in (["graph", "show", "A", "--depth", "0"],
                    ["graph", "export", "A", "--html", "test.html", "--depth", "-1"]):
        with pytest.raises(SystemExit):
            parser.parse_args(command)


def test_unexpected_news_failure_does_not_abort_chart_refresh():
    app = _app()
    app.calculate_sentiment = Mock(side_effect=RuntimeError("model unavailable"))
    assert app._chart_news("A", app.stock) == (None, [])
    app.log.assert_called_once()
