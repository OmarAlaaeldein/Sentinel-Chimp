"""Offline tests for `sentinel.py batch` (one process, shared provider)."""
from types import SimpleNamespace

import pytest

from main import cli


class _Row:
    def __init__(self, verdict="Under", category=None):
        self.date = "2026-01-16"
        self.type = "PUT"
        self.strike = 100.0
        self.mid = 1.0
        self.fair = 2.0
        self.ev_at_ask = 1.0
        self.edge_pct = 0.5
        self.delta = -0.2
        self.iv = 0.4
        self.oi = 1000
        self.verdict = verdict
        self.category = category or verdict

    def to_dict(self):
        return {"date": self.date, "type": self.type, "strike": self.strike,
                "mid": self.mid, "fair": self.fair, "ev_at_ask": self.ev_at_ask,
                "edge_pct": self.edge_pct, "delta": self.delta, "iv": self.iv,
                "oi": self.oi, "verdict": self.verdict, "category": self.category}


class _Analysis:
    summary_lines = ["TEST summary line"]

    def to_dict(self):
        return {"ticker": "TEST"}


def _make_result(errors=None):
    return SimpleNamespace(
        rows=[_Row(), _Row(verdict="Over")],
        forecast_vol=0.3, dividend_yield=0.0, dividend_status="ok",
        rules_log=[], requested_expiries=2, errors=errors or [],
    )


@pytest.fixture()
def fake_scan(monkeypatch):
    """Patch the data + scan-service seams cmd_batch imports."""
    import core.data as data_mod
    import core.scan_service as scan_mod

    calls = []

    def fake_run(provider, ticker, **kw):
        calls.append((ticker, kw.get("under_only"), kw.get("max_expiries")))
        if ticker == "BAD":
            raise RuntimeError("rate limited")
        return _Analysis(), _make_result()

    monkeypatch.setattr(data_mod, "YFinanceProvider", lambda: object())
    monkeypatch.setattr(scan_mod, "run_ticker_scan", fake_run)
    return calls


def test_batch_writes_artifacts_for_every_ticker(tmp_path, capsys, fake_scan):
    rc = cli.main(["batch", "TEST", "TEST2", "--max-expiries", "2",
                   "--under-only", "--out-dir", str(tmp_path), "--pause", "0"])
    assert rc == 0
    for ticker in ("TEST", "TEST2"):
        assert (tmp_path / f"sentinel_analyze_{ticker}.txt").exists()
        assert (tmp_path / f"sentinel_scan_{ticker}.json").exists()
        assert (tmp_path / f"sentinel_scan_{ticker}.txt").exists()
    payload = json_load(tmp_path / "sentinel_scan_TEST.json")
    assert payload["status"] == "ok"
    out = capsys.readouterr().out
    assert "BATCH TEST ok under=1 contracts=2 status=ok" in out
    assert fake_scan == [("TEST", True, 2), ("TEST2", True, 2)]


def test_batch_continues_past_a_failed_ticker(tmp_path, capsys, fake_scan):
    rc = cli.main(["batch", "BAD", "TEST", "--out-dir", str(tmp_path), "--pause", "0"])
    assert rc == 1  # one failure, one success
    out = capsys.readouterr().out
    assert "BATCH BAD error" in out
    assert "BATCH TEST ok" in out
    assert not (tmp_path / "sentinel_scan_BAD.json").exists()
    assert (tmp_path / "sentinel_scan_TEST.json").exists()


def test_batch_all_failed_is_exit_two(tmp_path, capsys, fake_scan):
    rc = cli.main(["batch", "BAD", "BAD", "--out-dir", str(tmp_path), "--pause", "0"])
    assert rc == 2


def json_load(path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))
