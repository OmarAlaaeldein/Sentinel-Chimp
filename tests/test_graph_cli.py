"""Tests for sentinel graph CLI commands and machine-readable output."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from main.cli import main


def test_cli_graph_summary_json(capsys):
    assert main(["graph", "show", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["data"]["action"] == "summary"
    assert out["data"]["total_nodes"] >= 20
    assert "NVDA" in out["data"]["tickers"]


def test_cli_graph_show_ticker_json(capsys):
    assert main(["graph", "show", "NVDA", "--depth", "1", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["data"]["ticker"] == "NVDA"
    assert out["data"]["node"]["sector"] == "Semiconductors"
    assert len(out["data"]["connections"]) > 0


def test_cli_graph_peers_json(capsys):
    assert main(["graph", "peers", "AMD", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["data"]["ticker"] == "AMD"
    categories = out["data"]["peer_categories"]
    assert "COMPETITOR" in categories
    assert any(p["ticker"] == "NVDA" for p in categories["COMPETITOR"])


def test_cli_graph_export_html(tmp_path, capsys):
    out_file = tmp_path / "graph_export.html"
    assert main(["graph", "export", "NVDA", "--html", str(out_file), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["data"]["action"] == "export"
    assert Path(out["data"]["path"]).exists()


def test_cli_graph_divergence_mocked(monkeypatch, capsys):
    import core.data
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Close": np.linspace(100, 110, 10)}, index=dates)

    class FakeProvider:
        def create_ticker(self, sym):
            return sym

        def fetch_history(self, ticker, period="1mo", interval="1d"):
            return df

    monkeypatch.setattr(core.data, "YFinanceProvider", FakeProvider)

    assert main(["graph", "divergence", "NVDA", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["data"]["ticker"] == "NVDA"
    assert "divergences" in out["data"]


def test_cli_graph_unknown_ticker(capsys):
    assert main(["graph", "show", "NONEXISTENT_TICKER_XYZ", "--json"]) == 4
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "error"
    assert "not found" in out["error"]["message"].lower()
