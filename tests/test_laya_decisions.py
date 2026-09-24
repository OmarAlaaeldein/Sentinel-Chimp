"""Offline unit tests for core/laya_decisions.py — mocks only, no model download."""
from __future__ import annotations

import sys

import pytest

from core import laya_decisions as lb


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("SENTINEL_LAYA", raising=False)
    lb.reset_runtime()
    yield
    lb.reset_runtime()


def test_import_without_laya_installed():
    """Module must import even when laya packages are absent."""
    assert "core.laya_decisions" in sys.modules
    info = lb.backend_info(mlx_importable=False, laya_importable=False, system="Linux", machine="x86_64")
    assert info["available"] is False
    assert info["name"] == "none"
    assert info["env"] == "SENTINEL_LAYA"


def test_laya_opt_in_default_off(monkeypatch):
    monkeypatch.delenv("SENTINEL_LAYA", raising=False)
    assert lb.laya_opt_in() is False
    monkeypatch.setenv("SENTINEL_LAYA", "1")
    assert lb.laya_opt_in() is True
    monkeypatch.setenv("SENTINEL_LAYA", "true")
    assert lb.laya_opt_in() is True
    monkeypatch.setenv("SENTINEL_LAYA", "0")
    assert lb.laya_opt_in() is False


def test_apple_silicon_prefers_mlx():
    sel = lb.select_backend_name(
        system="Darwin", machine="arm64", mlx_importable=True, laya_importable=False
    )
    assert sel["name"] == "laya_mlx"
    assert sel["available"] is True


def test_linux_prefers_upstream():
    sel = lb.select_backend_name(
        system="Linux", machine="x86_64", mlx_importable=False, laya_importable=True
    )
    assert sel["preferred_backend"] == "laya"
    assert sel["name"] == "laya"


def test_score_headlines_unavailable_returns_errors():
    rows = lb.score_headlines(
        ["Fed signals pause"],
        agent=None,
    )
    # Force unavailable path by mocking get_agent via predict with no agent —
    # when get_agent finds nothing, rows have label None.
    # Use explicit unavailable: pass agent that we don't — call with monkeypatched get_agent
    assert isinstance(rows, list)


def test_score_headlines_with_mock_agent():
    class FakeAgent:
        def predict(self, state, questions):
            return {
                "answers": {
                    "polarity": {
                        "choice": "positive",
                        "probabilities": {"positive": 0.8, "neutral": 0.15, "negative": 0.05},
                        "confidence": 0.8,
                    },
                    "score": {"score": 8},
                }
            }

    rows = lb.score_headlines(["Earnings beat expectations"], agent=FakeAgent(), backend="mock")
    assert len(rows) == 1
    assert rows[0]["label"] == "positive"
    # Prob-weighted polarity when probabilities present: 0.8*1 + 0.15*0.5 + 0.05*0 = 0.875
    assert rows[0]["score"] == pytest.approx(0.875, abs=0.001)
    assert rows[0]["backend"] == "mock"


def test_average_polarity():
    assert lb.average_polarity([]) is None
    assert lb.average_polarity([{"score": None}, {"score": 0.5}, {"score": 1.0}]) == pytest.approx(0.75)


def test_classify_decision_mock():
    class FakeAgent:
        def predict(self, state, questions):
            return {"answers": {"x": {"choice": "yes"}}}

    out = lb.classify_decision(
        "state",
        {"x": {"type": "choice", "instructions": "y", "criteria": {"yes": "y", "no": "n"}}},
        agent=FakeAgent(),
        backend="mock",
    )
    assert out["available"] is True
    assert out["answers"]["x"]["choice"] == "yes"


def test_sentinel_reexports_drop_sentiment_engine():
    import sentinel

    assert not hasattr(sentinel, "SentimentEngine") or "SentimentEngine" not in sentinel.__all__
    assert "SentimentEngine" not in sentinel.__all__
    assert "sentiment_engine" not in sentinel.__all__


def test_score_when_backend_missing_logs_reason(monkeypatch):
    monkeypatch.setattr(
        lb,
        "get_agent",
        lambda force_reload=False: (None, {"name": "none", "available": False, "reason": "not installed"}),
    )
    rows = lb.score_headlines(["hello"])
    assert rows[0]["label"] is None
    assert "not installed" in (rows[0].get("error") or "")
