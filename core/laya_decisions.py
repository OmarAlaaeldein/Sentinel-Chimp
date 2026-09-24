"""Optional cross-OS Laya adapter for typed local decisions / headline polarity.

Self-contained in Sentinel — no private umbrella imports.

OS matrix:
  - Darwin + Apple Silicon → prefer ``laya_mlx`` / PyPI ``laya-mlx`` (no PyTorch)
  - Linux / Windows / Intel Mac → prefer upstream ``laya`` (torch/transformers)
  - Neither importable → unavailable; callers skip scoring

Opt-in: set ``SENTINEL_LAYA=1`` (or enable ``MarketApp.use_laya``). Default off.
Lite / release builds never require Laya. Heavy imports are lazy — this module
never hard-fails at import time.
"""
from __future__ import annotations

import os
import platform
import sys
from collections.abc import Callable
from typing import Any

HEADLINE_QUESTIONS: dict[str, Any] = {
    "polarity": {
        "type": "choice",
        "instructions": (
            "For an equity / markets news headline, which tone best fits the "
            "likely near-term market reaction for the named asset or market?"
        ),
        "criteria": {
            "positive": "bullish, constructive, or risk-on for the asset or market",
            "neutral": "mixed, factual, or unclear directional tone",
            "negative": "bearish, risk-off, or adverse for the asset or market",
        },
    },
    "score": {
        "type": "score",
        "instructions": (
            "How bullish is this headline for the asset or market "
            "(0=very bearish, 10=very bullish)?"
        ),
        "criteria": [
            "very bearish",
            "bearish",
            "moderately bearish",
            "slightly bearish",
            "lean negative",
            "neutral",
            "lean positive",
            "slightly bullish",
            "moderately bullish",
            "bullish",
            "very bullish",
        ],
    },
}

_TRUE = frozenset({"1", "true", "yes", "on"})
_RUNTIME: dict[str, Any] = {"agent": None, "backend": None, "error": None}


def laya_opt_in(env: dict[str, str] | None = None) -> bool:
    """True when SENTINEL_LAYA enables optional scoring (default off)."""
    raw = (env if env is not None else os.environ).get("SENTINEL_LAYA", "0")
    return str(raw).strip().lower() in _TRUE


def detect_platform(
    *,
    system: str | None = None,
    machine: str | None = None,
) -> dict[str, Any]:
    """Pure OS/arch facts used for backend selection (no imports)."""
    sys_name = (system if system is not None else platform.system()).strip()
    mach = (machine if machine is not None else platform.machine()).strip()
    mach_l = mach.lower()
    apple_silicon = sys_name == "Darwin" and mach_l in {"arm64", "aarch64"}
    preferred = "laya_mlx" if apple_silicon else "laya"
    return {
        "os": sys_name or "unknown",
        "machine": mach or "unknown",
        "python": sys.version.split()[0],
        "apple_silicon": apple_silicon,
        "preferred_backend": preferred,
    }


def select_backend_name(
    *,
    system: str | None = None,
    machine: str | None = None,
    mlx_importable: bool | None = None,
    laya_importable: bool | None = None,
) -> dict[str, Any]:
    """Choose backend name without loading models. Pure when import flags passed."""
    plat = detect_platform(system=system, machine=machine)
    preferred = plat["preferred_backend"]

    def _can_import(mod: str) -> bool:
        try:
            __import__(mod)
            return True
        except Exception:
            return False

    mlx_ok = mlx_importable if mlx_importable is not None else _can_import("laya_mlx")
    up_ok = laya_importable if laya_importable is not None else _can_import("laya")

    if preferred == "laya_mlx":
        if mlx_ok:
            return {
                **plat,
                "name": "laya_mlx",
                "available": True,
                "reason": "Apple Silicon + laya_mlx importable",
            }
        if up_ok:
            return {
                **plat,
                "name": "laya",
                "available": True,
                "reason": "laya_mlx missing; falling back to upstream laya on this Mac",
            }
        return {
            **plat,
            "name": "none",
            "available": False,
            "reason": "install laya-mlx (Apple Silicon, macOS 14+) for local typed decisions",
        }

    if up_ok:
        return {**plat, "name": "laya", "available": True, "reason": "upstream laya importable"}
    if mlx_ok and plat["os"] == "Darwin":
        return {
            **plat,
            "name": "laya_mlx",
            "available": True,
            "reason": "upstream laya missing; laya_mlx importable on Darwin",
        }
    return {
        **plat,
        "name": "none",
        "available": False,
        "reason": "install PyPI package `laya` (Linux/Windows/Intel Mac) for local typed decisions",
    }


def backend_info(
    *,
    system: str | None = None,
    machine: str | None = None,
    mlx_importable: bool | None = None,
    laya_importable: bool | None = None,
) -> dict[str, Any]:
    """Status dict: name, available, reason, os, opt_in, machine, preferred_backend."""
    sel = select_backend_name(
        system=system,
        machine=machine,
        mlx_importable=mlx_importable,
        laya_importable=laya_importable,
    )
    return {
        "name": sel["name"],
        "available": bool(sel["available"]),
        "reason": sel["reason"],
        "os": sel["os"],
        "machine": sel["machine"],
        "python": sel.get("python"),
        "apple_silicon": sel.get("apple_silicon"),
        "preferred_backend": sel.get("preferred_backend"),
        "opt_in": laya_opt_in(),
        "env": "SENTINEL_LAYA",
    }


def reset_runtime() -> None:
    """Drop cached agent (tests / after uninstall)."""
    _RUNTIME["agent"] = None
    _RUNTIME["backend"] = None
    _RUNTIME["error"] = None


def _default_mlx_repo() -> str:
    return os.environ.get("SENTINEL_LAYA_MLX_MODEL", "aac6fef/laya-mlx")


def _default_upstream_repo() -> str:
    return os.environ.get("SENTINEL_LAYA_MODEL", "convaiinnovations/laya")


def _load_agent(backend: str) -> Any:
    if backend == "laya_mlx":
        import laya_mlx as laya_mod

        return laya_mod.load(_default_mlx_repo())
    if backend == "laya":
        import laya as laya_mod

        router_cls = getattr(laya_mod, "Router", None)
        if router_cls is not None:
            return router_cls()
        return laya_mod.load(_default_upstream_repo())
    raise RuntimeError(f"unknown Laya backend: {backend}")


def get_agent(*, force_reload: bool = False) -> tuple[Any | None, dict[str, Any]]:
    """Lazy-load the selected backend agent. Returns (agent_or_None, info)."""
    info = backend_info()
    if not info["available"]:
        return None, info
    if force_reload:
        reset_runtime()
    if _RUNTIME["agent"] is not None and _RUNTIME["backend"] == info["name"]:
        return _RUNTIME["agent"], info
    try:
        agent = _load_agent(info["name"])
    except Exception as exc:
        info = {**info, "available": False, "reason": f"load failed: {exc}"}
        _RUNTIME["error"] = str(exc)
        return None, info
    _RUNTIME["agent"] = agent
    _RUNTIME["backend"] = info["name"]
    _RUNTIME["error"] = None
    return agent, info


def _predict(agent: Any, state: str, questions: dict[str, Any]) -> dict[str, Any]:
    if hasattr(agent, "predict"):
        return agent.predict(state, questions)
    if hasattr(agent, "system_one"):
        return agent.system_one(state, questions)
    raise TypeError("Laya agent has neither predict nor system_one")


def _choice_probs(answer: Any) -> dict[str, float]:
    if not isinstance(answer, dict):
        return {}
    probs = answer.get("probabilities") or answer.get("probs") or {}
    if isinstance(probs, dict):
        out: dict[str, float] = {}
        for key, val in probs.items():
            try:
                out[str(key).lower()] = float(val)
            except (TypeError, ValueError):
                continue
        return out
    return {}


def _label_from_choice(answer: Any) -> str:
    if not isinstance(answer, dict):
        return "neutral"
    choice = answer.get("choice") or answer.get("label") or "neutral"
    return str(choice).lower()


def _choice_confidence(answer: Any) -> float | None:
    if not isinstance(answer, dict):
        return None
    conf = answer.get("confidence")
    if conf is None:
        return None
    try:
        return round(float(conf), 4)
    except (TypeError, ValueError):
        return None


def _score_0_1(answer_choice: Any, answer_score: Any) -> float:
    """Map Laya answers to a 0–1 polarity (0.5 = neutral)."""
    probs = _choice_probs(answer_choice)
    if probs:
        pos = probs.get("positive", 0.0)
        neu = probs.get("neutral", 0.0)
        neg = probs.get("negative", 0.0)
        total = pos + neu + neg
        if total > 0:
            return (pos * 1.0 + neu * 0.5 + neg * 0.0) / total
    label = _label_from_choice(answer_choice)
    if label == "positive":
        base = 0.75
    elif label == "negative":
        base = 0.25
    else:
        base = 0.5
    if isinstance(answer_score, dict) and answer_score.get("score") is not None:
        try:
            level = float(answer_score["score"])
            return max(0.0, min(1.0, level / 10.0))
        except (TypeError, ValueError):
            pass
    return base


def score_headlines(
    titles: list[str],
    *,
    questions: dict[str, Any] | None = None,
    agent: Any | None = None,
    backend: str | None = None,
    predict_fn: Callable[[Any, str, dict[str, Any]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Score headlines with Laya typed questions.

    Returns list of ``{title, label, score, backend}``. Unavailable backend →
    rows with ``label=None`` and an ``error`` reason (never raises).

    Pass ``agent`` / ``predict_fn`` in tests to avoid real model loads.
    """
    cleaned = [str(t).strip() for t in titles if str(t or "").strip()]
    if not cleaned:
        return []

    if agent is None:
        agent, info = get_agent()
        backend_name = info.get("name") or "none"
        if agent is None:
            return [
                {
                    "title": t,
                    "label": None,
                    "score": None,
                    "backend": backend_name,
                    "error": info.get("reason"),
                }
                for t in cleaned
            ]
    else:
        backend_name = backend or "mock"

    q = questions or HEADLINE_QUESTIONS
    predict = predict_fn or _predict
    rows: list[dict[str, Any]] = []
    for title in cleaned:
        try:
            result = predict(agent, title, q)
            answers = (result or {}).get("answers") or {}
            choice_ans = answers.get("polarity") or answers.get("sentiment") or {}
            score_ans = answers.get("score") or {}
            label = _label_from_choice(choice_ans)
            score = _score_0_1(choice_ans, score_ans)
            rows.append({
                "title": title,
                "label": label,
                "score": round(float(score), 4),
                "backend": backend_name,
                "confidence": _choice_confidence(choice_ans),
            })
        except Exception as exc:
            rows.append({
                "title": title,
                "label": None,
                "score": None,
                "backend": backend_name,
                "error": str(exc),
            })
    return rows


def classify_decision(
    state: str,
    questions: dict[str, Any],
    *,
    agent: Any | None = None,
    backend: str | None = None,
    predict_fn: Callable[[Any, str, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Thin wrap around Laya ``predict`` / ``system_one``.

    Returns ``{backend, available, answers, routing?, error?}``.
    """
    if agent is None:
        agent, info = get_agent()
        backend_name = info.get("name") or "none"
        if agent is None:
            return {
                "backend": backend_name,
                "available": False,
                "answers": {},
                "error": info.get("reason"),
            }
    else:
        backend_name = backend or "mock"

    predict = predict_fn or _predict
    try:
        result = predict(agent, state, questions)
    except Exception as exc:
        return {
            "backend": backend_name,
            "available": False,
            "answers": {},
            "error": str(exc),
        }
    out: dict[str, Any] = {
        "backend": backend_name,
        "available": True,
        "answers": (result or {}).get("answers") or {},
        "usage": (result or {}).get("usage"),
    }
    if isinstance(result, dict) and "routing" in result:
        out["routing"] = result["routing"]
    return out


def average_polarity(rows: list[dict[str, Any]]) -> float | None:
    """Mean 0–1 score across scored headline rows, or None if none scored."""
    vals = [r["score"] for r in rows if isinstance(r.get("score"), (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)
