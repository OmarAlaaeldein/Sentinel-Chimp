# CLAUDE.md

Guidance for coding agents working in this repository.

## Project Overview

Sentinel Chimp is a Python tkinter market terminal: American options (Bjerksund-Stensland 2002), technicals, EWMA / optional GARCH volatility, probability cones, and optional FinBERT sentiment.

**Layout (Phase I MVC on `main`):** `core/` (math + data), `ui/` (views), `main/app.py` (controller), thin `sentinel.py` entrypoint.

## Running the Application

```bash
python sentinel.py
# or, with a venv:
.venv/bin/python sentinel.py
```

- Needs a working tkinter build for your Python
- GUI app — run in the foreground or background as you prefer

## Running Tests

```bash
python -m pytest tests/ -q
```

Suite includes pricing (incl. BS2002 paper tables), technicals, probability cone, options_scan helpers, and related coverage.

## Installing Dependencies

```bash
python -m pip install -r requirements.txt
```

CI-lean deps (no torch): `requirements-ci.txt`.  
`transformers` / `torch` / `accelerate` are optional (sentiment from source only).

## Architecture

| Module | Role |
| :--- | :--- |
| `core/pricing.py` | `VegaChimpCore` — BS, IV, EWMA, BS2002 (+ batch), American FD Greeks |
| `core/technicals.py` | `calculate_technicals` |
| `core/sentiment.py` | FinBERT `SentimentEngine` |
| `core/data.py` | `DataProvider` / `YFinanceProvider` |
| `core/vol_models.py` | Cone math, GARCH(1,1), quadratic smile |
| `core/options_scan.py` | Tradeable-edge / liquidity / ATM filters |
| `ui/*` | Theme, chart, news, options explorer, tooltip, prefs |
| `main/app.py` | `MarketApp` controller + remaining orchestration |
| `sentinel.py` | Launcher + re-exports for tests / scripts |

## Key Patterns

- **Threading:** I/O in daemon threads; UI via `root.after(0, …)`
- **Caching:** TTL caches for history, rates, option chains
- **Optional features:** `TRANSFORMERS_AVAILABLE`, `PLOTLY_AVAILABLE`; GUI flags `use_garch_blend`, `use_smile_vol`, `show_prob_cone`, `show_fib`
- **Options Finder:** Fair = American price under **forecast vol**; compare to **ask/bid** with spread/OI/ATM gates (not mid-only / not IV-circular)
- **Shutdown:** `on_close` → `os._exit(0)`

## Docs

- `to_do.md` — roadmap with live done/pending statuses
- `docs/LOGIC_REVIEW.md` — paper map, scan rules, perf notes
- `plan.md` — historical audit (status block at top is authoritative over older “missing” sections)
- `docs/github-actions-ci.yml` — CI template (publishing `.github/workflows/` needs `workflow` OAuth scope)
