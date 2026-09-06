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

## CLI (headless)

No tkinter import on CLI paths (`sentinel.py` lazy-loads GUI only when argv is empty).

```bash
.venv/bin/python sentinel.py analyze LULU
.venv/bin/python sentinel.py scan LULU --under-only --max-expiries 6
.venv/bin/python -m main.cli scan LULU --type call --json
.venv/bin/python -m sentinel_cli analyze AMD --json
```

Shared scan orchestration: `core/scan_service.py` (GUI `MarketApp.fetch_options_batch` is a thin adapter).

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
| `core/scan_service.py` | Shared options-scan + ticker analyze (GUI + CLI) |
| `ui/*` | Theme, chart, news, options explorer, tooltip, prefs |
| `main/app.py` | `MarketApp` controller + remaining orchestration |
| `main/cli.py` | Headless argparse CLI (`analyze` / `scan`) |
| `sentinel.py` | Launcher (GUI or CLI) + re-exports for tests / scripts |
| `sentinel_cli.py` | Thin `python -m sentinel_cli` alias |

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
