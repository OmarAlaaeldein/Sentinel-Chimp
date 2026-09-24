# CLAUDE.md

Guidance for coding agents working in this repository.

## Project Overview

Sentinel Chimp is a Python tkinter market terminal: American options (Bjerksund-Stensland 2002), technicals, EWMA / optional GARCH volatility, probability cones, and optional local **Laya** headline polarity.

**Layout (Phase I MVC on `main`):** `core/` (math + data), `ui/` (views), `main/app.py` (controller), thin `sentinel.py` entrypoint.

## Running the Application

```bash
python sentinel.py
# or, with a venv:
.venv/bin/python sentinel.py
```

- Needs a working tkinter build for your Python
- GUI app — run in the foreground or background as you prefer
- Optional Laya: `pip install laya` (or `laya-mlx` on Apple Silicon) then `SENTINEL_LAYA=1`

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

Suite includes pricing (incl. BS2002 paper tables), technicals, probability cone, options_scan helpers, optional Laya adapter mocks, and related coverage.

## Installing Dependencies

```bash
python -m pip install -r requirements.txt
```

CI-lean deps: `requirements-ci.txt`.  
Optional Laya packages (`laya` / `laya-mlx`) are **not** in requirements — install separately when needed.

## Architecture

| Module | Role |
| :--- | :--- |
| `core/pricing.py` | `VegaChimpCore` — BS, IV, EWMA, BS2002 (+ batch), American FD Greeks |
| `core/technicals.py` | `calculate_technicals` |
| `core/laya_decisions.py` | Optional Laya polarity / typed decisions (`SENTINEL_LAYA=1`) |
| `core/data.py` | `DataProvider` / `YFinanceProvider` |
| `core/vol_models.py` | Cone math, GARCH(1,1), quadratic smile |
| `core/options_scan.py` | Tradeable-edge / liquidity / ATM filters |
| `core/scan_service.py` | Shared options-scan + ticker analyze (GUI + CLI) |
| `ui/*` | Theme, chart, news, options explorer, tooltip, prefs |
| `main/app.py` | `MarketApp` controller + remaining orchestration |
| `main/cli.py` | Headless argparse CLI (`analyze` / `scan` / `verify`) |
| `sentinel.py` | Launcher (GUI or CLI) + re-exports for tests / scripts |
| `sentinel_cli.py` | Thin `python -m sentinel_cli` alias |

## Key Patterns

- **Threading:** I/O in daemon threads; UI via `root.after(0, …)`
- **Caching:** bounded TTL caches for history, rates, option chains, news, valuation
- **Startup discipline:** `matplotlib.pyplot`, `plotly`, and Laya backends import lazily; polarity loads only if `SENTINEL_LAYA=1`
- **Optional features:** Laya via `core.laya_decisions`; `PLOTLY_AVAILABLE`; GUI flags `use_garch_blend`, `use_smile_vol`, `show_prob_cone`, `show_fib`, `use_laya`
- **Options Finder:** Fair = American price under **forecast vol**; compare to **ask/bid** with spread/OI/ATM gates (not mid-only / not IV-circular)
- **Shutdown:** `on_close` → `os._exit(0)`

## Docs

- `to_do.md` — roadmap with live done/pending statuses
- `docs/LOGIC_REVIEW.md` — paper map, scan rules, perf notes
- `plan.md` — historical audit (status block at top is authoritative over older “missing” sections)
- `docs/github-actions-ci.yml` — original CI template (live workflow: `.github/workflows/ci.yml`)
