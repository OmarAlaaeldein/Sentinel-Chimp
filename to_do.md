# 🛡️ Sentinel 2.0
**Strategic Objective:** Transition from a passive retail dashboard to an active, relative-value quantitative research platform.

*Last updated: 2026-09-07 — statuses reflect `main` at v2.2.3 (runtime module restore on top of v2.2.2).*

---

## ✅ Shipped (do not re-list as TODO)

| Item | Notes |
| :--- | :--- |
| Phase I MVC + DataProvider | `core/`, `ui/`, `main/app.py` |
| Probability cones + daily EMA mapping | EMA spans are **days**, not chart bars |
| Tradeable-edge Options Finder | Forecast-vol EV@Ask; Lite 3D Plotly |
| Readable fonts + Options Explorer labels | PR #8 |
| Native releases | `.exe` / `.dmg` / `Sentinel-Linux-x64` (1.8+) |
| Dynamic watchlist | `watchlist.json` |
| Ichimoku (9/26/52) | Chart toggle, default off |
| Earnings markers | Chart toggle like Fib, default off |
| Fib from latest fractal swing | Not window high/low |
| Math audit | EMA look-ahead + Ichimoku RTH (PR #12) |
| P/E Percentile datetime units | History `[s]` vs earnings `[us]` merge_asof |
| v2.1 resource pass | Bounded caches; lazy pyplot/plotly/torch; news cap 150; vectorized scan filters; leaner GARCH; `requirements.txt` minus pytest/accelerate |
| v2.2 left-panel truncation fix | Wrapping value column + shorter Vol/BB strings + pane min-width |
| v2.2 docs refresh | CLI flags, structure, sentiment accuracy, perf notes, template markers |
| v2.2.2 smaller binaries | UPX (Win) + `--strip` (Linux/mac) + test/backend excludes; plotly + 3D kept |
| v2.2.3 binary startup fix | Restored `unittest` and `pydoc` stdlib modules for PyInstaller; Windows CLI console encoding |

---

## 🧠 Still open (real backlog)

### 2.2 Fundamental Bias Filter — **Pending**
Rank option opportunities with a fundamental Z-score (P/E, PEG, Debt/Eq, …). P/E / PEG already shown.

### 2.3 Semantic Arbitrage Scanner — **Pending**
FinBERT peer clustering / basket divergence. Headline FinBERT exists from source only.

### 3.1 Set & Forget Scanners — **Pending**
Background criteria + tray / webhook notifications.

### 3.2 Economic Context Layer — **Partial**
Earnings verticals **done**. FOMC / CPI overlays still pending.

### Data providers — **Optional**
`PolygonProvider` / IBKR — not started (Yahoo only).

### Optional quant polish
Full SVI smile; analytic BS2002 Greeks; Ichimoku Senkou extension past last bar.

---

## 🗓️ Phases

| Phase | Focus | Status |
| :--- | :--- | :--- |
| **I** Clean Up | MVC, DataProvider | **Done** |
| **II** Eyes | Cones; earnings markers; econ calendar | Cones + earnings **done**; FOMC/CPI **pending** |
| **III** Brain | Semantic arb / fund bias | **Pending** |
| **IV** Automaton | Background scans | **Pending** |
