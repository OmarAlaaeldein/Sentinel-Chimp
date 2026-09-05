# 🛡️ Sentinel 2.0
**Strategic Objective:** Transition from a passive retail dashboard to an active, relative-value quantitative research platform.

*Last updated: 2026-09-05 — statuses reflect `main` as shipped.*

---

## 🏗️ Track 1: Architecture & Core (The Foundation)
*Priority: Critical | Status: **Done** (Phase I MVC + DataProvider + UI peel)*

### 1.1 "God Object" Decomposition (MVC Pattern) — **Done**
Landed on `main`:
* **Model (`core/`):** `VegaChimpCore`, technicals, sentiment, vol models, options scan helpers, `DataProvider` / `YFinanceProvider`
* **View (`ui/`):** theme, chart, news, options explorer, tooltip, prefs
* **Controller (`main/app.py`):** `MarketApp` wires inputs ↔ data ↔ views
* **Entrypoint:** thin `sentinel.py` (keeps `Stocks.cmd` / builds working)

### 1.2 Data Abstraction Layer — **Done (YFinance); more providers future**
* ✅ `DataProvider` ABC + `YFinanceProvider` in `core/data.py`
* *Future (optional):* `PolygonProvider` / `IBKR_API` for paid realtime — not started

---

## 🧠 Track 2: The Quant Engine (New Features)

### 2.1 The "Probability Cone" (Vol-Adjusted Charting) — **Done**
* Math + chart overlay + **Prob Cone** toggle on `main`
* σ = EWMA by default; 50/50 EWMA+GARCH when **GARCH blend** is on

### 2.2 The "Fundamental Bias" Filter (Revised) — **Pending**
* Goal: rank option opportunities with a fundamental Z-score (P/E, PEG, Debt/Eq, RevGrowth, EarningsAccel)
* Not started (P/E / PEG already shown in the metrics panel)

### 2.3 The "Semantic Arbitrage" Scanner — **Pending**
* FinBERT peer clustering / basket divergence — not started
* Note: FinBERT sentiment on headlines is already available when running from source

### Related quant work already on `main` (not in original 2.x list)
* ✅ Optional **GARCH(1,1)** forecast + GUI toggle (`use_garch_blend`)
* ✅ Optional **quadratic smile** smoother + GUI toggle (`use_smile_vol`)
* ✅ Batch BS2002 + American FD Greeks in the scanner
* ✅ Options Finder uses **tradeable edge vs forecast vol** (BBO / liquidity / ATM filters) — see `core/options_scan.py` and `docs/LOGIC_REVIEW.md`

---

## 🎨 Track 3: UX & Quality of Life

### 3.1 "Set & Forget" Scanners — **Pending**
Background criteria + tray / webhook notifications — not started

### 3.2 Economic Context Layer — **Pending**
FOMC / CPI / earnings verticals on the chart — not started

### 3.3 Dynamic Watchlist — **Pending**
Persistent `watchlist.json` — not started

### Related UX already on `main`
* ✅ Modern dark trading-terminal theme (`ui/theme.py`)
* ✅ **Fib** levels off by default; checkbox to enable
* ✅ Vol prefs persisted in `user_prefs.json`

---

## 🗓️ Execution Phases

| Phase | Name | Focus | Status |
| :--- | :--- | :--- | :--- |
| **I** | **The Clean Up** | MVC, DataProvider, UI peel | **Done** |
| **II** | **The Eyes** | Probability cones; econ calendar | Cones **done**; calendar **pending** |
| **III** | **The Brain** | Semantic arb / fund bias / alpha | **Pending** |
| **IV** | **The Automaton** | Background scans, watchlists | **Pending** |
