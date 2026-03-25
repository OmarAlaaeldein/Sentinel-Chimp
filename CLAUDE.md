# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sentinel Chimp is a monolithic, single-file Python tkinter GUI application for market analysis. It provides American options pricing (Bjerksund-Stensland 2002), technical indicators, EWMA volatility forecasting, and optional FinBERT-powered sentiment analysis.

## Running the Application

```bash
/opt/homebrew/bin/python3.11 sentinel.py &
```

- Always use `python3.11`, not system Python 3.9
- `python-tk@3.11` must be installed (`brew install python-tk@3.11`) for tkinter
- Launch with `&` since it's a GUI application

## Running Tests

```bash
# Run all tests
/opt/homebrew/bin/python3.11 -m pytest tests/ -v

# Run a single test file
/opt/homebrew/bin/python3.11 -m pytest tests/test_option_pricing.py -v

# Run a specific test class or method
/opt/homebrew/bin/python3.11 -m pytest tests/test_option_pricing.py::TestBlackScholesPricing -v
/opt/homebrew/bin/python3.11 -m pytest tests/test_option_pricing.py::TestBlackScholesPricing::test_atm_call -v
```

Tests use pytest. No pytest configuration file exists. Test files add the parent directory to `sys.path` to import from `sentinel.py`.

## Installing Dependencies

```bash
/opt/homebrew/bin/python3.11 -m pip install -r requirements.txt
```

`transformers`, `torch`, and `accelerate` are optional (only needed for AI sentiment features).

## Architecture

**Everything is in `sentinel.py` (~2500 lines).** There are 4 classes and 1 standalone function:

### `VegaChimpCore` (static math library)
All `@staticmethod` methods. Contains Black-Scholes pricing (`bs_price`), Greeks (`bs_greeks`), Newton-Raphson IV solver (`implied_vol`), EWMA volatility forecasting (`ewma_vol_forecast`), and Bjerksund-Stensland American option pricing (`bjerksund_stensland`). This is the testable, pure-math core of the app.

### `calculate_technicals(df)` (standalone function)
Takes a pandas OHLCV DataFrame and adds ~21 technical indicator columns: RSI, MACD, Bollinger Bands, ATR, StochRSI, VWAP, OBV, ADX, Williams %R, CCI.

### `SentimentEngine`
Manages FinBERT model loading and batch prediction. Optional feature guarded by `TRANSFORMERS_AVAILABLE` flag. A module-level singleton `sentiment_engine` is created.

### `MarketApp` (the GUI "god object")
Contains all UI construction, data fetching (via `yfinance`), charting (matplotlib embedded via `FigureCanvasTkAgg`), options scanning, and display logic. Not directly unit-testable.

### `Tooltip`
Small utility for hover tooltips on tkinter widgets.

## Key Patterns

- **Threading**: All I/O (data fetching, model loading, options scanning) runs in daemon threads. UI updates dispatch to the main thread via `root.after(0, callback)`.
- **Caching**: TTL-based in-memory caches (`data_cache`, `sent_cache`, `valuation_cache`) protected by `threading.Lock`.
- **Optional features**: `TRANSFORMERS_AVAILABLE` and `PLOTLY_AVAILABLE` flags gate heavy dependencies. `self.use_sentiment` toggles the sentiment UI (default `False`).
- **Error fallback**: Bjerksund-Stensland falls back to Black-Scholes on numerical failure. Broad try/except blocks throughout.
- **Shutdown**: `on_close` uses `os._exit(0)` to force-kill all daemon threads.

## Testing Conventions

- Tests live in `tests/` (no `__init__.py`).
- `test_option_pricing.py` tests `VegaChimpCore` (BS pricing, Greeks, IV solver, EWMA, American options).
- `test_technicals.py` tests `calculate_technicals()` (all indicator calculations).
- Tests use `pytest.approx` with tolerances for floating-point math (typically `abs=0.01` to `abs=0.5` depending on the calculation).
- Test data is constructed synthetically (no network calls or fixtures).

## Known Audit Items

`plan.md` contains a detailed audit of bugs and improvement priorities. Key items include dividend yield handling, VWAP daily reset behavior, and volatility blend methodology.
