# Sentinel Chimp 🛡️

**Sentinel Chimp** is a sophisticated, Python-based market analysis dashboard designed for retail traders who demand institutional-grade mathematics. It bridges the gap between basic charting tools and professional quantitative platforms, featuring advanced options pricing models, volatility forecasting, and real-time technical analysis.

> **⚠️ Release Note:** The standalone release package operates in **"Lite Mode"** for maximum compatibility. It **does not include** the AI Sentiment Analysis engine to keep file sizes manageable and ensure it runs smoothly on standard systems. To use AI features, run the application from source.

---

## Local Ollama CLI

Run from source with Python 3.11. Model commands use the Python standard library;
Ollama must already be running locally with the model installed.

```bash
python3.11 sentinel.py models list
python3.11 sentinel.py models show qwen3.5:4b --json
python3.11 sentinel.py models select qwen3.5:4b --save-as fast --num-ctx 4096 --num-predict 512
python3.11 sentinel.py profiles list
python3.11 sentinel.py profiles use fast
python3.11 sentinel.py models current
python3.11 sentinel.py ask "Explain implied volatility in two sentences." --profile fast
python3.11 sentinel.py config --json
```

`models select` validates the exact installed name, saves its generation settings,
and sets the default profile. Use `--replace` to overwrite an existing profile.
Names containing `/` and `:` work as returned by `models list`; aliases remain separate names.
Selection checks metadata; the first `ask` establishes whether the server can actually load that architecture.
The GUI's FinBERT sentiment engine is separate from these new CLI profiles.

Every new command accepts `--json`, `--config PATH` and `--host URL` **after the
leaf command** (for example, `models list --json`). JSON success uses
`{"schema_version":1,"status":"ok","data":{...}}`; errors use
`{"schema_version":1,"status":"error","error":{"code":"...","message":"..."}}`.
There are no interactive prompts. Exit codes: `0` success, `2` invalid CLI input,
`4` command/config/server failure or every scan expiry failed, `5` partial scan failure.
`verify` retains its existing text output and failure code `1`.
Existing analyze/scan success JSON retains its original fields; scan adds `status`,
`errors`, `requested_expiries` and `total_count`. `--limit` limits displayed/JSON rows;
`--csv` exports all matched rows, with headers even for an empty result. Export notices go to stderr.

Configuration precedence: `--config` → `SENTINEL_CONFIG` → platform user config.
Defaults are `~/Library/Application Support/SentinelChimp/config.json` on macOS,
`%APPDATA%/SentinelChimp/config.json` on Windows, and
`$XDG_CONFIG_HOME/sentinel-chimp/config.json` (normally `~/.config/...`) on Linux.
Writes are validated, locked, and atomically replaced. Read-only commands create no config file.
Host precedence: `--host` → `SENTINEL_OLLAMA_HOST` → `OLLAMA_HOST` →
`http://127.0.0.1:11434`. `ask` profile precedence: `--profile` → `SENTINEL_PROFILE`
→ saved default. Host settings are supplied per invocation/environment, not saved in profiles.

The client allows loopback hosts, bypasses HTTP proxies, rejects redirects and
remote-backed models, and never pulls a model. `ask` sends only the provided prompt,
limited to 16,000 characters; generation uses the saved temperature/context/token
limits, a 120-second timeout and a five-minute keep-alive. A `done_reason` of `length`
means the token limit stopped generation. This is a basic local prompt command;
structured explanations of saved market analysis and GUI model selection remain planned in [astra.md](astra.md).

## Screenshots

![Main terminal — AMD chart, technicals, probability cone](Screenshots/main-terminal.png)

![Options Explorer — tradeable-edge scan](Screenshots/options-explorer.png)


## 🚀 Key Features
### 5. Watchlist, Ichimoku, Earnings & Fib
* **Watchlist:** Persistent `watchlist.json` with quick ticker switch.
* **Ichimoku:** Standard 9/26/52 cloud (toggle, default off).
* **Earnings markers:** Vertical dashed lines (toggle like Fib, default off).
* **Fib:** Retracement from the latest confirmed fractal swing.
* **EMAs:** Daily-span EMAs (5/21/63/200 **days**) mapped onto any chart interval.


### 1. Advanced Options Valuation
Unlike standard calculators that use Black-Scholes, Sentinel uses the **Bjerksund-Stensland (2002)** model to price American options.
* **Log-Space Algebra:** Prevents mathematical overflow/underflow during extreme volatility events.
* **Dynamic Risk-Free Rate:** Automatically uses term-aware treasury inputs with interpolation between short-end (^IRX) and long-end (^TNX) rates.
* **Edge Detection:** Scans option chains to find contracts where the Market Price diverges significantly from the Theoretical Value (EV).
* **3D Landscape Visualization:** Interactive matplotlib view + Plotly HTML export of **Strike × Days-to-expiry × EV@Ask ($)**, with a labeled colorbar (Fair − Ask), richer hover, fixed camera angle, and an optional EV heatmap underlay when the scatter is dense. Still uses existing Lite deps (`plotly` / `matplotlib` — no pyvista/torch).

### 2. Institutional Volatility Forecasting
Sentinel looks beyond simple Historical Volatility (HV).
* **EWMA Forecasting:** RiskMetrics-style λ=0.94 EWMA (always).
* **Optional GARCH / smile:** GUI toggles for fitted GARCH(1,1) blend and quadratic IV smile smoothing.
* **Options fair value:** Forecast vol only (EWMA ± GARCH); market IV is shown and used for Greeks — see Options Finder rules in `docs/LOGIC_REVIEW.md`.

### 3. Smart Technical Dashboard
A threaded, non-blocking GUI featuring a professional Dark Mode interface with a readable type ladder (≈12pt body / 13–14pt headers) optimized for low eye strain:
* **Momentum:** RSI (14), Stoch RSI, MACD.
* **Trend Strength:** ADX (Average Directional Index) to distinguish between trending and chopping markets.
* **Volume Analysis:** OBV (On-Balance Volume) trend detection and **VWAP Gap** analysis (Intraday Bull/Bear control).
* **Risk:** ATR (Average True Range) for volatility-based stop losses.
* **Fundamental Context:** Displays P/E Ratios (TTM/Fwd) and calculates a **P/E Percentile** to show if the stock is historically cheap or expensive.

### 4. AI Sentiment Engine (Source Code Only, Off by Default)
* **Model:** Powered by `ProsusAI/finbert` (Financial BERT).
* **Function:** Scrapes news headlines (Yahoo/Google RSS, capped at 150) and computes a sentiment score (0–1 scale, 0.5 = neutral) using a Transformer model specifically fine-tuned for financial text.
* *Note: Requires PyTorch and Transformers libraries; disabled in-app by default (`use_sentiment = False`).*

---

## 📦 Compatibility & Release Info

Prebuilt **Lite Mode** native installers/binaries are published on the [Releases](https://github.com/OmarAlaaeldein/Sentinel-Chimp/releases) page for **Windows**, **Linux**, and **unsigned macOS** (no zip packs).

| Artifact | Platform |
| :--- | :--- |
| `Sentinel.exe` | Windows 10/11 x64 — download and run |
| `Sentinel-Linux-x64` | Linux x64 — `chmod +x Sentinel-Linux-x64 && ./Sentinel-Linux-x64` |
| `Sentinel-macOS-unsigned.dmg` | macOS — open DMG; **unsigned** (Gatekeeper: right-click → Open) |

To ensure this tool works on standard trading laptops without requiring NVIDIA GPUs or massive libraries, the **pre-compiled Release Package** differs from the source code:

| Feature | Source Code (`.py`) | Release Package (`.exe` / `.dmg` / Linux binary) |
| :--- | :---: | :---: |
| **Charting & Technicals** | ✅ Included | ✅ Included |
| **Bjerksund-Stensland Math** | ✅ Included | ✅ Included |
| **EWMA/HV Volatility Logic** | ✅ Included | ✅ Included |
| **Options Scanner** | ✅ Included | ✅ Included |
| **3D Visualizer** | ✅ Included | ✅ Included |
| **AI Sentiment (FinBERT)** | ⚠️ Source only, **off by default** | ❌ **Disabled** |

**Why is AI disabled in the release?**
The AI engine relies on `PyTorch` and `Transformers`, which can add over 1GB to the file size and may cause compatibility issues on computers without specific drivers. The Release Package is optimized for speed and portability.

---

## 🛠️ Installation

### Option A: Running from Source (Full Features)
To use the AI Sentiment engine, you must run from the source:

1.  **Clone the Repo**
    ```bash
    git clone https://github.com/OmarAlaaeldein/Sentinel-Chimp.git
    cd Sentinel-Chimp
    ```
2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `torch`, `transformers`, `yfinance`, `pandas`, `numpy`, `matplotlib`, `plotly` are installed)*
3.  **Run**
    ```bash
    python sentinel.py
    ```

### CLI (headless — no tkinter)

The same Options Finder rules as the GUI (`core/options_scan.py` + `core/scan_service.py`).

```bash
# Spot + EWMA/HV + short technicals
python sentinel.py analyze LULU
python -m main.cli analyze LULU
python -m sentinel_cli analyze LULU --json

# Options scan (fair = BS2002 @ forecast vol; EV@Ask = fair − ask)
python sentinel.py scan LULU --under-only --max-expiries 6
python sentinel.py scan LULU --type call --garch --json
python -m main.cli scan AMD --under-only --max-expiries 4 --type put

# Offline math self-test (no network): BSM/BS2002/IV/EWMA/cone checks
python sentinel.py verify
```

| Flag | Meaning |
| :--- | :--- |
| `--under-only` | Only Under / Earnings Under, ranked by edge % |
| `--max-expiries N` | First N listed expirations |
| `--expiry DATE` | Only this expiry (prefix match); repeatable |
| `--type call\|put\|all` | Side filter (default `all`) |
| `--garch` | 50/50 EWMA+GARCH forecast vol blend |
| `--smile` | Smooth display IV with per-expiry quadratic smile |
| `--euro-greeks` | Analytic European Greeks instead of American FD |
| `--div YIELD` | Dividend override: decimal (`0.0098`) or percent (`0.98`) |
| `--limit N` | Print only the first N rows |
| `--csv PATH` | Also write full scan rows to CSV |
| `--json` | Machine-readable output |

`python sentinel.py` with **no args** still launches the GUI.

### Option B: Prebuilt Releases (Lite Mode)
Download from **[Releases](https://github.com/OmarAlaaeldein/Sentinel-Chimp/releases)** — no Python required. Assets are native binaries/installers (not zips).

**Windows**
1. Download `Sentinel.exe`.
2. Run it (Windows Defender may scan the unsigned exe on first launch).

**Linux**
1. Download `Sentinel-Linux-x64`.
2. Make executable and run:
   ```bash
   chmod +x Sentinel-Linux-x64
   ./Sentinel-Linux-x64
   ```

**macOS (unsigned)**
1. Download `Sentinel-macOS-unsigned.dmg`.
2. Open the DMG and run (or copy) `Sentinel.app`.
3. Because the app is **not signed/notarized**, first launch via **right-click → Open** (or `xattr -dr com.apple.quarantine /path/to/Sentinel.app`).

### Option C: Build macOS locally
1.  Build the macOS app bundle:
    ```bash
    ./build_macos.sh --auto --onedir --install-deps
    ```
2.  Launch the generated app:
    ```bash
    open "dist/Sentinel.app"
    ```
3.  You can also run `./build_macos.command` to launch the build script interactively.


---

## 🧱 Project Structure (Phase I MVC)

`python sentinel.py` remains the supported entrypoint (`Stocks.cmd` / build scripts unchanged).

| Path | Role |
| :--- | :--- |
| `sentinel.py` | Thin launcher + backwards-compatible re-exports |
| `core/pricing.py` | `VegaChimpCore` (BS / BS2002 batch / EWMA / American FD Greeks) |
| `core/technicals.py` | `calculate_technicals` |
| `core/sentiment.py` | FinBERT `SentimentEngine` |
| `core/data.py` | `DataProvider` ABC + `YFinanceProvider` |
| `core/vol_models.py` | Probability cone, GARCH(1,1), quadratic smile |
| `core/options_scan.py` | Tradeable-edge / liquidity filters for Options Finder |
| `core/scan_service.py` | Shared scan + analyze orchestration (GUI + CLI) |
| `ui/` | Theme, chart, news, options explorer, 3D plot, watchlist, prefs, tooltip |
| `main/app.py` | `MarketApp` controller |
| `main/cli.py` | Headless CLI (`analyze` / `scan` / `verify`) |
| `sentinel_cli.py` | Thin `python -m sentinel_cli` alias |
| `docs/LOGIC_REVIEW.md` | Paper mapping, scan rules, perf notes |
| `to_do.md` | Roadmap with live statuses |

---

## 📉 Usage Guide

1.  **Ticker Entry:** Type a ticker (e.g., `NVDA`, `SPY`) and press Enter / **Load**.
2.  **Technicals:** Review the left panel for RSI, MACD, VWAP Gap, and Volatility stats.
3.  **Chart toggles:** **Prob Cone** (on by default), **Fib** (off by default), period buttons (1D…25Y). Optional **GARCH blend** / **Smile vol** near the ticker bar.
4.  **Options Scanner:**
    * Click **"Open … Options"**.
    * Select expiration(s), or **Scan ALL Undervalued**.
    * Fair value uses **forecast vol** (EWMA ± GARCH), not the contract’s own IV.
    * **EV@Ask** is tradeable edge vs the ask (must clear half-spread + liquidity/ATM filters).
    * **Green** = Under (candidate long); **Red** = Over (candidate write). See `docs/LOGIC_REVIEW.md`.
    * **3D Plot** visualizes the filtered surface.
5.  **Export:** CSV scan results or HTML 3D plots.

---

## ⚡ Performance notes (v2.1+)

* **Bounded caches:** history/chain/news/valuation caches are size-capped with TTL eviction — long sessions can't grow memory without bound.
* **Lazy heavy imports:** `matplotlib.pyplot`, `plotly`, and `torch`/`transformers` load only when their feature is used, keeping cold startup light.
* **Cheaper scans:** vectorized liquidity/ATM filters, leaner GARCH fit, throttled chart hover, capped log widget and news list (150 headlines).

---

## 💡 Inspiration & Credits
This project was built with inspiration from the open-source community. Special thanks to the following projects for their foundational concepts and approaches:

* [**Vegachimp**](https://github.com/Orange-The-Fruit/vegachimp/tree/main) by *Orange-The-Fruit*
* [**PyStock**](https://github.com/ikitcheng/pystock) by *ikitcheng*

---

## ⚖️ Disclaimer
*This software is for educational and research purposes only. It is not financial advice. The Bjerksund-Stensland model and volatility estimates (EWMA/HV/IV-based) are theoretical approximations and do not guarantee future market behavior. Always trade at your own risk.*
