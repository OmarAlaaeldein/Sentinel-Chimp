# Sentinel Chimp 🛡️

**Sentinel Chimp** is a sophisticated, Python-based market analysis dashboard designed for retail traders who demand institutional-grade mathematics. It bridges the gap between basic charting tools and professional quantitative platforms, featuring advanced options pricing models, volatility forecasting, and real-time technical analysis.

> **⚠️ Release Note:** The standalone executable (Release Package) operates in **"Lite Mode"** for maximum compatibility. It **does not include** the AI Sentiment Analysis engine to keep file sizes manageable and ensure it runs on any Windows machine without heavy dependencies. To use the AI features, run the application from the source.

---

## 🚀 Key Features

### 1. Advanced Options Valuation
Unlike standard calculators that use Black-Scholes, Sentinel uses the **Bjerksund-Stensland (2002)** model to price American options.
* **Log-Space Algebra:** Prevents mathematical overflow/underflow during extreme volatility events.
* **Dynamic Risk-Free Rate:** Automatically uses term-aware treasury inputs with interpolation between short-end (^IRX) and long-end (^TNX) rates.
* **Edge Detection:** Scans option chains to find contracts where the Market Price diverges significantly from the Theoretical Value (EV).
* **3D Landscape Visualization:** Interactive 3D plotting of "Strike vs. Expiry vs. Expected Value," allowing you to visually spot "islands of value" across the entire option chain.

### 2. Institutional Volatility Forecasting
Sentinel looks beyond simple Historical Volatility (HV).
* **EWMA Forecasting:** Computes an EWMA volatility estimate (RiskMetrics-style decay) for forward-looking volatility context.
* **Smart Blending:** Option pricing inputs use market implied volatility blended with 30-day historical volatility (time-weighted by maturity) to estimate fair value.

### 3. Smart Technical Dashboard
A threaded, non-blocking GUI featuring a professional Dark Mode interface optimized for low eye strain:
* **Momentum:** RSI (14), Stoch RSI, MACD.
* **Trend Strength:** ADX (Average Directional Index) to distinguish between trending and chopping markets.
* **Volume Analysis:** OBV (On-Balance Volume) trend detection and **VWAP Gap** analysis (Intraday Bull/Bear control).
* **Risk:** ATR (Average True Range) for volatility-based stop losses.
* **Fundamental Context:** Displays P/E Ratios (TTM/Fwd) and calculates a **P/E Percentile** to show if the stock is historically cheap or expensive.

### 4. AI Sentiment Engine (Source Code Only)
* **Model:** Powered by `ProsusAI/finbert` (Financial BERT).
* **Function:** Scrapes news headlines (Yahoo/Google RSS) and computes a sentiment score (-1 to +1) using a Transformer model specifically fine-tuned for financial text.
* *Note: Requires PyTorch and Transformers libraries.*

---

## 📦 Compatibility & Release Info

To ensure this tool works on standard trading laptops without requiring NVIDIA GPUs or massive libraries, the **pre-compiled Release Package** differs from the source code:

| Feature | Source Code (`.py`) | Release Package (`.exe`) |
| :--- | :---: | :---: |
| **Charting & Technicals** | ✅ Included | ✅ Included |
| **Bjerksund-Stensland Math** | ✅ Included | ✅ Included |
| **EWMA/HV Volatility Logic** | ✅ Included | ✅ Included |
| **Options Scanner** | ✅ Included | ✅ Included |
| **3D Visualizer** | ✅ Included | ✅ Included |
| **AI Sentiment (FinBERT)** | ✅ **Active** | ❌ **Disabled** |

**Why is AI disabled in the release?**
The AI engine relies on `PyTorch` and `Transformers`, which can add over 1GB to the file size and may cause compatibility issues on computers without specific drivers. The Release Package is optimized for speed and portability.

---

## 🛠️ Installation

### Option A: Running from Source (Full Features)
To use the AI Sentiment engine, you must run from the source:

1.  **Clone the Repo**
    ```bash
    git clone [https://github.com/omaralaaeldein/Sentinel-Chimp.git](https://github.com/omaralaaeldein/Sentinel-Chimp.git)
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

### Option B: Using the Release Package
1.  Download the latest `.zip` from the **Releases** tab.
2.  Extract the folder.
3.  Run `Sentinel.exe`.
4.  *No Python installation required.*

---

## 📉 Usage Guide

1.  **Ticker Entry:** Type a ticker (e.g., `NVDA`, `SPY`) and press Enter.
2.  **Technicals:** Review the left panel for RSI, MACD, VWAP Gap, and Volatility stats.
3.  **Options Scanner:**
    * Click **"Open Options Explorer"**.
    * Select an expiration date (or multiple).
    * Click **"Scan ALL Undervalued"** to find contracts where `EV > 0`.
    * **Green Rows** indicate "Undervalued" (Potential Buy/Cover).
    * **Red Rows** indicate "Overvalued" (Potential Sell/Write).
    * **3D Plot:** Click the "3D Plot" buttons to visualize the data in an interactive cube.
4.  **Export:** Save your scan results to CSV or your 3D plots to HTML for sharing and toggling.

---

## 💡 Inspiration & Credits
This project was built with inspiration from the open-source community. Special thanks to the following projects for their foundational concepts and approaches:

* [**Vegachimp**](https://github.com/Orange-The-Fruit/vegachimp/tree/main) by *Orange-The-Fruit*
* [**PyStock**](https://github.com/ikitcheng/pystock) by *ikitcheng*

---

## ⚖️ Disclaimer
*This software is for educational and research purposes only. It is not financial advice. The Bjerksund-Stensland model and volatility estimates (EWMA/HV/IV-based) are theoretical approximations and do not guarantee future market behavior. Always trade at your own risk.*
