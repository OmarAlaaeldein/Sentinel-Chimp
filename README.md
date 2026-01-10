# PyStock-Sentinel 🛡️📈

**PyStock-Sentinel** is a local, AI-powered stock market dashboard designed for traders who want privacy and depth. It combines standard technical analysis with local Large Language Model (LLM) sentiment analysis and a quantitative options scanner.

![Status](https://img.shields.io/badge/Status-Active-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![AI](https://img.shields.io/badge/AI-Transformers-orange)

## 🚀 Features

### 1. Technical Command Center
* **Live Charting:** Interactive matplotlib charts with switchable timeframes (1D to 5Y) and moving averages (EMA 5, 21, 63).
* **Key Indicators:** Real-time calculation of RSI, Stochastic RSI, MACD, Bollinger Bands, and ATR.
* **Volatility Tracking:** Monitors 30-Day Historical Volatility vs. Implied Volatility.

### 2. Local AI Sentiment Engine 🧠
Instead of relying on third-party APIs with rate limits, PyStock-Sentinel runs **Hugging Face Transformers** locally on your machine.
* **Switchable Models:** Toggle between `DistilBERT` (general sentiment) and `FinBERT` (financial-specific sentiment) on the fly.
* **News Aggregation:** Scrapes Yahoo Finance and Google News RSS for headlines.
* **Privacy:** No data is sent to external AI servers; models are cached locally.

### 3. Options Explorer (EV Scanner) 🔎
A quantitative approach to options, featuring the **VegaChimp** logic core.
* **Fair Value Calculation:** Uses Black-Scholes modeling based on Historical Volatility (HV) rather than Implied Volatility (IV).
* **Expected Value (EV):** Highlights options where the current price is significantly below the theoretical "Fair Price."
* **Visual Verdicts:** Color-coded rows indicate "Over" or "Under" valued contracts.

---

## 🛠️ Installation

### Prerequisites
Ensure you have Python 3.8+ installed. This project relies on `torch` and `transformers` for the AI components.

1. **Clone the repository**
   ```bash
   git clone [https://github.com/OmarAlaaeldein/PyStock-Sentinel.git](https://github.com/OmarAlaaeldein/PyStock-Sentinel.git)
   cd PyStock-Sentinel
