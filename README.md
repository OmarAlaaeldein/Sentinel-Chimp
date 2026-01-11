# PyStock-Sentinel 🛡️📈

**PyStock-Sentinel** is a local, AI-powered stock market dashboard designed for traders who want privacy and depth. It combines institutional-grade quantitative modeling with local Large Language Model (LLM) sentiment analysis.

![Status](https://img.shields.io/badge/Status-Active-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![AI](https://img.shields.io/badge/AI-Transformers-orange)

## 🚀 Features

### 1. Advanced Options Explorer (Quant Core) 🔎
The scanner uses the **Bjerksund-Stensland (2002)** approximation, a professional standard for pricing American-style options.

* **Dynamic Macro Inputs:**
    * **Risk-Free Rate ($r$):** Automatically pulls the 13-week US Treasury Bill yield (`^IRX`) to ensure the time-value of money is accurate to the current hour.
    * **Smart Dividend Yield ($q$):** Dynamically calculates dividend drag from metadata and history. Includes a "Glitch Filter" to prevent percentage-formatting errors from inflating values.
* **Stability Engine:** Rewritten in log-space algebra to prevent `RuntimeWarnings` and mathematical overflows during high-volatility events.
* **Liquidity Filtering:** Uses the **Bid/Ask Mid-Price** for all calculations to ensure "Fair Value" is compared against real tradeable prices, not stale "Last" prints.
* **Breakeven Analysis:** Integrated column showing the exact price the underlying must hit by expiration to reach a $0.00 profit/loss.

### 2. Risk Management & Verdict Logic
The scanner features a switchable **Verdict Engine** controlled by `self.ev_absolute`:

* **Percentage Mode (`False`):** Focuses on "Edge %." Requires a higher threshold for Earnings events (10%) than normal days (5%) to protect against **IV Crush**.
* **Absolute Mode (`True`):** Uses fixed dollar-amount thresholds ($0.1 - $0.2) for high-priced or stable tickers.
* **CSV Export:** One-click backup of all "Under" or "Earnings Under" opportunities for journaling and backtesting.

### 3. Local AI Sentiment Engine 🧠
Runs **Hugging Face Transformers** locally on your machine—no APIs, no costs, total privacy.
* **FinBERT Optimization:** Uses a model specifically pre-trained on financial communication (Reuters, Bloomberg) for superior sentiment accuracy.
* **News Aggregator:** Real-time scraping of Yahoo Finance and Google News RSS feeds.

### 4. Technical Analysis & Volatility
* **GARCH(1,1) Forecasting:** Goes beyond simple moving averages to forecast future volatility based on the clustering of recent price shocks.
* **Live Indicators:** Real-time MACD, RSI, Bollinger Bands, and Stochastic RSI.

## 🛠️ Installation

```bash
pip install yfinance pandas numpy matplotlib transformers torch urllib3