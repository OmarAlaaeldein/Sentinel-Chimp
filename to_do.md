
---

# 📋 Sentinel 2.0 Upgrade Roadmap

This document outlines the architectural upgrades required to transition `sentinel.py` from a retail dashboard to a quantitative research tool.

## Phase 1: The Math Core (Volatility & Pricing)

*Goal: Replace static "backward-looking" metrics with dynamic "forward-looking" models.*

### 1. Upgrade `VegaChimpCore` Class

Replace the current Black-Scholes implementation with a robust pricing engine.

* **Implement GARCH(1,1):**
* **Objective:** Forecast *future* volatility instead of relying on past Historical Volatility (HV).
* **Method:** Add a static method that accepts a list of log returns. Use the recursive formula: .
* **Output:** An annualized forward-looking volatility percentage.


* **Implement Bjerksund-Stensland (Closed Form 2002):**
* **Objective:** Price American Options correctly (accounting for early exercise).
* **Logic:** The method must check if the "Cost of Carry" (b) is less than the Risk-Free Rate (r). If true, it checks if early exercise is optimal and adjusts the premium upward compared to standard Black-Scholes.



### 2. Dynamic Risk-Free Rate

* **Issue:** The current `fetch_options_batch` method uses a hardcoded rate of `0.045` (4.5%).
* **Fix:** Create a helper function to fetch the current **13-Week Treasury Bill (^IRX)** yield from Yahoo Finance.
* **Implementation:** Use this dynamic rate in all Option Pricing models (Bjerksund-Stensland) to ensure fair value accuracy relative to the current macro environment.

---

## Phase 2: Technical Intelligence

*Goal: Detect regimes of compression and extreme extension.*

### 3. Update `calculate_technicals`

* **TTM Squeeze Indicator:**
* Calculate **Keltner Channels** (20 SMA +/- 1.5 ATR).
* Create a boolean column `Squeeze_On`.
* **Logic:** `True` if Bollinger Bands are completely *inside* Keltner Channels. This signals a period of volatility compression often followed by an explosive move.


* **GARCH Integration:**
* Calculate daily Log Returns of the Close price.
* Pass these returns to your new `VegaChimpCore.garch` method.
* Store the result in a new column `GARCH_Vol` for use in the Option Scanner.



### 4. Charting Upgrades ("Probability Cone")

* **Objective:** Visualize the "Expected Move" on the daily chart.
* **Logic:**
* Take the last known Price and current `GARCH_Vol`.
* Project 30 days into the future.
* Calculate Upper/Lower bounds: .


* **Visual:** Plot these bounds as dashed "Cones" overlaying the price action to show if current price action is staying within statistical expectations.

---

## Phase 3: The Options Scanner

*Goal: Automated edge finding.*

### 5. Scanner Logic Overhaul

Update `fetch_options_batch` in `MarketApp`:

* **Input Switch:** Prioritize `GARCH_Vol` over `hv_30` for pricing inputs.
* **Model Switch:** Call `VegaChimpCore.bjerksund_stensland` instead of `bs_price`.
* **Verdict Logic:**
* Calculate `Edge = Fair Value - Market Price`.
* **Green (Underpriced):** Edge > +15% of Fair Value.
* **Red (Overpriced):** Edge < -15% of Fair Value.



---

## Phase 4: The Quant Arbitrage Engine (New)

*Goal: Identify mispricings based on relative value and peer behavior rather than single-ticker technicals.*

This module will allow the user to search for a stock and immediately see "broken correlations" with its peers. We will implement two distinct approaches for finding peers:

### Approach A: Price Correlation (The "Technical" Cluster)

* **Concept:** "Which stocks *move* together?"
* **Method:**
* **Similarity Matrix:** Fetch 1 year of daily returns for a universe of stocks (e.g., S&P 500 or Nasdaq 100).
* **Calculation:** Compute a Pearson Correlation Matrix between all assets.
* **Usage:** When observing Stock A, look up its top 5 correlated peers (correlation > 0.85).
* **Signal:** If Stock A is down -2% but its highly correlated peers are flat or green, Stock A is statistically "cheap" relative to its own cluster (Mean Reversion trade).



### Approach B: Semantic Embeddings (The "Fundamental" Cluster)

* **Concept:** "Which stocks *are* the same?" (NLP-Driven Asset Allocation).
* **Method:**
* **Data:** Scrape the "Business Summary" text for the target universe.
* **Vectorization:** Feed these summaries into **FinBERT** (using the base model, not the sentiment head) to generate 768-dimensional embeddings.
* **Clustering:** Use **Cosine Similarity** to find stocks with the closest vector distance.
* **Advantage:** This detects hidden relationships (e.g., two companies in different sectors that rely on the same underlying commodity or technology) that pure price correlation might miss.



### **Recommended Hybrid Implementation**

1. **Filter by Semantics:** Use FinBERT to identify a "Fundamental Basket" of 10-20 truly similar companies.
2. **Trade by Correlation:** Inside that basket, calculate the Z-Score of the price spread between the target and the basket average.
3. **Trigger:** Buy when the target stock deviates > 2 Standard Deviations from the basket's price action.