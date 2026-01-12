# 📋 Sentinel 2.0 Upgrade Roadmap

This document outlines the architectural upgrades required to transition Sentinel from a retail dashboard to a high-fidelity quantitative research tool. 

## ⚖️ Licensing Strategy: Business Source License (BSL) 1.1

To balance community contribution with financial sustainability, Sentinel 2.0 will adopt the **Business Source License (BSL) 1.1**.

### Why BSL 1.1?
* **Monetization & Control**: It allows the developer to control commercialization while providing open access to the community. 
* **Non-Production Freedom**: Copying, modifying, and non-production use (testing, development, personal hobbies) are free for everyone.
* **Production Protection**: Organizations intending to use the software for commercial production (e.g., as a managed service or embedded in a competing product) must purchase a commercial license.
* **Guaranteed Open Source**: The license automatically converts to a fully open-source **GPL-compatible** license after a specified "Change Date" (typically 3–4 years), preventing permanent vendor lock-in.

---

## Phase 1: Technical Intelligence

*Goal: Detect regimes of compression and extreme extension.*

### 1. Update `calculate_technicals`

* **TTM Squeeze Indicator**:
    * Calculate **Keltner Channels** ($20\text{ SMA} \pm 1.5\text{ ATR}$).
    * Create a boolean column `Squeeze_On`.
    * **Logic**: `True` if Bollinger Bands are completely *inside* Keltner Channels. This signals a period of volatility compression often followed by an explosive move.

### 2. Charting Upgrades ("Probability Cone")

* **Objective**: Visualize the "Expected Move" on the daily chart.
* **Logic**:
    * Take the last known Price and current `GARCH_Vol`.
    * Project 30 days into the future.
    * Calculate Upper/Lower bounds: $\text{Price} \cdot e^{\pm \sigma \sqrt{t/252}}$.
* **Visual**: Plot these bounds as dashed "Cones" overlaying the price action to show if current price action is staying within statistical expectations.



---

## Phase 2: The Quant Arbitrage Engine (New)

*Goal: Identify mispricings based on relative value and peer behavior rather than single-ticker technicals.*

This module allows users to search for a stock and immediately see "broken correlations" with its peers using two distinct approaches:

### Approach A: Price Correlation (The "Technical" Cluster)

* **Concept**: "Which stocks *move* together?"
* **Method**:
    * **Similarity Matrix**: Fetch 1 year of daily returns for a universe of stocks (e.g., S&P 500 or Nasdaq 100).
    * **Calculation**: Compute a **Pearson Correlation Matrix** between all assets.
    * **Usage**: When observing Stock A, look up its top 5 correlated peers (correlation $> 0.85$).
    * **Signal**: If Stock A is down -2% but its highly correlated peers are flat or green, Stock A is statistically "cheap" relative to its own cluster (Mean Reversion trade).

### Approach B: Semantic Embeddings (The "Fundamental" Cluster)

* **Concept**: "Which stocks *are* the same?" (NLP-Driven Asset Allocation).
* **Method**:
    * **Data**: Scrape the "Business Summary" text for the target universe.
    * **Vectorization**: Feed these summaries into **FinBERT** (using the base model, not the sentiment head) to generate 768-dimensional embeddings.
    * **Clustering**: Use **Cosine Similarity** to find stocks with the closest vector distance.
    * **Advantage**: This detects hidden relationships (e.g., companies in different sectors that rely on the same technology) that pure price correlation might miss.

### **Recommended Hybrid Implementation**

1.  **Filter by Semantics**: Use FinBERT to identify a "Fundamental Basket" of 10–20 truly similar companies.
2.  **Trade by Correlation**: Inside that basket, calculate the Z-Score of the price spread between the target and the basket average.
3.  **Trigger**: Buy when the target stock deviates $> 2$ Standard Deviations from the basket's price action.


### Minor To-Do Items
1. option for fibbonaci retracements on charts
2. add option to save an option in a watchlist that is updated everytime the user opens the app
3. add economic calendar events to the chart (fed announcements, jobs report, cpi, etc)
4. add support for derivative tickers (leveraged etfs, etc)
5. use lighter models for financial sentiment analysis
6. subroutines to run undervalued scans every x hours automatically and notify the user
7. disable arbitrage for options not updated today