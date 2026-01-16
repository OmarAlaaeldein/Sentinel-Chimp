# 🛡️ Sentinel 2.0:
**Strategic Objective:** Transition from a passive retail dashboard to an active, relative-value quantitative research platform.

---

## 🏗️ Track 1: Architecture & Core (The Foundation)
*Priority: Critical | Status: Pending*

### 1.1 "God Object" Decomposition (MVC Pattern)
Refactor the monolithic `MarketApp` class into three distinct layers to ensure stability before adding complex features.
* **Model (`/core`):** Pure logic. Contains `VegaChimpCore` (Options), `GARCH` (Vol), and the new `ArbEngine`. *No UI code allowed here.*
* **View (`/ui`):** Pure Tkinter. Handles windows, buttons, and charts. It observes the Model but calculates nothing.
* **Controller (`/main`):** Handles user inputs (e.g., "User clicked Scan") and coordinates data fetching.

### 1.2 Data Abstraction Layer
* **Objective:** Decouple the app from `yfinance`.
* **Implementation:** Create a `DataProvider` interface.
    * *Current:* `YFinanceProvider` (Free, Delayed).
    * *Future:* `PolygonProvider` or `IBKR_API` (Paid, Real-time).
    * *Benefit:* Allows the "Arbitrage Engine" to run on real-time data later without rewriting the whole app.

---

## 🧠 Track 2: The Quant Engine (New Features)

### 2.1 The "Probability Cone" (Vol-Adjusted Charting)
* **Goal:** Visualize whether current price action is "normal" or "extreme" relative to forecast volatility.
* **Math:**
    $$\text{Upper} = P_0 \times \exp(+\sigma \sqrt{t/252})$$
    $$\text{Lower} = P_0 \times \exp(-\sigma \sqrt{t/252})$$
* **Visual:** Plot these bounds as a translucent "Cone" extending 30 days right of the current price.
    * *Signal:* If price breaks the cone **without** news, Mean Reversion is likely.

### 2.2 The "Fundamental Bias" Filter (Revised)
* **Goal:** Use fundamentals to filter/rank option opportunities, rather than distorting the pricing math.
* **Implementation:**
    * Calculate a **Fundamental Z-Score** based on 5 factors: `P/E`, `PEG`, `Debt/Eq`, `RevGrowth`, `EarningsAccel`.
    * **The Logic:**
        * If `Math_Edge > 0` AND `Fund_Score > 70` → **Strong Buy (🟢)**
        * If `Math_Edge > 0` AND `Fund_Score < 30` → **Value Trap Warning (⚠️)**

### 2.3 The "Semantic Arbitrage" Scanner (The Killer App)
* **Concept:** Trade a stock against its "True Peers" defined by AI, not rigid sectors.
* **Workflow:**
    1.  **Vector Database:** On startup, FinBERT encodes the "Business Summary" of the S&P 500 into 768-dim vectors.
    2.  **Cluster:** User types "NVDA". System finds the 10 nearest vectors (Cosine Similarity).
    3.  **Spread Tracking:** Calculate the custom index of that basket.
    4.  **Signal:** Alert when Target Stock diverges $> 2\sigma$ from its Semantic Basket.

---

## 🎨 Track 3: UX & Quality of Life

### 3.1 "Set & Forget" Scanners
* **Feature:** Automated background scanning.
* **Logic:** User sets criteria (e.g., "Notify me if SPY Puts > 5% Edge"). The app runs the `scan_options()` subroutine silently every 15 minutes.
* **Notification:** System Tray bubble or discord webhook.

### 3.2 Economic Context Layer
* **Feature:** Vertical lines on the chart for key events.
* **Data:** FOMC Meetings, CPI Releases, Earnings Dates.
* **Visual:** dashed vertical lines colored by impact (Red = High Impact).

### 3.3 Dynamic Watchlist
* **Feature:** Persistence for tracked opportunities.
* **Storage:** Simple `watchlist.json` file.
* **Action:** When app loads, it auto-refreshes data for saved tickers/contracts.

---

## 🗓️ Execution Phases

| Phase | Name | Focus | Key Deliverable |
| :--- | :--- | :--- | :--- |
| **I** | **The Clean Up** | Refactoring | MVC Architecture, `requirements.txt` update. |
| **II** | **The Eyes** | Charting | Probability Cones, Econ Calendar. |
| **III** | **The Brain** | Alpha Logic | Semantic Embeddings (FinBERT), Arb Engine. |
| **IV** | **The Automaton** | Automation | Background Scanning, Watchlists. |