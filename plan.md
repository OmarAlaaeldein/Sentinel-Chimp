# PyStock-Sentinel: Technical Indicators & Option Pricing Audit Plan

A line-by-line audit of `sentinel.py` identifying every bug, missing feature, and mathematical
flaw in the technical indicators and option pricing engine.

---

## PART A: BUGS (Fix Immediately)

### A1. Dividend Yield Division Bug (Line 1862)
**File:** `sentinel.py:1862`
**Severity:** Critical (option mispricing)

`yfinance`'s `dividendYield` is already in decimal form (e.g., `0.0294` for 2.94%).
The code divides by 100 again, turning it into `0.000294`. This under-reports dividend
yield by 100x, causing **every American put** to be mispriced (lower put value than correct).

```python
# CURRENT (BROKEN)
div = info.get('dividendYield') or info.get('trailingAnnualDividendYield')
div = div / 100  # <-- WRONG for dividendYield, only possibly needed for trailingAnnualDividendYield

# FIX: Handle each field correctly
div = info.get('dividendYield')
if div is None:
    div = info.get('trailingAnnualDividendYield')
    if div is not None and div > 1:  # Likely in percentage form
        div = div / 100
```

### A2. FinBERT Label Order May Be Wrong (Line 130-131)
**File:** `sentinel.py:130-131`
**Severity:** Medium (sentiment inversion risk)

FinBERT (ProsusAI/finbert) outputs logits in the order: `[positive, negative, neutral]`.
The code assumes `probs[:, 0]` is positive and `probs[:, 1]` is negative. **This is correct**
for ProsusAI/finbert, but the code has no assertion or check. If the model is ever swapped
(the class supports multiple models), scores silently flip.

**Fix:** Add a validation check in `load_model()`:
```python
# Verify label order after loading
labels = target["model"].config.id2label
assert labels[0] == "positive" and labels[1] == "negative", f"Unexpected label order: {labels}"
```

### A3. Dead Growth Drift Code (Line 2109)
**File:** `sentinel.py:2109`
**Severity:** Low (unused code, potential confusion)

`annual_growth_offset` is computed (lines 1170-1197) but **never applied** to option pricing.
Line 2109 shows `adjusted_rfr = RFR` with no modification. Either remove the growth drift
calculation entirely, or apply it as intended (e.g., drift-adjusted forward price).

### A4. No Exception Guard on `div / 100` When div is None
**File:** `sentinel.py:1861-1862`
**Severity:** Medium (crash on stocks with no dividend info)

If both `info.get('dividendYield')` and `info.get('trailingAnnualDividendYield')` return `None`,
the `or` expression yields `None`, and `None / 100` raises `TypeError`. The `except` block
catches it and returns `0.0`, but this silently swallows a division error that masks the root
cause during debugging.

---

## PART B: TECHNICAL INDICATOR FLAWS

### B1. VWAP Never Resets (Line 308)
**File:** `sentinel.py:306-308`
**Severity:** High

VWAP must reset at the start of each trading session. The current cumulative VWAP across the
entire dataset becomes meaningless over multi-day periods. On 1Y data, early bars utterly
dominate the calculation and the current VWAP gap reading is unreliable.

**Fix:**
- For intraday data (1D, 5D): group by trading date, reset cumsum each day, display only the
  current day's VWAP.
- For daily data (1M+): switch to Anchored VWAP (start from period beginning) **or** disable
  the VWAP gap indicator for non-intraday views and label it as "Intraday Only".

### B2. StochRSI Missing %K/%D Smoothing (Line 298-301)
**File:** `sentinel.py:298-301`
**Severity:** Medium

The standard Stochastic RSI applies a 3-period SMA to get %K, then a 3-period SMA of %K
to get %D (signal line). Raw un-smoothed StochRSI is extremely noisy and generates many
false signals.

```python
# CURRENT (raw, noisy)
df['StochRSI'] = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)

# FIX: Add smoothing
raw_stoch = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)
df['StochRSI_K'] = raw_stoch.rolling(3).mean()   # %K
df['StochRSI_D'] = df['StochRSI_K'].rolling(3).mean()  # %D (signal)
df['StochRSI'] = df['StochRSI_K']  # Display %K as main value
```

### B3. MACD Histogram Not Computed (Line 278-282)
**File:** `sentinel.py:278-282`
**Severity:** Medium

MACD Histogram (`MACD - Signal`) is one of the most used components of the MACD system.
It shows momentum direction changes earlier than the MACD line itself. Currently missing.

```python
df['MACD_Hist'] = df['MACD'] - df['Signal']
```

Display: show histogram value and whether it's increasing/decreasing (momentum shifting).

### B4. Bollinger Band Width / %B Not Computed
**File:** `sentinel.py:284-288`
**Severity:** Low

Two useful derived metrics from Bollinger Bands are missing:
- **%B**: `(Close - BB_Lower) / (BB_Upper - BB_Lower)` - normalized position within bands (0=lower, 1=upper)
- **Bandwidth**: `(BB_Upper - BB_Lower) / SMA_20` - measures volatility squeeze/expansion

Bandwidth squeeze (narrow bands) often precedes explosive moves. This is a standard signal
that should be flagged.

### B5. ADX Does Not Distinguish +DI/-DI Crossovers
**File:** `sentinel.py:315-325`
**Severity:** Medium

The ADX display only shows trend strength (Strong/Weak) but ignores the directional component.
The whole point of DI is to show direction: `+DI > -DI` = bullish trend, `-DI > +DI` = bearish
trend. The crossover is a key trading signal.

**Fix:** Display as "25.3 Strong (Bullish)" or "25.3 Strong (Bearish)" based on +DI vs -DI.

---

## PART C: OPTION PRICING ENGINE FLAWS

### C1. Bjerksund-Stensland is the 1993 Version, Not 2002 (Line 184)
**File:** `sentinel.py:183-267`
**Severity:** High (pricing accuracy)

The docstring says "2002" but the implementation uses a single exercise boundary, which is
the **1993 approximation**. The 2002 version splits the time interval at a midpoint, computes
**two** exercise boundaries (`I1` at `T/2` and `I2` at `T`), and sums two single-boundary
components. This gives significantly better accuracy for long-dated options.

**Fix:** Implement the actual 2002 two-boundary version:
1. Split at `t1 = T/2`
2. Compute `I2` (boundary at time `T`) using the existing single-boundary formula
3. Compute `I1` (boundary at time `t1`) similarly
4. Price = `alpha2 * S^beta` contribution from `[t1, T]` plus first-interval adjustment
5. Reference: Bjerksund & Stensland, "Closed Form Valuation of American Options" (2002),
   Discussion Paper 2002/09.

### C2. GARCH Parameters Are Not Fitted (Line 172-181)
**File:** `sentinel.py:172-181`
**Severity:** High (volatility forecast quality)

The GARCH(1,1) uses hardcoded `alpha=0.05, beta=0.94, omega=1e-6` (the RiskMetrics/J.P.Morgan
defaults from the 1990s). These are NOT fitted to the actual return series. This makes the
"GARCH forecast" essentially just an exponentially-weighted variance with decay factor 0.94.

**Two options:**

**Option A (Recommended - Quick Fix):** Rename the display from "GARCH" to "EWMA Vol" to
avoid misleading users. The calculation is valid as an EWMA volatility estimate; it's just
not a fitted GARCH model.

**Option B (Proper Fix):** Fit GARCH parameters via MLE using the `arch` library:
```python
from arch import arch_model
am = arch_model(returns * 100, vol='GARCH', p=1, q=1, dist='normal')
res = am.fit(disp='off')
forecast = res.forecast(horizon=1)
annualized_vol = np.sqrt(forecast.variance.iloc[-1].values[0] / 10000 * 252)
```

### C3. No Greeks Computed or Displayed
**File:** Not present (entirely missing)
**Severity:** Critical (core feature gap)

Greeks are fundamental to options analysis. Every serious options tool displays them.

**Required Greeks (all computable from Black-Scholes closed-form):**
- **Delta** = `e^(-qT) * N(d1)` for calls, `e^(-qT) * (N(d1) - 1)` for puts
- **Gamma** = `e^(-qT) * n(d1) / (S * sigma * sqrt(T))` where n(x) is the standard normal PDF
- **Theta** = `-(S * n(d1) * sigma * e^(-qT)) / (2*sqrt(T)) - r*K*e^(-rT)*N(d2)` (calls)
- **Vega** = `S * e^(-qT) * n(d1) * sqrt(T)` (same for calls and puts)
- **Rho** = `K * T * e^(-rT) * N(d2)` for calls

**Implementation:**
1. Add a `bs_greeks(S, K, r, q, sig, T, kind)` method to `VegaChimpCore`
2. Add columns to the options scanner treeview: Delta, Gamma, Theta, Vega
3. Color-code theta (red = heavy decay) and delta (for position sizing)

### C4. No Implied Volatility Solver
**File:** Not present
**Severity:** High

The code trusts yfinance's IV blindly. A proper options engine should have its own IV solver
to cross-validate or compute IV from the model price. This also enables:
- Detecting stale or incorrect IV from the data provider
- Computing IV surfaces from raw option prices
- Pricing exotic options where IV isn't provided

**Implementation:** Newton-Raphson IV solver:
```python
@staticmethod
def implied_vol(market_price, S, K, r, q, T, kind, tol=1e-6, max_iter=100):
    sig = 0.3  # Initial guess
    for _ in range(max_iter):
        price = VegaChimpCore.bs_price(S, K, r, q, sig, T, kind)
        vega = S * math.exp(-q*T) * VegaChimpCore.n(d1) * math.sqrt(T)
        if abs(vega) < 1e-12:
            break
        sig -= (price - market_price) / vega
        if abs(price - market_price) < tol:
            return sig
    return sig
```

### C5. Volatility Blend is Arbitrary (Line 2098-2106)
**File:** `sentinel.py:2098-2106`
**Severity:** Medium

The 50/50 blend of IV and HV_30 has no theoretical basis. For short-dated options, realized
vol matters more. For long-dated options, implied vol reflects market consensus better.

**Fix:** Time-weighted blending:
```python
# Weight IV more for longer-dated, HV more for shorter-dated
iv_weight = min(T * 4, 0.8)  # Caps at 80% IV weight at T >= 0.2yr (~73 days)
hv_weight = 1.0 - iv_weight
vol_input = iv_weight * iv + hv_weight * hv_30
```

### C6. No Volatility Smile/Skew Awareness
**File:** Not present
**Severity:** Medium

All options use a single flat volatility regardless of moneyness. In reality, deep OTM puts
have much higher IV than ATM options (the "skew"). Pricing far-from-ATM options with ATM
volatility systematically misprices them.

**Minimum fix:** When scanning a chain, compute a simple quadratic smile fit:
`IV(K) = a + b*(K/S - 1) + c*(K/S - 1)^2` from the market-observed IVs, then use the
fitted IV for each strike instead of the raw provider value.

### C7. Risk-Free Rate Uses Single Tenor (Line 2006-2015)
**File:** `sentinel.py:2006-2015`
**Severity:** Medium

All options regardless of maturity use the 13-week T-Bill rate. A 2-year LEAP uses the same
rate as a weekly option. For proper pricing:

| Option Maturity | Appropriate Rate |
|----------------|-----------------|
| < 3 months     | ^IRX (13-week) -- current |
| 3-6 months     | 6-month T-Bill |
| 6-12 months    | ^FVX (5-year note) interpolated |
| 1-2+ years     | ^TNX (10-year note) |

**Minimum fix:** Interpolate between ^IRX and ^TNX based on time to expiry.

### C8. Time-to-Expiry Uses Calendar Days (Line 2075)
**File:** `sentinel.py:2075`
**Severity:** Medium (significant for short-dated options)

```python
T = (datetime.strptime(date, "%Y-%m-%d") - datetime.now()).days / 365.0
```

This uses calendar days / 365. For options expiring this week, weekends inflate T by ~40%.
Standard practice for short-dated options:
- Use trading days / 252, not calendar days / 365
- Account for remaining market hours on the current day

```python
from numpy import busday_count
exp_date = datetime.strptime(date, "%Y-%m-%d").date()
today = datetime.now().date()
trading_days = np.busday_count(today, exp_date)
T = max(trading_days / 252.0, 1/252)  # Minimum 1 trading day
```

### C9. No Put-Call Parity Validation
**File:** Not present
**Severity:** Medium

Put-call parity (`C - P = S*e^(-qT) - K*e^(-rT)` for Europeans) provides a model-free
arbitrage check. Significant violations indicate:
1. A real arbitrage opportunity (rare but valuable)
2. Stale/bad data from the provider (common)
3. An error in the pricing logic

**Fix:** For each strike where both call and put exist, compute the parity residual.
Flag any residual > $0.10 as a data warning.

### C10. No Probability of Profit (POP)
**File:** Not present
**Severity:** Medium

POP is the probability that a position is profitable at expiration. For a long call:
`POP = N(-d2)` where d2 uses the market IV and strike = breakeven.

This is one of the first things options traders look at. Add as a column in the scanner.

---

## PART D: MISSING TECHNICAL INDICATORS

### D1. Williams %R
Momentum oscillator similar to StochRSI. Range: -100 to 0.
```python
highest = df['High'].rolling(14).max()
lowest = df['Low'].rolling(14).min()
df['Williams_R'] = -100 * (highest - df['Close']) / (highest - lowest)
```
Oversold < -80, Overbought > -20.

### D2. CCI (Commodity Channel Index)
Mean-reversion indicator. Standard 20-period:
```python
TP = (df['High'] + df['Low'] + df['Close']) / 3
df['CCI'] = (TP - TP.rolling(20).mean()) / (0.015 * TP.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean()))
```

### D3. Ichimoku Cloud (Optional - Complex)
Comprehensive trend system. Requires 5 lines:
- Tenkan-sen (9-period midpoint)
- Kijun-sen (26-period midpoint)
- Senkou Span A (midpoint of Tenkan/Kijun, shifted 26 periods forward)
- Senkou Span B (52-period midpoint, shifted 26 periods forward)
- Chikou Span (Close shifted 26 periods back)

Consider adding as a chart overlay toggle, not a panel indicator.

### D4. Fibonacci Retracement Levels
Auto-compute from period high/low:
```python
high = df['High'].max()
low = df['Low'].min()
levels = {
    '23.6%': high - 0.236 * (high - low),
    '38.2%': high - 0.382 * (high - low),
    '50.0%': high - 0.500 * (high - low),
    '61.8%': high - 0.618 * (high - low),
}
```
Draw as horizontal dashed lines on chart.

---

## PART E: ROBUSTNESS & TESTING

### E1. Restore and Expand Test Suite
**Severity:** Critical

The test directory is empty (tests removed in last commit). Need tests for:

**Technical Indicator Tests (known-value checks):**
```
test_rsi_known_values()        - RSI of [44,44.34,44.09,...] should be ~70.53
test_macd_crossover()          - Verify signal detection on synthetic data
test_bollinger_band_width()    - Verify band width calculation
test_atr_wilder_smoothing()    - Compare with manual Wilder's calculation
test_stochrsi_range()          - Must always be in [0, 1]
test_adx_strong_trend()        - Synthetic trending data should give ADX > 25
test_obv_direction()           - Rising prices + volume = rising OBV
```

**Option Pricing Tests (analytical benchmarks):**
```
test_bs_call_put_parity()      - C - P = S*e^(-qT) - K*e^(-rT) within tolerance
test_bs_known_values()         - S=100, K=100, r=0.05, q=0, sig=0.2, T=1 -> C=10.4506
test_bs_deep_itm()             - Deep ITM call ~ S - K*e^(-rT)
test_bs_deep_otm()             - Deep OTM call ~ 0
test_bs_edge_cases()           - T=0, sig=0, S=K edge cases
test_american_call_no_div()    - American call = European call when q=0
test_american_put_premium()    - American put >= European put
test_american_vs_bs_bounds()   - American price >= intrinsic value
test_bjerksund_published()     - Compare against published benchmark tables
```

**Integration Tests:**
```
test_garch_positive_output()   - Forecast should always be > 0
test_iv_solver_roundtrip()     - price -> IV -> price should roundtrip
test_greeks_sum_to_zero()      - Portfolio-level Greek checks
```

### E2. Thread Safety
**Severity:** Medium

`self.scan_data`, `self.data_cache`, `self.sent_cache`, and `self.valuation_cache` are
read/written from multiple threads without locks. Under heavy concurrent use (e.g., scanner
running while user switches tickers), this can cause data corruption.

**Fix:** Use `threading.Lock()` for each shared cache:
```python
self._scan_lock = threading.Lock()
# Then in fetch_options_batch:
with self._scan_lock:
    self.scan_data.append({...})
```

### E3. No Open Interest or Spread Analysis in Scanner
**Severity:** Low-Medium

The scanner filters on volume but ignores:
- **Open Interest**: Low OI = illiquid, hard to exit position
- **Bid-Ask Spread**: Wide spread = hidden cost that eats into the theoretical edge
- **Spread as % of Price**: A $0.10 spread on a $0.50 option is 20% slippage

**Fix:** Add columns for OI, Spread, and Spread%. Filter out options where spread > 15%
of midpoint price.

---

## PART F: EXECUTION PRIORITY

| Priority | Item | Impact | Effort |
|----------|------|--------|--------|
| 1 | A1: Fix dividend yield bug | Critical pricing fix | 10 min |
| 2 | A3: Remove or apply dead drift code | Code hygiene | 5 min |
| 3 | A4: Guard div/100 None crash | Crash prevention | 5 min |
| 4 | C3: Add Greeks (Delta, Gamma, Theta, Vega) | Core feature gap | 2-3 hrs |
| 5 | C8: Fix T calculation (trading days) | Pricing accuracy | 20 min |
| 6 | E1: Restore test suite | Quality gate | 3-4 hrs |
| 7 | C1: Upgrade to actual BS2002 | Pricing accuracy | 2-3 hrs |
| 8 | B1: Fix VWAP daily reset | Indicator correctness | 1 hr |
| 9 | C2: Rename GARCH to EWMA or fit properly | Honesty/accuracy | 30 min / 2 hrs |
| 10 | B2: Add StochRSI smoothing | Signal quality | 15 min |
| 11 | B3: Add MACD Histogram | Easy win | 10 min |
| 12 | B5: ADX directional display (+DI/-DI) | Easy win | 15 min |
| 13 | C4: Add IV solver | Advanced pricing | 2 hrs |
| 14 | C7: Term-matched risk-free rate | Pricing accuracy | 1 hr |
| 15 | C5: Time-weighted vol blend | Pricing accuracy | 30 min |
| 16 | C9: Put-call parity check | Data validation | 1 hr |
| 17 | C10: Add POP column | Trader utility | 30 min |
| 18 | A2: Validate FinBERT label order | Safety check | 10 min |
| 19 | E3: Add OI and spread columns | Trader utility | 1 hr |
| 20 | E2: Add threading locks | Robustness | 30 min |
| 21 | B4: Bollinger %B and Bandwidth | Signal enhancement | 15 min |
| 22 | D1-D4: Additional indicators | Feature expansion | 2-3 hrs |
| 23 | C6: Volatility smile fitting | Advanced pricing | 3-4 hrs |

---

## Summary

- **3 bugs** that actively cause incorrect results (A1 dividend is the worst)
- **5 technical indicator flaws** in existing code (VWAP reset is the worst)
- **10 option pricing gaps**, from missing Greeks to incorrect model version
- **4 missing indicators** commonly expected in a technical analysis tool
- **3 robustness items** including the empty test suite

Total estimated effort: ~25-30 hours for everything, or ~5-6 hours for just priorities 1-12
(the highest-impact items).
