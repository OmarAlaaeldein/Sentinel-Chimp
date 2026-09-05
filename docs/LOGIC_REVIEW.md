# Sentinel-Chimp Logic Review (2026-09)

Research-backed mapping of pricing / technical / sentiment logic to published
sources, correctness spot-checks, and performance/memory changes applied on
`main`.

## Papers & sources consulted

| Source | Year | How used |
| :--- | :---: | :--- |
| Black & Scholes, “The Pricing of Options and Corporate Liabilities” | 1973 | European `bs_price` / `bs_greeks` baseline |
| Merton, “Theory of Rational Option Pricing” | 1973 | Continuous dividend yield `q` in BSM |
| Bjerksund & Stensland, “Closed Form Valuation of American Options,” NHH Discussion Paper 2002/09 | 2002 | Two-boundary American approx.; Prop. 1 `φ`/`Ψ`; golden-ratio split `t=½(√5−1)T`; Tables 1–3 regression |
| Bjerksund & Stensland (Scandinavian Journal of Management Suppl.) | 1993a | Flat-boundary `φ` building block reused by 2002 |
| Bjerksund & Stensland; McDonald & Schroder | 1993b / 1998 | American put ↔ call transformation `P(S,K,r,b)=C(K,S,r−b,−b)` |
| Haug, *The Complete Guide to Option Pricing Formulas* | 1997 | Bivariate normal CDF quadrature reference cited by BS2002 |
| J.P. Morgan / RiskMetrics Technical Document | 1996 | EWMA λ=0.94 daily variance recursion |
| Wilder, *New Concepts in Technical Trading Systems* | 1978 | RSI, ATR, ADX / ±DI smoothing |
| Appel | 1979 | MACD 12/26/9 |
| Bollinger | ~1980s | Bollinger bands; %B / bandwidth derived |
| Chande & Kroll | 1994 | Stochastic RSI with %K/%D SMA(3) |
| Granville | 1963 | On-Balance Volume |
| Williams | 1973 | Williams %R (14) |
| Lambert | 1980 | CCI (20, 0.015) |
| Araci / ProsusAI FinBERT (arXiv:1908.10063 lineage; HuggingFace `ProsusAI/finbert`) | 2019 | Financial sentiment encoder; label map from `id2label` |
| Engle, “Autoregressive Conditional Heteroscedasticity…” | 1982 | ARCH foundation for `garch11_vol_forecast` |
| Bollerslev, “Generalized Autoregressive Conditional Heteroskedasticity” | 1986 | GARCH(1,1) variance-targeted QMLE |
| Gatheral, *The Volatility Surface* | 2006 | Smile phenomenology; code uses OLS quadratic in log-moneyness (not SVI) |
| Abramowitz & Stegun 7.1.26 | 1964 | Vectorized `erf` for batch CDF |

No fabricated citations were introduced.

## Formula → code map

| Model / formula | Module | Notes |
| :--- | :--- | :--- |
| BSM European price & Greeks | `core/pricing.py` `bs_price`, `bs_greeks` | Continuous `q`; theta per calendar day; vega/rho per 1% |
| IV (European) | `implied_vol` | Bracketed bisection; rejects no-arb violations |
| BS2002 American | `bjerksund_stensland` | Two barriers `X=X_T`, `x=X_{T-t}`; put via transform; floors at European/intrinsic |
| Bivariate CDF `M` | `_M` | 10-point Gauss–Legendre on Drezner/Haug integral |
| American C−P bounds | `american_put_call_parity_bounds` | Inequality bounds (not European equality) |
| EWMA vol | `ewma_vol_forecast` | λ=0.94; init at sample variance; annualized `√(var·252)` |
| Technicals | `core/technicals.py` | Wilder / Appel / Bollinger / StochRSI / VWAP daily reset / OBV / ADX / %R / CCI |
| FinBERT | `core/sentiment.py` | Lazy load; `id2label` → pos/neg indices; `eval()` + `no_grad` |
| Term RFR | `main/app.py` + `YFinanceProvider.fetch_rate_curve` | ^IRX / ^TNX interpolate by T |
| Vol blend for fair value | `fetch_options_batch` | Time-weighted IV/HV; **pricing semantics unchanged** unless smile/GARCH flags on |
| Batch BS2002 | `bjerksund_stensland_batch` | Vectorized φ/Ψ/M; scanner uses when n≥64 |
| American FD Greeks | `american_greeks(_batch)` | Δ/Γ/ν/Θ on BS2002; default in scanner |
| GARCH(1,1) | `core/vol_models.py` | Optional; display always when fit ok |
| Quadratic smile | `core/vol_models.py` | Optional IV smoother per expiry |

## Correctness findings

1. **BS2002 matches paper Tables 1–3** within ~0.005 (paper prints 2 decimals). Prop. 1 term structure, golden-ratio split, barrier assignment (`X` first period, `x` second), and put transform all verified against the PDF.
2. **No pricing-semantics changes** were required. Existing European floors / intrinsic clamps remain (consistent with American dominance and BS2002 being a feasible-strategy lower bound).
3. **Dividend normalization** (`_normalize_div_yield`) already handles yfinance percent-vs-decimal ambiguity; left as-is.
4. **IV solver** is bisection (not Newton); numerically robust and covered by round-trip tests.
5. **Technicals** cross-check suite remains the regression oracle; vectorized CCI MAD matches prior `rolling.apply` definition.
6. **Deferred (not bugs):** fitted GARCH, vol smile, American Greeks (scanner still shows European BS Greeks on market IV), FinBERT already raises on unexpected labels.

## Performance / memory changes

| Area | Change | Measured effect (box microbench) |
| :--- | :--- | :--- |
| EWMA | Closed-form weighted sum (exact recursion) instead of Python `for` | ~0.184 ms → ~0.019 ms / 1000 returns (~10×) |
| BS2002 | Hoisted `_phi`/`_psi`; module constants; tight Python `_M` loop | ~0.063 ms → ~0.050 ms / price (modest; correctness-first) |
| Technicals | Vectorized rolling MAD for CCI; reuse TP; fewer helper columns | ~11.9 ms → ~8.0 ms / 500 bars (~1.5×) |
| Data | TTL cache for ^IRX/^TNX (~5 min) and option chains (~60 s) | Removes repeat yfinance hits within a scan session |
| Options scan | Vectorized mid/spread; index loop vs `iterrows`; batched `tree.insert` (40) | Lower CPU + far fewer UI-thread callbacks |
| FinBERT | `model.eval()` after load (with existing `no_grad`) | Avoids dropout / train-mode overhead |

Pricing semantics (fair value, EV threshold, vol blend weights, dividend handling) were **not** altered for speed.

## Tests

- Existing: `tests/test_option_pricing.py`, `tests/test_technicals.py`, `tests/test_technicals_crosscheck.py` — kept green.
- Added: `TestBS2002PaperTables` — published Table 1/2/3 anchors.

## Deferred

- Peel chart/options UI out of `main/app.py` (Phase I follow-up) — still open.
- Analytic BS2002 Greeks from the 2002 paper (paper omitted them; we ship FD instead).
- Cross-expiry mega-batch (currently batch-per-expiry with n≥64 threshold).
- Full SVI / local-vol smile (quadratic OLS kept as the practical lite path).

## Deferred follow-up (2026-09-05) — experiments landed

### 1. Batch / vector BS2002 — **kept**

- Added `bjerksund_stensland_batch`, vectorized `_M` / `_φ` / `_ψ` / American-call
  core in `core/pricing.py` (Abramowitz–Stegun 7.1.26 `erf` ≈ 1.4e-7; price
  vs scalar max |Δ| ≈ 2e-5).
- Scanner (`fetch_options_batch`) uses batch when ≥64 eligible contracts per
  expiry; scalar below that (numpy overhead otherwise wins).
- Microbench (box):

  | n | batch | scalar | speedup |
  | ---: | ---: | ---: | ---: |
  | 100 | 4.7 ms | 5.0 ms | ~1.1× |
  | 500 | 6.5 ms | 27.8 ms | ~4.3× |
  | 1000 | 8.3 ms | 51.0 ms | ~6.1× |

### 2. GARCH(1,1) + quadratic smile — **kept (optional flags)**

| Piece | Module | Default |
| :--- | :--- | :--- |
| Variance-targeted GARCH(1,1) QMLE | `core/vol_models.garch11_vol_forecast` | Displayed next to EWMA; **not** in FV unless `use_garch_blend` |
| Quadratic smile `σ(k)=a+bk+ck²`, `k=log(K/F)` | `fit_quadratic_smile` / `smile_vol*` | Off unless `use_smile_vol` |

Citations (real): Engle (1982) ARCH; Bollerslev (1986) GARCH; Gatheral (2006)
for smile phenomenology (fit itself is OLS quadratic, **not** SVI).

EWMA (RiskMetrics λ=0.94) remains the default historical-vol path. Flags on
`MarketApp`: `use_garch_blend=False`, `use_smile_vol=False`.

GARCH fit cost ≈ 8 ms / 252 returns (coarse grid + coordinate refine; no SciPy).

### 3. American Greeks — **kept**

- `american_greeks` / `american_greeks_batch`: central FD on BS2002
  (Δ/Γ/ν/Θ; Θ = one calendar-day bump; ν per 1% vol).
- Wired into scanner when `use_american_greeks=True` (default). Set False to
  restore European `bs_greeks`.
- Sanity: q=0 American call FD Greeks ≈ analytic BS; batch matches scalar FD.
- Bench n=200: American batch FD ~25 ms vs scalar FD ~82 ms (~3.3×); European
  analytic still ~0.2 ms if flag off.

### 4. Further UI peel — **dropped (time)**

Left for a later Phase I follow-up; did not block 1–3.

### How to exercise

```python
from core.pricing import VegaChimpCore
from core.vol_models import garch11_vol_forecast, fit_quadratic_smile

# Batch American prices
VegaChimpCore.bjerksund_stensland_batch(100, [90,100,110], 0.5, 0.05, 0.02, [0.25]*3, 'put')

# American FD Greeks
VegaChimpCore.american_greeks(100, 100, 0.05, 0.02, 0.25, 1.0, 'put')

# GARCH / smile (optional)
vol, info = garch11_vol_forecast(log_returns)
coef = fit_quadratic_smile(strikes, ivs, forward=100.0)

# In the GUI (after Load Data):
#   app.use_garch_blend = True
#   app.use_smile_vol = True
#   app.use_american_greeks = True  # default
```

Tests: `tests/test_vol_models_and_batch.py` (+ prior suite) — 129 passed.
