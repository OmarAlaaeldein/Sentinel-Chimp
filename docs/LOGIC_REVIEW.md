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
| Vol blend for fair value | `fetch_options_batch` | Time-weighted IV/HV; **pricing semantics unchanged** |

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

- Vectorized / batch American pricing across an entire chain (still scalar BS2002 per contract).
- Fitted GARCH(1,1) or local-vol smile.
- Peel chart/options UI out of `main/app.py` (Phase I follow-up).
- American Greeks / analytic BS2002 greeks from the 2002 paper (omitted there for space; numerical bump possible later).
