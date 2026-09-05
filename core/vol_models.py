"""Volatility models beyond RiskMetrics EWMA.

References
----------
- Engle, R. F. (1982). "Autoregressive Conditional Heteroscedasticity with
  Estimates of the Variance of United Kingdom Inflation." *Econometrica*
  50(4), 987–1007. — ARCH foundation.
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional
  Heteroskedasticity." *Journal of Econometrics* 31, 307–327. — GARCH(1,1).
- Gatheral, J. (2006). *The Volatility Surface* — smile/skew phenomenology;
  we use a simple quadratic-in-log-moneyness fit (industry practice; not SVI).

Feature flags (callers): keep EWMA as the default historical-vol path; blend
GARCH and/or smile only when explicitly enabled so scan semantics stay stable.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def garch11_vol_forecast(
    log_returns,
    days: int = 252,
    max_iter: int = 80,
) -> Tuple[float, dict]:
    """Fit a variance-targeted GARCH(1,1) and return annualized σ forecast.

    Model (Bollerslev 1986)::

        σ²_t = ω + α r²_{t-1} + β σ²_{t-1}

    with variance targeting ``ω = σ̄² (1 − α − β)`` so the unconditional
    variance equals the sample variance. ``(α, β)`` are chosen by a coarse
    grid + coordinate refinement maximizing the Gaussian QMLE. No SciPy
    dependency.

    Returns
    -------
    vol : float
        Annualized volatility ``sqrt(σ²_T * days)``, or 0.0 on failure.
    info : dict
        Keys: alpha, beta, omega, var_last, n, ok.
    """
    info = {"alpha": math.nan, "beta": math.nan, "omega": math.nan,
            "var_last": math.nan, "n": 0, "ok": False}
    try:
        returns = np.asarray(log_returns, dtype=np.float64).reshape(-1)
        returns = returns[np.isfinite(returns)]
        n = int(returns.size)
        info["n"] = n
        if n < 60:
            return 0.0, info

        r2 = returns * returns
        var_bar = float(np.mean(r2))
        if var_bar <= 0.0 or not math.isfinite(var_bar):
            return 0.0, info

        def nll(alpha: float, beta: float) -> float:
            if alpha < 0.0 or beta < 0.0 or alpha + beta >= 0.999:
                return 1e300
            omega = var_bar * (1.0 - alpha - beta)
            if omega <= 0.0:
                return 1e300
            var = var_bar
            ll = 0.0
            for i in range(n):
                # predict var_t from r2_{t-1}; use var as current forecast
                if var <= 1e-18:
                    return 1e300
                ll += math.log(var) + r2[i] / var
                var = omega + alpha * r2[i] + beta * var
            return ll

        # Coarse grid (α, β) on the stationarity triangle
        best = (1e300, 0.05, 0.90)
        for alpha in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
            for beta in (0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94):
                if alpha + beta >= 0.995:
                    continue
                score = nll(alpha, beta)
                if score < best[0]:
                    best = (score, alpha, beta)

        # Coordinate refinement
        score, alpha, beta = best
        step = 0.02
        for _ in range(max_iter):
            improved = False
            for da, db in (
                (step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step),
                (step, step), (-step, -step), (step, -step), (-step, step),
            ):
                a2, b2 = alpha + da, beta + db
                s2 = nll(a2, b2)
                if s2 < score:
                    score, alpha, beta = s2, a2, b2
                    improved = True
            if not improved:
                step *= 0.5
                if step < 1e-4:
                    break

        omega = var_bar * (1.0 - alpha - beta)
        var = var_bar
        for i in range(n):
            var = omega + alpha * r2[i] + beta * var
        if var <= 0.0 or not math.isfinite(var):
            return 0.0, info

        vol = float(math.sqrt(var * days))
        info.update({
            "alpha": float(alpha),
            "beta": float(beta),
            "omega": float(omega),
            "var_last": float(var),
            "ok": True,
        })
        return vol, info
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return 0.0, info


def fit_quadratic_smile(
    strikes,
    ivs,
    forward: float,
    min_points: int = 5,
) -> Optional[Tuple[float, float, float]]:
    """Fit ``σ(k) = a + b k + c k²`` with ``k = log(K / F)``.

    Ordinary least squares via ``numpy.linalg.lstsq``. Returns ``(a, b, c)``
    or ``None`` if the fit is under-determined / degenerate.

    This is a local parametric smile (common screening tool), **not**
    Gatheral SVI and **not** an arbitrage-free density. Suitable for
    smoothing noisy listed IVs inside a single expiry.
    """
    strikes = np.asarray(strikes, dtype=np.float64).reshape(-1)
    ivs = np.asarray(ivs, dtype=np.float64).reshape(-1)
    if forward <= 0 or strikes.size != ivs.size:
        return None
    mask = (
        np.isfinite(strikes) & np.isfinite(ivs)
        & (strikes > 0) & (ivs > 0.01) & (ivs < 5.0)
    )
    strikes = strikes[mask]
    ivs = ivs[mask]
    if strikes.size < min_points:
        return None
    k = np.log(strikes / forward)
    # Design matrix [1, k, k^2]
    A = np.column_stack([np.ones(k.size), k, k * k])
    try:
        coef, _, rank, _ = np.linalg.lstsq(A, ivs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:
        return None
    a, b, c = (float(x) for x in coef)
    if not all(math.isfinite(v) for v in (a, b, c)):
        return None
    if a <= 0.0:
        return None
    return a, b, c


def smile_vol(strike: float, forward: float, coef: Tuple[float, float, float],
              floor: float = 0.01, cap: float = 5.0) -> float:
    """Evaluate quadratic smile at a strike; clamp to ``[floor, cap]``."""
    a, b, c = coef
    if strike <= 0 or forward <= 0:
        return a
    k = math.log(strike / forward)
    vol = a + b * k + c * k * k
    if not math.isfinite(vol):
        return a
    return float(min(cap, max(floor, vol)))


def smile_vol_arr(strikes, forward: float, coef: Tuple[float, float, float],
                  floor: float = 0.01, cap: float = 5.0) -> np.ndarray:
    """Vectorized ``smile_vol``."""
    a, b, c = coef
    strikes = np.asarray(strikes, dtype=np.float64)
    out = np.full(strikes.shape, a, dtype=np.float64)
    ok = np.isfinite(strikes) & (strikes > 0) & (forward > 0)
    k = np.zeros_like(strikes)
    k[ok] = np.log(strikes[ok] / forward)
    vol = a + b * k + c * k * k
    vol = np.clip(vol, floor, cap)
    out[ok] = vol[ok]
    return out


def probability_cone(
    p0: float,
    sigma: float,
    horizon_days: int = 30,
    trading_days_per_year: int = 252,
    include_t0: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lognormal volatility probability cone (Phase II).

    For trading-day horizon ``t``::

        Upper = P0 * exp(+σ √(t / N))
        Lower = P0 * exp(-σ √(t / N))

    where ``N`` defaults to 252 trading days/year and ``σ`` is annualized
    volatility (same units as EWMA / GARCH forecasts in this package).

    Parameters
    ----------
    p0 : float
        Spot / last price (> 0).
    sigma : float
        Annualized vol forecast (≥ 0). Zero vol ⇒ flat cone at ``p0``.
    horizon_days : int
        Number of trading days to project (default 30).
    trading_days_per_year : int
        Day-count convention inside the square root (default 252).
    include_t0 : bool
        If True, prepend ``t=0`` so the cone starts at the last print.

    Returns
    -------
    days, upper, lower : np.ndarray
        Parallel arrays of length ``horizon_days`` (+1 if ``include_t0``).
    """
    horizon_days = int(horizon_days)
    if horizon_days < 0:
        raise ValueError("horizon_days must be >= 0")
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be > 0")
    if not math.isfinite(p0) or p0 <= 0:
        raise ValueError("p0 must be a finite positive price")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be a finite non-negative vol")

    start = 0 if include_t0 else 1
    days = np.arange(start, horizon_days + 1, dtype=np.float64)
    if days.size == 0:
        return days, days.copy(), days.copy()

    # √(t/N); t=0 → 0 so bounds collapse to p0
    scale = np.sqrt(days / float(trading_days_per_year)) * float(sigma)
    upper = p0 * np.exp(+scale)
    lower = p0 * np.exp(-scale)
    return days, upper, lower


def blend_forecast_vol(
    ewma: float,
    garch: float,
    use_garch_blend: bool,
    garch_weight: float = 0.5,
) -> float:
    """Historical-vol input for the cone / FV path.

    Mirrors ``fetch_options_batch``: when ``use_garch_blend`` and GARCH > 0,
    return a convex blend; otherwise return EWMA (or 0 if invalid).
    """
    ewma = float(ewma) if ewma is not None and math.isfinite(ewma) else 0.0
    garch = float(garch) if garch is not None and math.isfinite(garch) else 0.0
    if ewma < 0:
        ewma = 0.0
    if garch < 0:
        garch = 0.0
    if use_garch_blend and garch > 0.0:
        w = min(1.0, max(0.0, float(garch_weight)))
        return (1.0 - w) * ewma + w * garch
    return ewma
