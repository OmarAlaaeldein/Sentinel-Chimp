"""Options pricing math core (Black-Scholes + Bjerksund-Stensland 2002).

References
----------
- Black, F. & Scholes, M. (1973); Merton, R. (1973) — European pricing / Greeks.
- Bjerksund, P. & Stensland, G. (2002). "Closed Form Valuation of American
  Options." Discussion paper 2002/09, NHH — two-boundary American approx.
- Bjerksund, P. & Stensland, G. (1993b); McDonald, R. & Schroder, M. (1998) —
  American put-call transformation.
- J.P. Morgan / RiskMetrics (1996) — EWMA variance with λ=0.94.
- Abramowitz & Stegun (1964) §7.1.26 — vectorized erf for batch CDFs.

Batch API: ``bjerksund_stensland_batch``, ``american_greeks`` /
``american_greeks_batch`` (finite-difference on BS2002). GARCH / smile live in
``core.vol_models``.
"""
from __future__ import annotations

import math

import numpy as np

# ---- module-level constants (avoid recompute in hot paths) ----
_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_TWO_PI = 2.0 * math.pi

# 10-point Gauss-Legendre nodes/weights on [-1, 1] for bivariate normal CDF.
# Tuples (not numpy) — 10-point quadrature is faster as a pure Python loop.
_GL_NODES = (
    -0.9739065285171717, -0.8650633666889845,
    -0.6794095682990244, -0.4333953941292472,
    -0.1488743389816312,  0.1488743389816312,
     0.4333953941292472,  0.6794095682990244,
     0.8650633666889845,  0.9739065285171717,
)
_GL_WEIGHTS = (
    0.0666713443086881, 0.1494513491505806,
    0.2190863625159820, 0.2692667193099963,
    0.2955242247147529, 0.2955242247147529,
    0.2692667193099963, 0.2190863625159820,
    0.1494513491505806, 0.0666713443086881,
)


class VegaChimpCore:
    @staticmethod
    def N(x):
        return 0.5 * (1.0 + math.erf(x / _SQRT2))

    @staticmethod
    def n(x):
        """Standard normal PDF."""
        return math.exp(-0.5 * x * x) * _INV_SQRT_2PI

    @staticmethod
    def bs_price(S, K, r, q, sig, T, kind):
        """Standard Black-Scholes-Merton (European) with continuous yield q."""
        if kind not in {"call", "put"}:
            raise ValueError("kind must be 'call' or 'put'")
        if S <= 0 or K <= 0:
            raise ValueError("S and K must be positive")
        if T <= 1e-8:
            return max(0.0, S - K) if kind == "call" else max(0.0, K - S)
        if sig <= 1e-8:
            # With deterministic risk-neutral growth, discount the terminal payoff.
            discounted_spot = S * math.exp(-q * T)
            discounted_strike = K * math.exp(-r * T)
            if kind == "call":
                return max(0.0, discounted_spot - discounted_strike)
            return max(0.0, discounted_strike - discounted_spot)

        d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
        d2 = d1 - sig * math.sqrt(T)
        disc = math.exp(-r * T)
        disc_q = math.exp(-q * T)

        if kind == "call":
            return S * disc_q * VegaChimpCore.N(d1) - K * disc * VegaChimpCore.N(d2)
        return K * disc * VegaChimpCore.N(-d2) - S * disc_q * VegaChimpCore.N(-d1)

    @staticmethod
    def bs_greeks(S, K, r, q, sig, T, kind):
        """Compute Black-Scholes Greeks: delta, gamma, theta, vega, rho."""
        if sig <= 1e-4 or T <= 1e-4:
            if kind == "call":
                delta = 1.0 if S > K else 0.0
            else:
                delta = -1.0 if K > S else 0.0
            return {'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}

        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * sqrtT)
        d2 = d1 - sig * sqrtT
        disc = math.exp(-r * T)
        disc_q = math.exp(-q * T)
        nd1 = VegaChimpCore.n(d1)

        # Gamma and Vega are the same for calls and puts
        gamma = disc_q * nd1 / (S * sig * sqrtT)
        vega = S * disc_q * nd1 * sqrtT / 100  # Per 1% vol move

        if kind == "call":
            delta = disc_q * VegaChimpCore.N(d1)
            theta = (-(S * nd1 * sig * disc_q) / (2 * sqrtT)
                     - r * K * disc * VegaChimpCore.N(d2)
                     + q * S * disc_q * VegaChimpCore.N(d1)) / 365  # Per day
            rho = K * T * disc * VegaChimpCore.N(d2) / 100  # Per 1% rate move
        else:
            delta = disc_q * (VegaChimpCore.N(d1) - 1)
            theta = (-(S * nd1 * sig * disc_q) / (2 * sqrtT)
                     + r * K * disc * VegaChimpCore.N(-d2)
                     - q * S * disc_q * VegaChimpCore.N(-d1)) / 365
            rho = -K * T * disc * VegaChimpCore.N(-d2) / 100

        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

    @staticmethod
    def implied_vol(market_price, S, K, r, q, T, kind, tol=1e-6, max_iter=100):
        """Solve European implied volatility with a bracketed bisection method.

        Returns ``nan`` when the supplied price violates Black-Scholes
        no-arbitrage bounds. Prices at the zero-volatility bound return 0.
        """
        if kind not in {"call", "put"}:
            raise ValueError("kind must be 'call' or 'put'")
        if S <= 0 or K <= 0:
            raise ValueError("S and K must be positive")
        if market_price < 0 or not math.isfinite(market_price):
            return math.nan
        if T <= 1e-8:
            return 0.0

        discounted_spot = S * math.exp(-q * T)
        discounted_strike = K * math.exp(-r * T)
        if kind == "call":
            lower = max(0.0, discounted_spot - discounted_strike)
            upper = discounted_spot
        else:
            lower = max(0.0, discounted_strike - discounted_spot)
            upper = discounted_strike

        if market_price < lower - tol or market_price >= upper - tol:
            return math.nan
        if market_price <= lower + tol:
            return 0.0

        low_sig = 0.0
        high_sig = 1.0
        while (VegaChimpCore.bs_price(S, K, r, q, high_sig, T, kind)
               < market_price and high_sig < 16.0):
            high_sig *= 2.0

        if VegaChimpCore.bs_price(S, K, r, q, high_sig, T, kind) < market_price:
            return math.nan

        mid_sig = 0.5 * (low_sig + high_sig)
        for _ in range(max_iter):
            mid_sig = 0.5 * (low_sig + high_sig)
            price = VegaChimpCore.bs_price(S, K, r, q, mid_sig, T, kind)
            if abs(price - market_price) <= tol:
                return mid_sig
            if price < market_price:
                low_sig = mid_sig
            else:
                high_sig = mid_sig
        return mid_sig

    @staticmethod
    def _M(a, b, rho):
        """Bivariate standard normal CDF: P(X <= a, Y <= b) where corr(X, Y) = rho.

        Uses 10-point Gauss-Legendre quadrature on the Drezner/Haug integral
        representation (see Haug 1997; Bjerksund-Stensland 2002 Prop. 1).
        Accurate to ~1e-7 for financial option pricing purposes.
        """
        N = VegaChimpCore.N
        if rho == 0.0:
            return N(a) * N(b)
        if a <= -1e15 or b <= -1e15:
            return 0.0
        if a >= 1e15:
            return N(b)
        if b >= 1e15:
            return N(a)

        # Identity: M(a,b;rho) = N(a)*N(b) + integral_0^rho f(s) ds
        # Map [0, rho] -> [-1, 1] via s = rho*(xi+1)/2
        half_rho = rho * 0.5
        bvn = N(a) * N(b)
        aa, bb = a * a, b * b
        ab2 = 2.0 * a * b
        for xi, wi in zip(_GL_NODES, _GL_WEIGHTS):
            s = half_rho * (xi + 1.0)
            denom = 1.0 - s * s
            if denom <= 0.0:
                continue
            exponent = -(aa - s * ab2 + bb) / (2.0 * denom)
            bvn += half_rho * wi * math.exp(exponent) / (_TWO_PI * math.sqrt(denom))
        return max(0.0, min(1.0, bvn))

    @staticmethod
    def ewma_vol_forecast(log_returns, days=252):
        """RiskMetrics EWMA volatility forecast.

        sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r_{t-1}^2
        with lambda=0.94 (J.P. Morgan / RiskMetrics 1996 daily decay).
        Pure EWMA (omega=0, alpha+beta=1). Initialized at sample variance.
        """
        try:
            returns = np.asarray(log_returns, dtype=np.float64).reshape(-1)
            returns = returns[np.isfinite(returns)]
            n = returns.size
            if n < 30:
                return 0.0
            lam = 0.94
            r2 = returns * returns
            variance0 = float(np.var(returns))
            # Closed form of the recursion (exact match to sequential update):
            # v_n = lam^n * v0 + (1-lam) * sum_{i=0}^{n-1} lam^{n-1-i} * r2[i]
            # powers[i] = lam^{n-1-i}  => powers = lam^{n-1}, lam^{n-2}, ..., 1
            exponents = np.arange(n - 1, -1, -1, dtype=np.float64)
            powers = np.power(lam, exponents)
            variance = (lam ** n) * variance0 + (1.0 - lam) * float(np.dot(powers, r2))
            if variance <= 0.0 or not math.isfinite(variance):
                return 0.0
            return float(math.sqrt(variance * days))
        except (TypeError, ValueError, FloatingPointError):
            return 0.0

    @staticmethod
    def american_put_call_parity_bounds(S, K, r, q, T):
        """Return no-arbitrage bounds for American call minus American put.

        American options satisfy an inequality, rather than the European parity
        equality, because calls and puts can have different early-exercise premia.
        """
        bound_a = S * math.exp(-q * T) - K
        bound_b = S - K * math.exp(-r * T)
        return min(bound_a, bound_b), max(bound_a, bound_b)

    @staticmethod
    def _safe_exp(val):
        if val > 700:
            return float('inf')
        if val < -700:
            return 0.0
        return math.exp(val)

    @staticmethod
    def _phi(S, T, gamma, H, X, r, b, sigma):
        """Single-barrier expectation φ from Bjerksund-Stensland (1993/2002)."""
        sig2 = sigma * sigma
        lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sig2) * T
        d_den = sigma * math.sqrt(T)
        d = -(math.log(S / H) + (b + (gamma - 0.5) * sig2) * T) / d_den
        kappa = 2.0 * b / sig2 + (2.0 * gamma - 1.0)
        ln_s = math.log(S)
        ln_x = math.log(X)
        power1 = lam + gamma * ln_s
        val_1 = VegaChimpCore._safe_exp(power1) * VegaChimpCore.N(d)
        d2 = d - 2.0 * math.log(X / S) / d_den
        power2 = power1 + kappa * (ln_x - ln_s)
        val_2 = VegaChimpCore._safe_exp(power2) * VegaChimpCore.N(d2)
        return val_1 - val_2

    @staticmethod
    def _psi(S, T, gamma, H, X, x, t, r, b, sigma):
        """Two-boundary expectation Ψ from Bjerksund-Stensland (2002) Prop. 1.

        Notation matches the paper: X = first-period barrier, x = second-period
        barrier, t = time split, T = maturity, H = payoff barrier.
        """
        if T <= 1e-10 or t <= 1e-10:
            return VegaChimpCore._phi(S, T, gamma, H, X, r, b, sigma)

        sig2 = sigma * sigma
        sqt = sigma * math.sqrt(T)
        sqt1 = sigma * math.sqrt(t)
        drift_t = (b + (gamma - 0.5) * sig2) * t
        drift_T = (b + (gamma - 0.5) * sig2) * T

        # d1..d4 and D1..D4 as in Prop. 1 (leading minus baked in)
        e1 = -(math.log(S / x) + drift_t) / sqt1
        e2 = -(math.log(X * X / (S * x)) + drift_t) / sqt1
        e3 = -(math.log(S / x) - drift_t) / sqt1
        e4 = -(math.log(X * X / (S * x)) - drift_t) / sqt1

        f1 = -(math.log(S / H) + drift_T) / sqt
        f2 = -(math.log(X * X / (S * H)) + drift_T) / sqt
        f3 = -(math.log(x * x / (S * H)) + drift_T) / sqt
        f4 = -(math.log(S * x * x / (H * X * X)) + drift_T) / sqt

        rho_val = math.sqrt(t / T)
        lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sig2
        kappa = 2.0 * b / sig2 + (2.0 * gamma - 1.0)

        ln_s = math.log(S)
        ln_X = math.log(X)
        ln_x = math.log(x)
        power = lam * T + gamma * ln_s

        M = VegaChimpCore._M
        se = VegaChimpCore._safe_exp
        term1 = se(power) * M(e1, f1, rho_val)
        term2 = se(power + kappa * (ln_X - ln_s)) * M(e2, f2, rho_val)
        term3 = se(power + kappa * (ln_x - ln_s)) * M(e3, f3, -rho_val)
        term4 = se(power + kappa * (ln_x - ln_X)) * M(e4, f4, -rho_val)
        return term1 - term2 - term3 + term4

    @staticmethod
    def bjerksund_stensland(S, K, T, r, q, sigma, option_type='call'):
        """
        Bjerksund-Stensland 2002 American option approximation.

        Two-boundary version using the paper's golden-ratio time split
        t = ½(√5 − 1)T. Uses log-space algebra to prevent overflow.
        Puts are valued via the Bjerksund-Stensland / McDonald-Schroder
        transformation: P(S,K,T,r,b,σ) = C(K,S,T,r−b,−b,σ).
        """
        if option_type not in {'call', 'put'}:
            raise ValueError("option_type must be 'call' or 'put'")
        if S <= 0 or K <= 0:
            return 0.0
        if T <= 0:
            return max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)
        if sigma <= 1e-8:
            european = VegaChimpCore.bs_price(S, K, r, q, sigma, T, option_type)
            intrinsic = max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)
            return max(european, intrinsic)

        if option_type == 'put':
            # P(S,K,r,q) with b=r-q  <=>  C(K,S,r'=q, q'=r)
            put_via_transform = VegaChimpCore.bjerksund_stensland(K, S, T, q, r, sigma, 'call')
            # American put should never price below the corresponding European put.
            put_euro_floor = VegaChimpCore.bs_price(S, K, r, q, sigma, T, 'put')
            return max(put_via_transform, put_euro_floor)

        b = r - q
        if b >= r:
            # No early exercise for calls when cost-of-carry >= r (q <= 0).
            return VegaChimpCore.bs_price(S, K, r, q, sigma, T, 'call')

        try:
            sig2 = sigma * sigma
            phi = VegaChimpCore._phi
            psi = VegaChimpCore._psi
            se = VegaChimpCore._safe_exp

            # --- 2002 Two-Boundary Method ---
            beta_val = (0.5 - b / sig2) + math.sqrt((b / sig2 - 0.5) ** 2 + 2.0 * r / sig2)
            if abs(beta_val - 1.0) < 1e-5:
                return max(S - K, 0.0)

            boundary_inf = K * beta_val / (beta_val - 1.0)
            boundary_zero = max(K, (r / (r - b)) * K)
            t1 = 0.5 * (math.sqrt(5.0) - 1.0) * T  # golden-ratio split (paper Eq. 16)

            def exercise_boundary(remaining_time):
                # Paper Eqs. (10)-(13): X_τ = B0 + (B∞ − B0)(1 − exp{h(τ)})
                h_val = (-(b * remaining_time + 2.0 * sigma * math.sqrt(remaining_time))
                         * K * K / ((boundary_inf - boundary_zero) * boundary_zero))
                return (boundary_zero + (boundary_inf - boundary_zero)
                        * (1.0 - se(h_val)))

            # Paper: X = X_T (first period [0,t]), x = X_{T-t} (second period).
            # Code names: I2 ≡ X, I1 ≡ x.
            I2 = exercise_boundary(T)
            I1 = exercise_boundary(T - t1)

            if S >= I2:
                intrinsic = max(S - K, 0.0)
                european = VegaChimpCore.bs_price(S, K, r, q, sigma, T, 'call')
                return max(intrinsic, european)

            alpha2 = (I2 - K) * se(-beta_val * math.log(I2))
            alpha1 = (I1 - K) * se(-beta_val * math.log(I1))

            # Proposition 1 (paper) with X=I2, x=I1, t=t1
            term1 = alpha2 * se(beta_val * math.log(S))
            term2 = alpha2 * phi(S, t1, beta_val, I2, I2, r, b, sigma)
            term3 = phi(S, t1, 1.0, I2, I2, r, b, sigma)
            term4 = phi(S, t1, 1.0, I1, I2, r, b, sigma)
            term5 = K * phi(S, t1, 0.0, I2, I2, r, b, sigma)
            term6 = K * phi(S, t1, 0.0, I1, I2, r, b, sigma)
            term7 = alpha1 * phi(S, t1, beta_val, I1, I2, r, b, sigma)
            term8 = alpha1 * psi(S, T, beta_val, I1, I2, I1, t1, r, b, sigma)
            term9 = psi(S, T, 1.0, I1, I2, I1, t1, r, b, sigma)
            term10 = psi(S, T, 1.0, K, I2, I1, t1, r, b, sigma)
            term11 = K * psi(S, T, 0.0, I1, I2, I1, t1, r, b, sigma)
            term12 = K * psi(S, T, 0.0, K, I2, I1, t1, r, b, sigma)

            price = (term1 - term2 + term3 - term4 - term5 + term6
                     + term7 - term8 + term9 - term10 - term11 + term12)

            # Feasible strategy => lower bound; clamp numerical noise to
            # European / intrinsic floors (American dominance).
            intrinsic = max(0.0, S - K)
            european = VegaChimpCore.bs_price(S, K, r, q, sigma, T, 'call')
            return max(price, intrinsic, european)

        except (OverflowError, ValueError, ZeroDivisionError):
            print("[Warning] Bjerksund-Stensland 2002 failed, falling back to Black-Scholes.")
            return VegaChimpCore.bs_price(S, K, r, q, sigma, T, 'call')


# ---------------------------------------------------------------------------
# Vectorized / batch helpers (numpy). Used by chain scans.
# erf: Abramowitz–Stegun 7.1.26 (max |err| ~1.4e-7) — no scipy required.
# ---------------------------------------------------------------------------

def _erf_arr(x):
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    # Horner form of the rational approximation
    poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    return sign * (1.0 - poly * np.exp(-ax * ax))


def _N_arr(x):
    return 0.5 * (1.0 + _erf_arr(np.asarray(x, dtype=np.float64) / _SQRT2))


def _safe_exp_arr(val):
    return np.exp(np.clip(np.asarray(val, dtype=np.float64), -700.0, 700.0))


def _M_arr(a, b, rho):
    """Vectorized bivariate standard normal CDF (same quadrature as ``_M``)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    a, b, rho = np.broadcast_arrays(a, b, rho)
    shape = a.shape
    a = a.ravel()
    b = b.ravel()
    rho = rho.ravel()
    na = _N_arr(a)
    nb = _N_arr(b)
    half_rho = rho * 0.5
    nodes = np.asarray(_GL_NODES, dtype=np.float64)
    weights = np.asarray(_GL_WEIGHTS, dtype=np.float64)
    s = half_rho[:, None] * (nodes[None, :] + 1.0)
    denom = 1.0 - s * s
    safe = denom > 1e-15
    aa = a * a
    bb = b * b
    ab2 = 2.0 * a * b
    exponent = np.full_like(s, -np.inf)
    numer = -(aa[:, None] - s * ab2[:, None] + bb[:, None])
    exponent[safe] = (numer / (2.0 * denom))[safe]
    contrib = np.zeros_like(s)
    denom_safe = np.where(safe, denom, 1.0)
    raw = (half_rho[:, None] * weights[None, :] * np.exp(exponent)
           / (_TWO_PI * np.sqrt(denom_safe)))
    contrib[safe] = raw[safe]
    out = np.clip(na * nb + contrib.sum(axis=1), 0.0, 1.0)
    out = np.where(rho == 0.0, na * nb, out)
    mask_neg = (a <= -1e15) | (b <= -1e15)
    out = np.where(mask_neg, 0.0, out)
    out = np.where((~mask_neg) & (a >= 1e15), nb, out)
    out = np.where((~mask_neg) & (b >= 1e15) & (a < 1e15), na, out)
    return out.reshape(shape)


def _phi_arr(S, T, gamma, H, X, r, b, sigma):
    S, T, gamma, H, X, r, b, sigma = np.broadcast_arrays(
        *[np.asarray(v, dtype=np.float64) for v in (S, T, gamma, H, X, r, b, sigma)]
    )
    sig2 = sigma * sigma
    lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sig2) * T
    d_den = sigma * np.sqrt(T)
    d = -(np.log(S / H) + (b + (gamma - 0.5) * sig2) * T) / d_den
    kappa = 2.0 * b / sig2 + (2.0 * gamma - 1.0)
    ln_s = np.log(S)
    ln_x = np.log(X)
    power1 = lam + gamma * ln_s
    val_1 = _safe_exp_arr(power1) * _N_arr(d)
    d2 = d - 2.0 * np.log(X / S) / d_den
    power2 = power1 + kappa * (ln_x - ln_s)
    val_2 = _safe_exp_arr(power2) * _N_arr(d2)
    return val_1 - val_2


def _psi_arr(S, T, gamma, H, X, x, t, r, b, sigma):
    S, T, gamma, H, X, x, t, r, b, sigma = np.broadcast_arrays(
        *[np.asarray(v, dtype=np.float64)
          for v in (S, T, gamma, H, X, x, t, r, b, sigma)]
    )
    tiny = (T <= 1e-10) | (t <= 1e-10)
    out = np.empty(S.shape, dtype=np.float64)
    if np.any(tiny):
        out[tiny] = _phi_arr(
            S[tiny], T[tiny], gamma[tiny], H[tiny], X[tiny],
            r[tiny], b[tiny], sigma[tiny],
        )
    m = ~tiny
    if not np.any(m):
        return out
    Sm, Tm, gm, Hm, Xm, xm, tm, rm, bm, sm = (
        arr[m] for arr in (S, T, gamma, H, X, x, t, r, b, sigma)
    )
    sig2 = sm * sm
    sqt = sm * np.sqrt(Tm)
    sqt1 = sm * np.sqrt(tm)
    drift_t = (bm + (gm - 0.5) * sig2) * tm
    drift_T = (bm + (gm - 0.5) * sig2) * Tm
    e1 = -(np.log(Sm / xm) + drift_t) / sqt1
    e2 = -(np.log(Xm * Xm / (Sm * xm)) + drift_t) / sqt1
    e3 = -(np.log(Sm / xm) - drift_t) / sqt1
    e4 = -(np.log(Xm * Xm / (Sm * xm)) - drift_t) / sqt1
    f1 = -(np.log(Sm / Hm) + drift_T) / sqt
    f2 = -(np.log(Xm * Xm / (Sm * Hm)) + drift_T) / sqt
    f3 = -(np.log(xm * xm / (Sm * Hm)) + drift_T) / sqt
    f4 = -(np.log(Sm * xm * xm / (Hm * Xm * Xm)) + drift_T) / sqt
    rho_val = np.sqrt(tm / Tm)
    lam = -rm + gm * bm + 0.5 * gm * (gm - 1.0) * sig2
    kappa = 2.0 * bm / sig2 + (2.0 * gm - 1.0)
    ln_s = np.log(Sm)
    ln_X = np.log(Xm)
    ln_x = np.log(xm)
    power = lam * Tm + gm * ln_s
    term1 = _safe_exp_arr(power) * _M_arr(e1, f1, rho_val)
    term2 = _safe_exp_arr(power + kappa * (ln_X - ln_s)) * _M_arr(e2, f2, rho_val)
    term3 = _safe_exp_arr(power + kappa * (ln_x - ln_s)) * _M_arr(e3, f3, -rho_val)
    term4 = _safe_exp_arr(power + kappa * (ln_x - ln_X)) * _M_arr(e4, f4, -rho_val)
    out[m] = term1 - term2 - term3 + term4
    return out


# Attach batch API onto VegaChimpCore
@staticmethod
def _bs_price_arr(S, K, r, q, sig, T, is_call):
    """Vectorized European BSM. ``is_call`` is a boolean array."""
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    sig = np.asarray(sig, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    is_call = np.asarray(is_call, dtype=bool)
    S, K, r, q, sig, T, is_call = np.broadcast_arrays(S, K, r, q, sig, T, is_call)
    out = np.empty(S.shape, dtype=np.float64)

    t0 = T <= 1e-8
    if np.any(t0):
        out[t0 & is_call] = np.maximum(0.0, S[t0 & is_call] - K[t0 & is_call])
        out[t0 & ~is_call] = np.maximum(0.0, K[t0 & ~is_call] - S[t0 & ~is_call])

    s0 = (~t0) & (sig <= 1e-8)
    if np.any(s0):
        ds = S[s0] * np.exp(-q[s0] * T[s0])
        dk = K[s0] * np.exp(-r[s0] * T[s0])
        ic = is_call[s0]
        tmp = np.empty(ds.shape)
        tmp[ic] = np.maximum(0.0, ds[ic] - dk[ic])
        tmp[~ic] = np.maximum(0.0, dk[~ic] - ds[~ic])
        out[s0] = tmp

    main = (~t0) & (~s0)
    if np.any(main):
        Sm, Km, rm, qm, sm, Tm, cm = (
            arr[main] for arr in (S, K, r, q, sig, T, is_call)
        )
        d1 = (np.log(Sm / Km) + (rm - qm + 0.5 * sm * sm) * Tm) / (sm * np.sqrt(Tm))
        d2 = d1 - sm * np.sqrt(Tm)
        disc = np.exp(-rm * Tm)
        disc_q = np.exp(-qm * Tm)
        call = Sm * disc_q * _N_arr(d1) - Km * disc * _N_arr(d2)
        put = Km * disc * _N_arr(-d2) - Sm * disc_q * _N_arr(-d1)
        out[main] = np.where(cm, call, put)
    return out


@staticmethod
def _american_call_arr(S, K, T, r, q, sigma):
    """Vectorized BS2002 American **call** (put via transform in batch API)."""
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    S, K, T, r, q, sigma = np.broadcast_arrays(S, K, T, r, q, sigma)
    out = np.zeros(S.shape, dtype=np.float64)

    bad = (S <= 0) | (K <= 0)
    t0 = (~bad) & (T <= 0)
    out[t0] = np.maximum(S[t0] - K[t0], 0.0)

    s0 = (~bad) & (~t0) & (sigma <= 1e-8)
    if np.any(s0):
        euro = _bs_price_arr(S[s0], K[s0], r[s0], q[s0], sigma[s0], T[s0],
                             np.ones(np.count_nonzero(s0), dtype=bool))
        out[s0] = np.maximum(euro, np.maximum(S[s0] - K[s0], 0.0))

    b = r - q
    no_ee = (~bad) & (~t0) & (~s0) & (b >= r)
    if np.any(no_ee):
        out[no_ee] = _bs_price_arr(
            S[no_ee], K[no_ee], r[no_ee], q[no_ee], sigma[no_ee], T[no_ee],
            np.ones(np.count_nonzero(no_ee), dtype=bool),
        )

    main = (~bad) & (~t0) & (~s0) & (b < r)
    if not np.any(main):
        return out

    Sm = S[main]; Km = K[main]; Tm = T[main]
    rm = r[main]; qm = q[main]; sm = sigma[main]
    bm = rm - qm
    sig2 = sm * sm
    # Guard r~0 / sig~0 already excluded; still protect division
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = (0.5 - bm / sig2) + np.sqrt((bm / sig2 - 0.5) ** 2 + 2.0 * rm / sig2)
    beta_bad = (~np.isfinite(beta)) | (np.abs(beta - 1.0) < 1e-5)
    sub = np.empty(Sm.shape, dtype=np.float64)
    sub[beta_bad] = np.maximum(Sm[beta_bad] - Km[beta_bad], 0.0)

    m2 = ~beta_bad
    if np.any(m2):
        S2, K2, T2, r2, q2, s2, b2, beta2 = (
            arr[m2] for arr in (Sm, Km, Tm, rm, qm, sm, bm, beta)
        )
        boundary_inf = K2 * beta2 / (beta2 - 1.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            boundary_zero = np.maximum(K2, (r2 / (r2 - b2)) * K2)
        # If carry edge makes boundary_zero non-finite, fall back to European
        bz_bad = ~np.isfinite(boundary_zero) | ~np.isfinite(boundary_inf)
        price2 = np.empty(S2.shape, dtype=np.float64)
        if np.any(bz_bad):
            price2[bz_bad] = _bs_price_arr(
                S2[bz_bad], K2[bz_bad], r2[bz_bad], q2[bz_bad], s2[bz_bad], T2[bz_bad],
                np.ones(np.count_nonzero(bz_bad), dtype=bool),
            )
        m_ok = ~bz_bad
        if np.any(m_ok):
            S3, K3, T3, r3, q3, s3, b3, beta3 = (
                arr[m_ok] for arr in (S2, K2, T2, r2, q2, s2, b2, beta2)
            )
            binf = boundary_inf[m_ok]
            bzero = boundary_zero[m_ok]
            t1 = 0.5 * (math.sqrt(5.0) - 1.0) * T3

            def _ex_bound(tau):
                h_val = (-(b3 * tau + 2.0 * s3 * np.sqrt(tau))
                         * K3 * K3 / ((binf - bzero) * bzero))
                return bzero + (binf - bzero) * (1.0 - _safe_exp_arr(h_val))

            I2 = _ex_bound(T3)
            I1 = _ex_bound(T3 - t1)
            hit = S3 >= I2
            price3 = np.empty(S3.shape, dtype=np.float64)
            if np.any(hit):
                intrinsic = np.maximum(S3[hit] - K3[hit], 0.0)
                euro = _bs_price_arr(
                    S3[hit], K3[hit], r3[hit], q3[hit], s3[hit], T3[hit],
                    np.ones(np.count_nonzero(hit), dtype=bool),
                )
                price3[hit] = np.maximum(intrinsic, euro)
            m4 = ~hit
            if np.any(m4):
                S4 = S3[m4]; K4 = K3[m4]; T4 = T3[m4]
                r4 = r3[m4]; q4 = q3[m4]; s4 = s3[m4]; b4 = b3[m4]
                beta4 = beta3[m4]; I2m = I2[m4]; I1m = I1[m4]; t1m = t1[m4]
                alpha2 = (I2m - K4) * _safe_exp_arr(-beta4 * np.log(I2m))
                alpha1 = (I1m - K4) * _safe_exp_arr(-beta4 * np.log(I1m))
                term1 = alpha2 * _safe_exp_arr(beta4 * np.log(S4))
                term2 = alpha2 * _phi_arr(S4, t1m, beta4, I2m, I2m, r4, b4, s4)
                term3 = _phi_arr(S4, t1m, 1.0, I2m, I2m, r4, b4, s4)
                term4 = _phi_arr(S4, t1m, 1.0, I1m, I2m, r4, b4, s4)
                term5 = K4 * _phi_arr(S4, t1m, 0.0, I2m, I2m, r4, b4, s4)
                term6 = K4 * _phi_arr(S4, t1m, 0.0, I1m, I2m, r4, b4, s4)
                term7 = alpha1 * _phi_arr(S4, t1m, beta4, I1m, I2m, r4, b4, s4)
                term8 = alpha1 * _psi_arr(S4, T4, beta4, I1m, I2m, I1m, t1m, r4, b4, s4)
                term9 = _psi_arr(S4, T4, 1.0, I1m, I2m, I1m, t1m, r4, b4, s4)
                term10 = _psi_arr(S4, T4, 1.0, K4, I2m, I1m, t1m, r4, b4, s4)
                term11 = K4 * _psi_arr(S4, T4, 0.0, I1m, I2m, I1m, t1m, r4, b4, s4)
                term12 = K4 * _psi_arr(S4, T4, 0.0, K4, I2m, I1m, t1m, r4, b4, s4)
                raw = (term1 - term2 + term3 - term4 - term5 + term6
                       + term7 - term8 + term9 - term10 - term11 + term12)
                intrinsic = np.maximum(0.0, S4 - K4)
                euro = _bs_price_arr(
                    S4, K4, r4, q4, s4, T4, np.ones(S4.shape, dtype=bool),
                )
                price3[m4] = np.maximum(np.maximum(raw, intrinsic), euro)
            price2[m_ok] = price3
        sub[m2] = price2

    out[main] = sub
    return out


# Bind as staticmethods on the class (defined at module level above for clarity
# of the helpers; re-attach below).
VegaChimpCore._bs_price_arr = staticmethod(_bs_price_arr.__func__
                                           if hasattr(_bs_price_arr, '__func__')
                                           else _bs_price_arr)
VegaChimpCore._american_call_arr = staticmethod(
    _american_call_arr.__func__ if hasattr(_american_call_arr, '__func__')
    else _american_call_arr
)


@staticmethod
def bjerksund_stensland_batch(S, K, T, r, q, sigma, option_type='call'):
    """Batch BS2002 American prices.

    Parameters
    ----------
    S, T, r, q : float or array
        Broadcastable with ``K`` / ``sigma``.
    K, sigma : array-like
        Strikes and vols (typically one per contract).
    option_type : {'call','put'} or sequence of those / bool is_call

    Returns
    -------
    np.ndarray of prices (broadcast shape).
    """
    K = np.asarray(K, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if isinstance(option_type, str):
        is_call = np.full(np.broadcast(K, sigma, S, T, r, q).shape,
                          option_type == 'call', dtype=bool)
    else:
        ot = np.asarray(option_type)
        if ot.dtype == bool:
            is_call = ot
        else:
            # bytes/str array
            is_call = np.array([
                (x == 'call' or x == b'call' or x == 'CALL' or x == b'CALL')
                for x in np.asarray(ot).ravel()
            ], dtype=bool).reshape(ot.shape)

    S, K, T, r, q, sigma, is_call = np.broadcast_arrays(
        S, K, T, r, q, sigma, is_call
    )
    out = np.empty(S.shape, dtype=np.float64)

    # Calls
    if np.any(is_call):
        out[is_call] = VegaChimpCore._american_call_arr(
            S[is_call], K[is_call], T[is_call], r[is_call], q[is_call], sigma[is_call]
        )

    # Puts via BS / McDonald–Schroder transform + European floor
    if np.any(~is_call):
        m = ~is_call
        put_via = VegaChimpCore._american_call_arr(
            K[m], S[m], T[m], q[m], r[m], sigma[m]
        )
        put_euro = VegaChimpCore._bs_price_arr(
            S[m], K[m], r[m], q[m], sigma[m], T[m],
            np.zeros(np.count_nonzero(m), dtype=bool),
        )
        out[m] = np.maximum(put_via, put_euro)
    return out


@staticmethod
def american_greeks(S, K, r, q, sig, T, kind, dS=None, dSig=0.01, dT=1.0 / 365.0):
    """Finite-difference Greeks on BS2002 American prices.

    Conventions match ``bs_greeks``: theta per calendar day, vega per 1% vol,
    rho per 1% rate. Delta/gamma via central differences in S.
    """
    if kind not in {'call', 'put'}:
        raise ValueError("kind must be 'call' or 'put'")
    if dS is None:
        dS = max(1e-4 * S, 1e-4)
    price = VegaChimpCore.bjerksund_stensland
    if sig <= 1e-4 or T <= 1e-4:
        if kind == 'call':
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if K > S else 0.0
        return {'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}

    p_up = price(S + dS, K, T, r, q, sig, kind)
    p_dn = price(S - dS, K, T, r, q, sig, kind)
    p0 = price(S, K, T, r, q, sig, kind)
    delta = (p_up - p_dn) / (2.0 * dS)
    gamma = (p_up - 2.0 * p0 + p_dn) / (dS * dS)

    p_vu = price(S, K, T, r, q, sig + dSig, kind)
    p_vd = price(S, K, T, r, q, max(sig - dSig, 1e-8), kind)
    vega = (p_vu - p_vd) / (2.0 * dSig) / 100.0

    T_dn = max(T - dT, 1e-8)
    # theta ≈ −∂V/∂T ; report per calendar day (dT = 1/365)
    p_t = price(S, K, T_dn, r, q, sig, kind)
    theta = (p_t - p0) / 1.0  # already one calendar-day bump

    # rho (per 1% rate): bump r by 1e-4 absolute (=1bp) then scale — match euro
    dr = 0.01
    p_ru = price(S, K, T, r + dr, q, sig, kind)
    p_rd = price(S, K, T, r - dr, q, sig, kind)
    rho = (p_ru - p_rd) / (2.0 * dr) / 100.0

    return {
        'delta': float(delta),
        'gamma': float(gamma),
        'theta': float(theta),
        'vega': float(vega),
        'rho': float(rho),
    }


@staticmethod
def american_greeks_batch(S, K, r, q, sigma, T, option_type, dS=None, dSig=0.01,
                          dT=1.0 / 365.0):
    """Vectorized FD Greeks for a chain slice (shared S,T,r,q typical).

    Returns dict of numpy arrays: delta, gamma, theta, vega, rho.
    """
    K = np.asarray(K, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    batch = VegaChimpCore.bjerksund_stensland_batch
    S_val = float(np.asarray(S).reshape(-1)[0]) if np.size(S) == 1 else None
    if dS is None:
        dS = max(1e-4 * (S_val if S_val is not None else float(np.mean(S))), 1e-4)

    p0 = batch(S, K, T, r, q, sigma, option_type)
    p_up = batch(np.asarray(S, dtype=np.float64) + dS, K, T, r, q, sigma, option_type)
    p_dn = batch(np.asarray(S, dtype=np.float64) - dS, K, T, r, q, sigma, option_type)
    delta = (p_up - p_dn) / (2.0 * dS)
    gamma = (p_up - 2.0 * p0 + p_dn) / (dS * dS)

    p_vu = batch(S, K, T, r, q, np.asarray(sigma, dtype=np.float64) + dSig, option_type)
    p_vd = batch(S, K, T, r, q, np.maximum(np.asarray(sigma, dtype=np.float64) - dSig, 1e-8),
                 option_type)
    vega = (p_vu - p_vd) / (2.0 * dSig) / 100.0

    T_arr = np.asarray(T, dtype=np.float64)
    p_t = batch(S, K, np.maximum(T_arr - dT, 1e-8), r, q, sigma, option_type)
    theta = p_t - p0

    dr = 0.01
    p_ru = batch(S, K, T, np.asarray(r, dtype=np.float64) + dr, q, sigma, option_type)
    p_rd = batch(S, K, T, np.asarray(r, dtype=np.float64) - dr, q, sigma, option_type)
    rho = (p_ru - p_rd) / (2.0 * dr) / 100.0

    # Edge: near expiry / tiny vol → match scalar american_greeks discrete delta
    sig_arr = np.asarray(sigma, dtype=np.float64)
    T_b = np.broadcast_to(T_arr, p0.shape)
    sig_b = np.broadcast_to(sig_arr, p0.shape)
    edge = (sig_b <= 1e-4) | (T_b <= 1e-4)
    if np.any(edge):
        S_b = np.broadcast_to(np.asarray(S, dtype=np.float64), p0.shape)
        K_b = np.broadcast_to(K, p0.shape)
        if isinstance(option_type, str):
            is_call = option_type == 'call'
            if is_call:
                delta = np.where(edge, np.where(S_b > K_b, 1.0, 0.0), delta)
            else:
                delta = np.where(edge, np.where(K_b > S_b, -1.0, 0.0), delta)
        else:
            ot = np.asarray(option_type)
            if ot.dtype == bool:
                is_call = ot
            else:
                is_call = np.array([
                    (x == 'call' or x == b'call' or x == 'CALL' or x == b'CALL')
                    for x in ot.ravel()
                ], dtype=bool).reshape(ot.shape)
            is_call_b = np.broadcast_to(is_call, p0.shape)
            d_edge = np.where(
                is_call_b,
                np.where(S_b > K_b, 1.0, 0.0),
                np.where(K_b > S_b, -1.0, 0.0),
            )
            delta = np.where(edge, d_edge, delta)
        gamma = np.where(edge, 0.0, gamma)
        theta = np.where(edge, 0.0, theta)
        vega = np.where(edge, 0.0, vega)
        rho = np.where(edge, 0.0, rho)

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho,
    }


VegaChimpCore.bjerksund_stensland_batch = staticmethod(
    bjerksund_stensland_batch.__func__
    if hasattr(bjerksund_stensland_batch, '__func__') else bjerksund_stensland_batch
)
VegaChimpCore.american_greeks = staticmethod(
    american_greeks.__func__ if hasattr(american_greeks, '__func__') else american_greeks
)
VegaChimpCore.american_greeks_batch = staticmethod(
    american_greeks_batch.__func__
    if hasattr(american_greeks_batch, '__func__') else american_greeks_batch
)
