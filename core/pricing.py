"""Options pricing math core (Black-Scholes + Bjerksund-Stensland 2002).

References
----------
- Black, F. & Scholes, M. (1973); Merton, R. (1973) — European pricing / Greeks.
- Bjerksund, P. & Stensland, G. (2002). "Closed Form Valuation of American
  Options." Discussion paper 2002/09, NHH — two-boundary American approx.
- Bjerksund, P. & Stensland, G. (1993b); McDonald, R. & Schroder, M. (1998) —
  American put-call transformation.
- J.P. Morgan / RiskMetrics (1996) — EWMA variance with λ=0.94.
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
