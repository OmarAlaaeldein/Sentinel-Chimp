"""Options pricing math core (Black-Scholes + Bjerksund-Stensland)."""
import math

import numpy as np


class VegaChimpCore:
    @staticmethod
    def N(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def n(x):
        """Standard normal PDF."""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def bs_price(S, K, r, q, sig, T, kind):
        """Standard Black-Scholes (European)."""
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
        disc = math.exp(-r * T); disc_q = math.exp(-q * T)

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
        Uses 10-point Gauss-Legendre quadrature on the integral representation.
        Accurate to ~1e-7 for financial option pricing purposes.
        """
        N = VegaChimpCore.N
        if rho == 0.0:
            return N(a) * N(b)
        # Handle infinite bounds
        if a <= -1e15 or b <= -1e15:
            return 0.0
        if a >= 1e15:
            return N(b)
        if b >= 1e15:
            return N(a)

        # 10-point GL nodes and weights on [-1, 1]
        GL_nodes = [
            -0.9739065285171717, -0.8650633666889845,
            -0.6794095682990244, -0.4333953941292472,
            -0.1488743389816312,  0.1488743389816312,
             0.4333953941292472,  0.6794095682990244,
             0.8650633666889845,  0.9739065285171717,
        ]
        GL_weights = [
            0.0666713443086881, 0.1494513491505806,
            0.2190863625159820, 0.2692667193099963,
            0.2955242247147529, 0.2955242247147529,
            0.2692667193099963, 0.2190863625159820,
            0.1494513491505806, 0.0666713443086881,
        ]

        # Identity: M(a,b;rho) = N(a)*N(b) + integral_0^rho f(s) ds
        # where f(s) = exp(-(a^2 - 2s*a*b + b^2)/(2*(1-s^2))) / (2*pi*sqrt(1-s^2))
        # Map [0, rho] -> [-1, 1] via s = rho*(xi+1)/2
        half_rho = rho / 2.0
        bvn = N(a) * N(b)
        two_pi = 2.0 * math.pi
        for xi, wi in zip(GL_nodes, GL_weights):
            s = half_rho * (xi + 1.0)
            denom = 1.0 - s * s
            if denom <= 0.0:
                continue
            exponent = -(a * a - 2.0 * s * a * b + b * b) / (2.0 * denom)
            bvn += half_rho * wi * math.exp(exponent) / (two_pi * math.sqrt(denom))
        return max(0.0, min(1.0, bvn))

    @staticmethod
    def ewma_vol_forecast(log_returns, days=252):
        """RiskMetrics EWMA volatility forecast: sigma^2_t = lambda*sigma^2_{t-1} + (1-lambda)*r_{t-1}^2
        with lambda=0.94. Pure EWMA (omega=0, alpha+beta=1)."""
        try:
            returns = np.asarray(log_returns, dtype=float).reshape(-1)
            returns = returns[np.isfinite(returns)]
            if len(returns) < 30:
                return 0.0
            lam = 0.94
            variance = float(np.var(returns))
            for r_val in returns:
                variance = lam * variance + (1.0 - lam) * (r_val ** 2)
            return float(np.sqrt(variance * days))
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
    def bjerksund_stensland(S, K, T, r, q, sigma, option_type='call'):
        """
        Bjerksund-Stensland 2002 American Option Approximation.
        Two-boundary version using the paper's golden-ratio time split.
        Uses Log-Space algebra to prevent overflow errors.
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
            put_via_transform = VegaChimpCore.bjerksund_stensland(K, S, T, q, r, sigma, 'call')
            # American put should never price below the corresponding European put.
            put_euro_floor = VegaChimpCore.bs_price(S, K, r, q, sigma, T, 'put')
            return max(put_via_transform, put_euro_floor)

        b = r - q
        if b >= r:
            return VegaChimpCore.bs_price(S, K, r, q, sigma, T, 'call')

        try:
            def safe_exp(val):
                if val > 700: return float('inf')
                if val < -700: return 0.0
                return math.exp(val)

            def phi(s, t, gamma, h_val, i_val):
                lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * sigma**2) * t
                d_den = sigma * math.sqrt(t)
                d_num = -(math.log(s / h_val) + (b + (gamma - 0.5) * sigma**2) * t)
                d = d_num / d_den
                k = 2 * b / (sigma**2) + (2 * gamma - 1)
                ln_s = math.log(s)
                ln_i = math.log(i_val)
                power1 = lam + (gamma * ln_s)
                val_1 = safe_exp(power1) * VegaChimpCore.N(d)
                d2 = d - 2 * math.log(i_val / s) / d_den
                power2 = power1 + k * (ln_i - ln_s)
                val_2 = safe_exp(power2) * VegaChimpCore.N(d2)
                return val_1 - val_2

            def psi(s, t, gamma, h_val, i2, i1, t1):
                """Extended Psi function for the two-boundary 2002 version."""
                if t <= 1e-10 or t1 <= 1e-10:
                    return phi(s, t, gamma, h_val, i2)

                sqt = sigma * math.sqrt(t)
                sqt1 = sigma * math.sqrt(t1)

                e1_num = -(math.log(s / i1) + (b + (gamma - 0.5) * sigma**2) * t1)
                e1 = e1_num / sqt1

                e2_num = -(math.log(i2**2 / (s * i1)) + (b + (gamma - 0.5) * sigma**2) * t1)
                e2 = e2_num / sqt1

                e3_num = -(math.log(s / i1) - (b + (gamma - 0.5) * sigma**2) * t1)
                e3 = e3_num / sqt1

                e4_num = -(math.log(i2**2 / (s * i1)) - (b + (gamma - 0.5) * sigma**2) * t1)
                e4 = e4_num / sqt1

                f1_num = -(math.log(s / h_val) + (b + (gamma - 0.5) * sigma**2) * t)
                f1 = f1_num / sqt

                f2_num = -(math.log(i2**2 / (s * h_val)) + (b + (gamma - 0.5) * sigma**2) * t)
                f2 = f2_num / sqt

                f3_num = -(math.log(i1**2 / (s * h_val)) + (b + (gamma - 0.5) * sigma**2) * t)
                f3 = f3_num / sqt

                f4_num = -(math.log(s * i1**2 / (h_val * i2**2)) + (b + (gamma - 0.5) * sigma**2) * t)
                f4 = f4_num / sqt

                rho_val = math.sqrt(t1 / t) if t > 0 else 0

                lam = -r + gamma * b + 0.5 * gamma * (gamma - 1) * sigma**2
                kappa = 2 * b / (sigma**2) + (2 * gamma - 1)

                ln_s = math.log(s)
                ln_i2 = math.log(i2)

                power = lam * t + gamma * ln_s

                # Bivariate normal with time-correlation rho = sqrt(t1/t)
                M = VegaChimpCore._M
                term1 = safe_exp(power) * M(e1, f1, rho_val)
                term2 = safe_exp(power + kappa * (ln_i2 - ln_s)) * M(e2, f2, rho_val)
                term3 = safe_exp(power + kappa * (math.log(i1) - ln_s)) * M(e3, f3, -rho_val)
                term4 = safe_exp(power + kappa * (math.log(i1) - ln_i2)) * M(e4, f4, -rho_val)

                return term1 - term2 - term3 + term4

            # --- 2002 Two-Boundary Method ---
            beta_val = (0.5 - b / sigma**2) + math.sqrt((b / sigma**2 - 0.5)**2 + 2 * r / sigma**2)
            if abs(beta_val - 1) < 1e-5: return max(S - K, 0.0)

            boundary_inf = K * beta_val / (beta_val - 1)
            boundary_zero = max(K, (r / (r - b)) * K)
            t1 = 0.5 * (math.sqrt(5.0) - 1.0) * T

            def exercise_boundary(remaining_time):
                h_val = (-(b * remaining_time + 2 * sigma * math.sqrt(remaining_time))
                         * K**2 / ((boundary_inf - boundary_zero) * boundary_zero))
                return (boundary_zero + (boundary_inf - boundary_zero)
                        * (1 - safe_exp(h_val)))

            # I2 applies before the split; I1 applies from the split to expiry.
            I2 = exercise_boundary(T)
            I1 = exercise_boundary(T - t1)

            if S >= I2:
                intrinsic = max(S - K, 0.0)
                european = VegaChimpCore.bs_price(S, K, r, q, sigma, T, 'call')
                return max(intrinsic, european)

            # Alpha coefficients
            alpha2 = (I2 - K) * safe_exp(-beta_val * math.log(I2))
            alpha1 = (I1 - K) * safe_exp(-beta_val * math.log(I1))

            # Main formula: two-interval approximation
            term1 = alpha2 * safe_exp(beta_val * math.log(S))
            term2 = alpha2 * phi(S, t1, beta_val, I2, I2)
            term3 = phi(S, t1, 1, I2, I2)
            term4 = phi(S, t1, 1, I1, I2)
            term5 = K * phi(S, t1, 0, I2, I2)
            term6 = K * phi(S, t1, 0, I1, I2)

            # Second interval correction (the 2002 upgrade over 1993)
            term7 = alpha1 * phi(S, t1, beta_val, I1, I2)
            term8 = alpha1 * psi(S, T, beta_val, I1, I2, I1, t1)
            term9 = psi(S, T, 1, I1, I2, I1, t1)
            term10 = psi(S, T, 1, K, I2, I1, t1)
            term11 = K * psi(S, T, 0, I1, I2, I1, t1)
            term12 = K * psi(S, T, 0, K, I2, I1, t1)

            price = (term1 - term2 + term3 - term4 - term5 + term6
                     + term7 - term8 + term9 - term10 - term11 + term12)

            # The approximation is a feasible exercise strategy, so numerical
            # quadrature noise must not push it below European or intrinsic value.
            intrinsic = max(0.0, S - K)
            european = VegaChimpCore.bs_price(S, K, r, q, sigma, T, 'call')
            return max(price, intrinsic, european)

        except (OverflowError, ValueError, ZeroDivisionError):
            print("[Warning] Bjerksund-Stensland 2002 failed, falling back to Black-Scholes.")
            return VegaChimpCore.bs_price(S, K, r, q, sigma, T, 'call')
