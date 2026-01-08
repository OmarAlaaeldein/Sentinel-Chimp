import math
import numpy as np
import streamlit as st

# ===================== Black–Scholes core (no SciPy) =====================
def N(x):  # CDF
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def n(x):  # PDF
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

def d1(S, K, r, q, sig, T):
    return (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))

def d2(S, K, r, q, sig, T):
    return d1(S, K, r, q, sig, T) - sig * math.sqrt(T)

def bs_price(S, K, r, q, sig, T, kind):
    if sig <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.0
    _d1, _d2 = d1(S, K, r, q, sig, T), d2(S, K, r, q, sig, T)
    disc_q, disc_r = math.exp(-q * T), math.exp(-r * T)
    if kind == "call":
        return S * disc_q * N(_d1) - K * disc_r * N(_d2)
    else:
        return K * disc_r * N(-_d2) - S * disc_q * N(-_d1)

def greeks(S, K, r, q, sig, T, kind):
    if sig <= 0 or T <= 0 or S <= 0 or K <= 0:
        return dict(delta=0, gamma=0, theta=0, vega=0)
    _d1, _d2 = d1(S, K, r, q, sig, T), d2(S, K, r, q, sig, T)
    pdf = n(_d1)
    disc_q, disc_r = math.exp(-q * T), math.exp(-r * T)
    gamma = disc_q * pdf / (S * sig * math.sqrt(T))
    vega = S * disc_q * pdf * math.sqrt(T)  # per 1.00 vol (100 vol pts)
    if kind == "call":
        delta = disc_q * N(_d1)
        theta = (
            -S * disc_q * pdf * sig / (2 * math.sqrt(T))
            - r * K * disc_r * N(_d2)
            + q * S * disc_q * N(_d1)
        ) / 365.0
    else:
        delta = -disc_q * N(-_d1)
        theta = (
            -S * disc_q * pdf * sig / (2 * math.sqrt(T))
            + r * K * disc_r * N(-_d2)
            - q * S * disc_q * N(-_d1)
        ) / 365.0
    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega)

# ===================== EV helpers =====================
def ev_single(S, K, T, r, q, iv_imp, rv_exp, kind):
    paid = bs_price(S, K, r, q, iv_imp, T, kind)
    fair = bs_price(S, K, r, q, rv_exp, T, kind)
    ev = fair - paid
    roi = (ev / paid) if paid > 0 else 0.0
    g = greeks(S, K, r, q, iv_imp, T, kind)
    return paid, fair, ev, roi, g

def ev_straddle(S, K, T, r, q, iv_imp, rv_exp):
    c_paid = bs_price(S, K, r, q, iv_imp, T, "call")
    p_paid = bs_price(S, K, r, q, iv_imp, T, "put")
    premium = c_paid + p_paid
    c_fair = bs_price(S, K, r, q, rv_exp, T, "call")
    p_fair = bs_price(S, K, r, q, rv_exp, T, "put")
    fair = c_fair + p_fair
    ev = fair - premium
    roi = (ev / premium) if premium > 0 else 0.0
    gC = greeks(S, K, r, q, iv_imp, T, "call")
    gP = greeks(S, K, r, q, iv_imp, T, "put")
    g = dict(
        delta=gC["delta"] + gP["delta"],
        gamma=gC["gamma"] + gP["gamma"],
        theta=gC["theta"] + gP["theta"],
        vega=gC["vega"] + gP["vega"],
    )
    return premium, fair, ev, roi, g, (c_paid, p_paid)

# ===================== Scenario & Breakeven helpers =====================
def price_single_after(S, K, r, q, T1, sigma_post, move_frac, direction, kind):
    S1 = S * (1 + move_frac if direction == "Up" else 1 - move_frac)
    return bs_price(S1, K, r, q, sigma_post, T1, kind)

def price_straddle_after(S, K, r, q, T1, sig_call_post, sig_put_post,
                         move_frac, direction, skew_bump_dn_pts=0.0):
    S1 = S * (1 + move_frac if direction == "Up" else 1 - move_frac)
    sig_put_eff = sig_put_post
    if direction == "Down" and skew_bump_dn_pts > 0:
        sig_put_eff = min(3.0, sig_put_post + skew_bump_dn_pts / 100.0)
    c = bs_price(S1, K, r, q, sig_call_post, T1, "call")
    p = bs_price(S1, K, r, q, sig_put_eff, T1, "put")
    return c + p

def bisection(fn, lo, hi, tol=1e-5, maxit=60):
    f_lo, f_hi = fn(lo), fn(hi)
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if f_lo * f_hi > 0:
        return None  # no root in range
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        f_mid = fn(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)

# ===================== UI =====================
st.set_page_config(page_title="Vol Breakout EV", page_icon="📈", layout="centered")
st.title("📈 VegaChimp - Option Calculator")
st.caption("Local app. Manual inputs. No data fees. Not advice. Model why you lose money.")

# ---- Basic inputs ----
cols = st.columns(4)
with cols[0]:
    S = st.number_input("Spot (S)", value=100.0, step=0.1, min_value=0.01)
with cols[1]:
    DTE = st.number_input("Expiry (days)", min_value=1, max_value=3650, value=45, step=1)
with cols[2]:
    structure = st.selectbox("Structure", ["ATM Straddle", "Put", "Call"], index=0)
with cols[3]:
    K = st.number_input("Strike (K)", value=float(round(S, 2)))

row2 = st.columns(3)
with row2[0]:
    iv_imp = st.slider("Implied vol σ_imp (%)", 5, 200, 20) / 100.0
with row2[1]:
    sigma_hint = max(int(iv_imp * 170), int(iv_imp * 100) + 5)  # ≈1.7× or +5 pts
    rv_exp = st.slider("Expected realized σ_real (%)", 5, 300, min(300, sigma_hint)) / 100.0
with row2[2]:
    r = st.number_input("r = risk free rate (%) (e.g., 30d ~ 1M T-bill)", value=0.0) / 100.0
    q = st.number_input("q (%) dividend yield (0 if none before expiry)", value=0.0) / 100.0

T = max(1, DTE) / 365.0

# ---- Manual mids (optional) ----
manual_call = manual_put = manual_mid = None
if structure == "ATM Straddle":
    c1, c2 = st.columns(2)
    with c1:
        v = st.number_input("Manual mid (call) — optional", value=0.0, min_value=0.0, step=0.01)
        manual_call = None if v == 0.0 else v
    with c2:
        v = st.number_input("Manual mid (put) — optional", value=0.0, min_value=0.0, step=0.01)
        manual_put = None if v == 0.0 else v
else:
    v = st.number_input("Manual mid premium — optional", value=0.0, min_value=0.0, step=0.01)
    manual_mid = None if v == 0.0 else v

# ---- Event Mode (optional) ----
st.divider()
event = st.checkbox("Event/Unwind mode (crush & gap)", value=False)

# Defaults so we never hit NameError later
unwind_days = 1
gap_up = gap_dn = 0.0
p_up = 0.5
sig_call_post = sig_put_post = iv_imp

if event:
    e1, e2, e3 = st.columns(3)
    with e1:
        unwind_days = st.slider("Unwind after (days)", 1, DTE, 1)
        Tprime = max(1, DTE - unwind_days) / 365.0
    with e2:
        gap_up = st.slider("Gap up (%)", 0, 20, 0) / 100.0
        p_up = st.slider("Prob up (%)", 0, 100, 50) / 100.0
    with e3:
        gap_dn = st.slider("Gap down (%)", 0, 20, 0) / 100.0
    c1, c2 = st.columns(2)
    with c1:
        crush_call = st.slider("IV crush Call (%)", 0, 80, 10) / 100.0
    with c2:
        crush_put = st.slider("IV crush Put  (%)", 0, 80, 10) / 100.0
    sig_call_post = max(0.01, iv_imp * (1.0 - crush_call))
    sig_put_post = max(0.01, iv_imp * (1.0 - crush_put))
    st.info("Event mode: EV uses post-event vols & gaps (σ_real slider is ignored).")
else:
    Tprime = T  # not used in non-event compute paths; set for safety

# ---- Quick deterministic scenario (move + IV change) ----
st.divider()
with st.expander("⚡ Quick scenario (move + IV change, ignores probabilities)"):
    qs_cols = st.columns(4)
    with qs_cols[0]:
        dir_q = st.radio("Direction", ["Up", "Down"], index=0, horizontal=True)
    with qs_cols[1]:
        move_q = st.slider("Move (%)", 0, 30, 5) / 100.0
    with qs_cols[2]:
        d_iv_pts = st.slider("IV change (vol pts)", -50, 50, 0)
    with qs_cols[3]:
        hold_q = st.slider("Hold (days)", 1, DTE, 1)

    Tq = max(1, DTE - hold_q) / 365.0
    sig_q_call = max(0.01, iv_imp + d_iv_pts / 100.0)
    sig_q_put = sig_q_call

    if structure == "ATM Straddle":
        cp_model = bs_price(S, K, r, q, iv_imp, T, "call")
        pp_model = bs_price(S, K, r, q, iv_imp, T, "put")
        cp_e = manual_call if manual_call is not None else cp_model
        pp_e = manual_put if manual_put is not None else pp_model
        prem_q = cp_e + pp_e
        fair_q = price_straddle_after(S, K, r, q, Tq, sig_q_call, sig_q_put, move_q, dir_q)
    else:
        kind_q = "put" if structure == "Put" else "call"
        paid_model = bs_price(S, K, r, q, iv_imp, T, kind_q)
        prem_q = manual_mid if (manual_mid is not None) else paid_model
        fair_q = price_single_after(S, K, r, q, Tq, sig_q_call, move_q, dir_q, kind_q)

    ev_q = fair_q - prem_q
    colq1, colq2, colq3 = st.columns(3)
    with colq1:
        st.metric("Quick: price paid", f"{prem_q:.4f}")
    with colq2:
        st.metric("Quick: fair after move", f"{fair_q:.4f}")
    with colq3:
        st.metric("Quick: EV", f"{ev_q:.4f}")

# ===================== Compute main outputs =====================
if structure == "ATM Straddle":
    # ----- Compute fair & EV -----
    if not event:
        # Entry premium: manual or model
        cp_model = bs_price(S, K, r, q, iv_imp, T, "call")
        pp_model = bs_price(S, K, r, q, iv_imp, T, "put")
        cp = manual_call if manual_call is not None else cp_model
        pp = manual_put if manual_put is not None else pp_model
        premium = cp + pp
        fair = bs_price(S, K, r, q, rv_exp, T, "call") + bs_price(S, K, r, q, rv_exp, T, "put")
        ev = fair - premium
        if (manual_call is None) or (manual_put is None):
            st.info("Using model price at σ_imp for any leg without a manual mid. Enter real mids to avoid fantasy EV.")
    else:
        # Up scenario
        Sup = S * (1 + gap_up)
        cup = bs_price(Sup, K, r, q, sig_call_post, Tprime, "call")
        pup = bs_price(Sup, K, r, q, sig_put_post, Tprime, "put")
        # Down scenario (+small skew bump for puts)
        Sdn = S * (1 - gap_dn)
        sig_put_eff = min(3.0, sig_put_post + 0.01 * gap_dn * 100 * 1.5)  # +1.5 IV pts per −1% down
        cdn = bs_price(Sdn, K, r, q, sig_call_post, Tprime, "call")
        pdn = bs_price(Sdn, K, r, q, sig_put_eff, Tprime, "put")
        # Entry premium: manual or model
        cp_model = bs_price(S, K, r, q, iv_imp, T, "call")
        pp_model = bs_price(S, K, r, q, iv_imp, T, "put")
        cp = manual_call if manual_call is not None else cp_model
        pp = manual_put if manual_put is not None else pp_model
        premium = cp + pp
        fair = p_up * (cup + pup) + (1 - p_up) * (cdn + pdn)
        ev = fair - premium
        if (manual_call is None) or (manual_put is None):
            st.info("Using model price at σ_imp for entry. Manual mids recommended around events.")

    # ----- Display -----
    st.subheader("ATM Straddle")
    cL, cR = st.columns(2)
    with cL:
        st.metric("Premium paid", f"{premium:.4f}")
        st.write(f"Call price used: **{cp:.4f}**  |  Put: **{pp:.4f}**")
    with cR:
        st.metric("Fair under scenario", f"{fair:.4f}")
        st.metric("EV (expected)", f"{ev:.4f}")
        st.metric("Expected ROI", f"{(ev / premium * 100 if premium > 0 else 0):.2f}%")

    # ----- Guardrails -----
    if ev <= 0:
        st.error("Expected EV is negative. Looks like we're burning money.")
    elif not event:
        mult = rv_exp / iv_imp if iv_imp > 0 else 0
        if mult >= 3.0:
            st.warning("σ_real is ≥ 3× σ_imp. Likely unrealistic without a known catalyst.")
        elif mult >= 2.0:
            st.info("σ_real is ≥ 2× σ_imp. Double-check or use Event mode.")
    if premium < 0.25:
        st.warning("Tiny premium detected. ROI can look huge on paper; fills/slippage matter. Enter real mids if possible.")

    # ----- Breakeven gaps (straddle) -----
    hold_days = unwind_days if event else 1
    T1 = max(1, DTE - hold_days) / 365.0
    cp_e = cp
    pp_e = pp
    prem_e = cp_e + pp_e
    if event:
        sigC, sigP = sig_call_post, sig_put_post
        skew_bump = 1.5  # IV pts on down
    else:
        sigC = sigP = rv_exp
        skew_bump = 0.0

    def f_up_str(m):
        return price_straddle_after(S, K, r, q, T1, sigC, sigP, m, "Up") - prem_e

    def f_dn_str(m):
        return price_straddle_after(S, K, r, q, T1, sigC, sigP, m, "Down", skew_bump_dn_pts=skew_bump) - prem_e

    m_up = bisection(f_up_str, 0.0, 2.0)
    m_dn = bisection(f_dn_str, 0.0, 2.0)
    st.caption(
        "Breakeven moves (EV=0): "
        + (f"Up ≈ {m_up*100:.2f}% " if m_up is not None else "")
        + ("| " if (m_up is not None and m_dn is not None) else "")
        + (f"Down ≈ {m_dn*100:.2f}%" if m_dn is not None else "no solution in ±200%")
    )

else:
    # ===================== Single leg =====================
    kind = "put" if structure == "Put" else "call"

    if not event:
        paid_model = bs_price(S, K, r, q, iv_imp, T, kind)
        paid = manual_mid if (manual_mid is not None) else paid_model
        fair = bs_price(S, K, r, q, rv_exp, T, kind)
        ev = fair - paid
        if manual_mid is None:
            st.info("Using model price at σ_imp for entry. Enter actual mid to avoid over/understated EV.")
    else:
        paid_model = bs_price(S, K, r, q, iv_imp, T, kind)
        paid = manual_mid if (manual_mid is not None) else paid_model
        # Reprice after event
        Tprime = max(1, DTE - unwind_days) / 365.0
        Sup = S * (1 + gap_up)
        Sdn = S * (1 - gap_dn)
        sig_post = sig_put_post if kind == "put" else sig_call_post
        if kind == "put":
            sig_dn_eff = min(3.0, sig_post + 0.01 * gap_dn * 100 * 1.5)
            down_val = bs_price(Sdn, K, r, q, sig_dn_eff, Tprime, kind)
            up_val = bs_price(Sup, K, r, q, max(0.01, sig_post - 0.005 * gap_up * 100), Tprime, kind)
        else:
            down_val = bs_price(Sdn, K, r, q, sig_post, Tprime, kind)
            up_val = bs_price(Sup, K, r, q, sig_post, Tprime, kind)
        fair = p_up * up_val + (1 - p_up) * down_val
        ev = fair - paid
        if manual_mid is None:
            st.info("Using model price at σ_imp for entry. Manual mid recommended around events.")

    st.subheader(f"Single {kind.capitalize()}")
    st.metric("Price paid", f"{paid:.4f}")
    st.metric("Fair under scenario", f"{fair:.4f}")
    st.metric("EV (expected)", f"{ev:.4f}")
    st.metric("Expected ROI", f"{(ev / paid * 100 if paid > 0 else 0):.2f}%")

    # Guardrails
    if ev <= 0:
        st.error("Expected EV is negative. Are you planning to lose money?")
    elif not event:
        mult = rv_exp / iv_imp if iv_imp > 0 else 0
        if mult >= 3.0:
            st.warning("σ_real is ≥ 3× σ_imp. Likely unrealistic without a known catalyst.")
        elif mult >= 2.0:
            st.info("σ_real is ≥ 2× σ_imp. Double-check or use Event mode.")
    if paid < 0.25:
        st.warning("Tiny premium detected. ROI can look huge on paper; fills/slippage matter. Enter real mid if possible.")

    # ---- Breakeven gap (single leg) ----
    hold_days = unwind_days if event else 1
    T1 = max(1, DTE - hold_days) / 365.0
    sigma_post = (sig_put_post if kind == "put" else sig_call_post) if event else rv_exp
    paid_entry_model = bs_price(S, K, r, q, iv_imp, T, kind)
    paid_entry = manual_mid if (manual_mid is not None) else paid_entry_model

    def f_up(m):
        return price_single_after(S, K, r, q, T1, sigma_post, m, "Up", kind) - paid_entry

    def f_dn(m):
        return price_single_after(S, K, r, q, T1, sigma_post, m, "Down", kind) - paid_entry

    # Profit direction only
    if kind == "call":
        f0 = f_up(0.0)
        m_profit = bisection(f_up, 0.0, 2.0)  # 0–200% up
        if f0 >= 0:
            st.caption("Profit breakeven: already ≥ 0% at no move (EV ≥ 0).")
        elif m_profit is None:
            st.caption("Profit breakeven: no up-move solution within +200%.")
        else:
            st.caption(f"Profit breakeven: Up ≈ {m_profit*100:.2f}%")
    else:  # put
        f0 = f_dn(0.0)
        m_profit = bisection(f_dn, 0.0, 2.0)  # 0–200% down
        if f0 >= 0:
            st.caption("Profit breakeven: already ≥ 0% at no move (EV ≥ 0).")
        elif m_profit is None:
            st.caption("Profit breakeven: no down-move solution within −200%.")
        else:
            st.caption(f"Profit breakeven: Down ≈ {m_profit*100:.2f}%")

    # (Optional) loss breakeven for power users
    with st.expander("Show loss breakeven (optional)"):
        if kind == "call":
            m_loss = bisection(f_dn, 0.0, 2.0)
            if m_loss is None:
                st.caption("Loss breakeven: no down-move solution within −200%.")
            else:
                st.caption(f"Loss breakeven: Down ≈ {m_loss*100:.2f}%")
        else:
            m_loss = bisection(f_up, 0.0, 2.0)
            if m_loss is None:
                st.caption("Loss breakeven: no up-move solution within +200%.")
            else:
                st.caption(f"Loss breakeven: Up ≈ {m_loss*100:.2f}%")

st.divider()
st.caption(
    "EV = fair value if your scenario happens − what you pay at σ_imp (or your mid). "
    "Positive EV needs σ_real > σ_imp or a favorable gap. Not advice."
)
