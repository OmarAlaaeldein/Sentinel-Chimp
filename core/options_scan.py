"""Pure helpers for options-scan liquidity filters and tradeable-edge verdicts.

Fair value is priced with forecast vol only (EWMA / optional GARCH blend) —
never with the contract's own IV — so "edge vs mid" is not circular.
Verdicts require beating the tradeable side of the quote (ask to buy, bid to sell).
"""
from __future__ import annotations

from typing import Tuple

# Liquidity / quote quality
MAX_SPREAD_FRAC = 0.20  # (ask-bid)/mid
MIN_MID = 0.05
MIN_OI = 10
MIN_VOLUME = 5

# Moneyness
STRIKE_BAND = 0.12  # |K/S - 1|
DELTA_LO = 0.20
DELTA_HI = 0.65

# Tradeable edge
MIN_EDGE_ABS = 0.10
HALF_SPREAD_PAD = 0.05
MIN_EDGE_PCT = 0.08
EARNINGS_BUMP = 0.05


def quote_passes_liquidity(
    bid: float,
    ask: float,
    open_interest: float,
    volume: float,
    *,
    max_spread_frac: float = MAX_SPREAD_FRAC,
    min_mid: float = MIN_MID,
    min_oi: int = MIN_OI,
    min_volume: int = MIN_VOLUME,
) -> bool:
    """Hard quote-quality filter: real BBO, tight spread, some interest, not junk."""
    try:
        bid_f = float(bid)
        ask_f = float(ask)
        oi_f = float(open_interest) if open_interest is not None else 0.0
        vol_f = float(volume) if volume is not None else 0.0
    except (TypeError, ValueError):
        return False
    if not (bid_f > 0.0 and ask_f > 0.0):
        return False
    if ask_f < bid_f:
        return False
    mid = 0.5 * (bid_f + ask_f)
    if mid < min_mid or mid <= 0.0:
        return False
    if (ask_f - bid_f) / mid > max_spread_frac:
        return False
    if oi_f < min_oi and vol_f < min_volume:
        return False
    return True


def near_atm_strike(strike: float, spot: float, band: float = STRIKE_BAND) -> bool:
    """Cheap prefilter: keep strikes within ``band`` of spot."""
    try:
        k = float(strike)
        s = float(spot)
    except (TypeError, ValueError):
        return False
    if s <= 0.0 or k <= 0.0:
        return False
    return abs(k / s - 1.0) <= band


def delta_in_band(delta: float, lo: float = DELTA_LO, hi: float = DELTA_HI) -> bool:
    """Prefer near-ATM by absolute delta once Greeks are available."""
    try:
        d = abs(float(delta))
    except (TypeError, ValueError):
        return False
    return lo <= d <= hi


def half_spread(bid: float, ask: float) -> float:
    return 0.5 * (float(ask) - float(bid))


def tradeable_edge(fair: float, bid: float, ask: float) -> Tuple[float, float, float]:
    """Return ``(edge_long, edge_short, mid)``.

    * ``edge_long``  = fair − ask  (must beat the ask to buy)
    * ``edge_short`` = bid − fair (must beat the bid to sell)
    * ``mid``        = (bid + ask) / 2
    """
    bid_f = float(bid)
    ask_f = float(ask)
    fair_f = float(fair)
    mid = 0.5 * (bid_f + ask_f)
    return fair_f - ask_f, bid_f - fair_f, mid


def min_abs_edge(bid: float, ask: float, *, is_earnings: bool = False) -> float:
    """Dollar hurdle: ``max(0.10, half_spread + 0.05)``, +earnings bump."""
    bump = EARNINGS_BUMP if is_earnings else 0.0
    hs = half_spread(bid, ask)
    return max(MIN_EDGE_ABS + bump, hs + HALF_SPREAD_PAD + bump)


def scan_verdict(
    fair: float,
    bid: float,
    ask: float,
    *,
    is_earnings: bool = False,
    min_edge_pct: float = MIN_EDGE_PCT,
) -> Tuple[str, float, float]:
    """Classify Under / Over / Fair from tradeable edge.

    Returns ``(verdict, ev_at_ask, edge_pct)`` where ``ev_at_ask = fair - ask``
    (buy-side tradeable edge for the EV column) and ``edge_pct`` is the
    relative edge used for the winning side (long for Under, short for Over,
    long for Fair).
    """
    edge_long, edge_short, mid = tradeable_edge(fair, bid, ask)
    hurdle = min_abs_edge(bid, ask, is_earnings=is_earnings)
    pct_long = (edge_long / mid) if mid > 0 else 0.0
    pct_short = (edge_short / mid) if mid > 0 else 0.0

    if edge_long > hurdle and pct_long >= min_edge_pct:
        label = "Earnings Under" if is_earnings else "Under"
        return label, edge_long, pct_long
    if edge_short > hurdle and pct_short >= min_edge_pct:
        label = "Earnings Over" if is_earnings else "Over"
        return label, edge_long, pct_short
    return "Fair", edge_long, pct_long


SCAN_RULES_LOG = (
    "Scan rules: BBO only (no last), spread≤20%, OI≥10|Vol≥5, mid≥$0.05; "
    "near-ATM (|K/S−1|≤12% then |Δ|∈[0.20,0.65]); "
    "FV=forecast vol only (EWMA±GARCH); "
    "Under if fair−ask beats max($0.10,½spread+$0.05) and ≥8% of mid "
    "(Over vs bid); earnings +$0.05 buffer. EV column = EV@Ask (fair−ask)."
)
