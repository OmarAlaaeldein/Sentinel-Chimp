"""Shared options-scan orchestration for GUI and CLI.

Extracts the Options Finder pricing / filter / verdict loop from
``MarketApp.fetch_options_batch`` so both surfaces share one code path.
Fair value = Bjerksund-Stensland under **forecast vol only** (EWMA ± optional
GARCH blend) — never contract IV.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.pricing import VegaChimpCore
from core.technicals import calculate_technicals
from core.vol_models import (
    blend_forecast_vol,
    fit_quadratic_smile,
    garch11_vol_forecast,
    smile_vol_arr,
)
from core.options_scan import (
    SCAN_RULES_LOG,
    delta_in_band,
    near_atm_strike,
    quote_passes_liquidity,
    scan_verdict,
)

LogFn = Callable[[str], None]
UiBatchFn = Callable[[List[Tuple[tuple, str]]], None]

_BATCH_N = 64
UI_BATCH_SIZE = 40


def to_finite_float(value: Any) -> Optional[float]:
    try:
        f_val = float(value)
        if not math.isfinite(f_val):
            return None
        return f_val
    except (TypeError, ValueError):
        return None


def normalize_div_yield(div: Any) -> Optional[float]:
    """Normalize dividend yield to decimal (e.g. 0.0294 for 2.94%)."""
    if div is None:
        return None
    try:
        d = float(div)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(d) or d < 0:
        return None
    if d > 1:
        d = d / 100.0
    return d


def resolve_dividend_yield(data_provider, stock_obj) -> float:
    """Priority: fast_info → info.dividendYield → trailing; else 0.0."""
    try:
        fast_div = normalize_div_yield(
            data_provider.get_fast_info(stock_obj).get("dividend_yield")
        )
        if fast_div is not None:
            return fast_div
        info = data_provider.get_info(stock_obj)
        div = normalize_div_yield(info.get("dividendYield"))
        if div is not None:
            return div
        div = normalize_div_yield(info.get("trailingAnnualDividendYield"))
        if div is not None:
            return div
    except Exception:
        pass
    return 0.0


def earnings_contract_set(
    projected_earnings: Optional[Sequence],
    all_exps: Optional[Sequence[str]],
) -> set:
    """Map projected earnings dates to the nearest option expiry on/after each."""
    out: set = set()
    if not projected_earnings or not all_exps:
        return out
    for p_date in projected_earnings:
        try:
            p_str = p_date.strftime("%Y-%m-%d") if hasattr(p_date, "strftime") else str(p_date)[:10]
        except Exception:
            continue
        valid_exps = [e for e in all_exps if e >= p_str]
        if valid_exps:
            out.add(min(valid_exps))
    return out


def resolve_forecast_vol(
    ewma_vol: float,
    garch_vol: float,
    use_garch_blend: bool,
    hv_30: float = 0.0,
    fallback: float = 0.25,
) -> float:
    forecast_vol = blend_forecast_vol(ewma_vol, garch_vol, use_garch_blend)
    if forecast_vol is None or forecast_vol <= 0:
        hv = to_finite_float(hv_30)
        forecast_vol = hv if hv and hv > 0 else fallback
    return float(forecast_vol)


def interpolate_rfr(short_rate: float, long_rate: float, T: float) -> float:
    if T <= 0.25:
        return short_rate
    t_clamped = min(max(T, 0.25), 10.0)
    weight = (t_clamped - 0.25) / (10.0 - 0.25)
    return short_rate + weight * (long_rate - short_rate)


def load_projected_earnings(data_provider, stock) -> list:
    """Best-effort earnings cycle from the provider calendar (matches GUI)."""
    from datetime import timedelta
    projected = []
    try:
        cal = data_provider.get_calendar(stock)
        anchor_date = None
        if isinstance(cal, dict) and "Earnings Date" in cal:
            dates = cal["Earnings Date"]
            if dates:
                anchor_date = pd.to_datetime(dates[0]).date()
        elif cal is not None and hasattr(cal, "empty") and not cal.empty:
            anchor_date = pd.to_datetime(cal.iloc[0].values[0]).date()
        if anchor_date:
            projected = [anchor_date]
            for i in range(1, 4):
                projected.append(anchor_date + timedelta(days=91 * i))
    except Exception:
        pass
    return projected


@dataclass
class OptionScanRow:
    date: str
    type: str
    strike: float
    volume: float
    oi: int
    mid: float
    spread_pct: float
    breakeven: float
    iv: float
    fair: float
    ev_at_ask: float
    edge_pct: float
    delta: float
    gamma: float
    theta: float
    vega: float
    pop: float
    verdict: str
    is_earnings: bool = False
    tag: str = ""

    def tree_vals(self) -> tuple:
        """Tuple matching Options Explorer column order."""
        return (
            self.date,
            self.type,
            f"{self.strike:.2f}",
            int(self.volume),
            self.oi,
            f"{self.mid:.2f}",
            f"{self.spread_pct:.0f}%",
            f"{self.breakeven:.2f}",
            f"{self.iv:.1%}",
            f"{self.fair:.2f}",
            f"{self.ev_at_ask:+.2f}",
            f"{self.delta:.3f}",
            f"{self.gamma:.4f}",
            f"{self.theta:.3f}",
            f"{self.vega:.3f}",
            f"{self.pop:.0f}%",
            self.verdict,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanResult:
    rows: List[OptionScanRow] = field(default_factory=list)
    scan_buf: List[dict] = field(default_factory=list)
    forecast_vol: float = 0.0
    spot: float = 0.0
    dividend_yield: float = 0.0
    rules_log: str = SCAN_RULES_LOG


@dataclass
class TickerAnalysis:
    ticker: str
    spot: float
    hv_30: float
    ewma_vol: float
    garch_vol: float
    garch_ok: bool
    technicals: dict
    summary_lines: List[str]

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "spot": self.spot,
            "hv_30": self.hv_30,
            "ewma_vol": self.ewma_vol,
            "garch_vol": self.garch_vol,
            "garch_ok": self.garch_ok,
            "technicals": self.technicals,
            "summary_lines": self.summary_lines,
        }


def _fmt_tech(val: Any, digits: int = 2) -> str:
    try:
        f = float(val)
        if not math.isfinite(f):
            return "N/A"
        return f"{f:.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def analyze_ticker(
    data_provider,
    symbol: str,
    *,
    log: Optional[LogFn] = None,
) -> TickerAnalysis:
    """Spot, HV/EWMA/(GARCH), and a short technicals summary — no tkinter."""
    sym = symbol.strip().upper()
    stock = data_provider.create_ticker(sym)
    df_tech = data_provider.fetch_history(stock, "1y", "1d", log=log)
    if df_tech is None or df_tech.empty:
        raise RuntimeError(f"No daily history for {sym}")
    df_tech = calculate_technicals(df_tech.copy())
    df_tech["log_ret"] = np.log(df_tech["Close"] / df_tech["Close"].shift(1))
    last = df_tech.iloc[-1]

    spot = None
    try:
        spot = to_finite_float(data_provider.get_fast_last_price(stock))
    except Exception:
        spot = None
    if spot is None or spot <= 0:
        spot = float(last["Close"])

    recent_returns = df_tech["log_ret"].dropna().tail(30)
    hv_30 = (
        float(recent_returns.std()) * np.sqrt(252)
        if len(recent_returns) >= 2
        else 0.0
    )
    log_rets = df_tech["log_ret"].dropna().values
    ewma_vol = float(VegaChimpCore.ewma_vol_forecast(log_rets))
    garch_vol, garch_info = garch11_vol_forecast(log_rets)
    garch_ok = bool(garch_info.get("ok"))

    tech = {
        "RSI": to_finite_float(last.get("RSI")),
        "StochRSI": to_finite_float(last.get("StochRSI")),
        "MACD": to_finite_float(last.get("MACD")),
        "MACD_Hist": to_finite_float(last.get("MACD_Hist")),
        "ADX": to_finite_float(last.get("ADX")),
        "ATR": to_finite_float(last.get("ATR")),
        "+DI": to_finite_float(last.get("+DI")),
        "-DI": to_finite_float(last.get("-DI")),
        "BB_PctB": to_finite_float(last.get("BB_PctB")),
        "BB_Upper": to_finite_float(last.get("BB_Upper")),
        "BB_Lower": to_finite_float(last.get("BB_Lower")),
    }

    rsi = tech["RSI"]
    rsi_note = (
        "oversold" if rsi is not None and rsi < 30
        else "overbought" if rsi is not None and rsi > 70
        else "neutral"
    )
    adx = tech["ADX"]
    plus_di, minus_di = tech["+DI"], tech["-DI"]
    if adx is not None and plus_di is not None and minus_di is not None:
        trend = "bull" if plus_di > minus_di else "bear"
        strength = "strong" if adx > 25 else "weak" if adx < 20 else "neutral"
        adx_note = f"{strength} {trend}"
    else:
        adx_note = "n/a"

    summary_lines = [
        f"{sym}  spot=${spot:.2f}",
        f"Vol  HV30={hv_30:.1%}  EWMA={ewma_vol:.1%}"
        + (f"  GARCH={garch_vol:.1%}" if garch_ok and garch_vol > 0 else ""),
        f"Technicals  RSI={_fmt_tech(rsi)} ({rsi_note})  "
        f"MACD={_fmt_tech(tech['MACD'])}  ADX={_fmt_tech(adx, 1)} ({adx_note})  "
        f"ATR=${_fmt_tech(tech['ATR'])}",
    ]

    return TickerAnalysis(
        ticker=sym,
        spot=float(spot),
        hv_30=float(hv_30),
        ewma_vol=float(ewma_vol),
        garch_vol=float(garch_vol) if garch_ok else 0.0,
        garch_ok=garch_ok,
        technicals=tech,
        summary_lines=summary_lines,
    )


def scan_option_chains(
    *,
    data_provider,
    stock,
    spot: float,
    dates: Sequence[str],
    all_exps: Optional[Sequence[str]] = None,
    projected_earnings: Optional[Sequence] = None,
    ewma_vol: float = 0.0,
    garch_vol: float = 0.0,
    hv_30: float = 0.0,
    use_garch_blend: bool = False,
    use_smile_vol: bool = False,
    use_american_greeks: bool = True,
    option_type: str = "all",
    under_only: bool = False,
    dividend_yield: Optional[float] = None,
    short_rate: Optional[float] = None,
    long_rate: Optional[float] = None,
    log: Optional[LogFn] = None,
    on_ui_batch: Optional[UiBatchFn] = None,
) -> ScanResult:
    """Run the Options Finder scan over ``dates``.

    Same liquidity / ATM / delta / ``scan_verdict`` rules as the GUI.
    When ``on_ui_batch`` is provided, progressive UI flushes happen for the
    non-``under_only`` path (and a final sorted flush for ``under_only``).
    """
    def _log(msg: str) -> None:
        if log:
            log(msg)

    spot = float(spot)
    if dividend_yield is None:
        dividend_yield = resolve_dividend_yield(data_provider, stock)
    DIV_YIELD = float(dividend_yield)

    if short_rate is None or long_rate is None:
        _short_rate, _long_rate = data_provider.fetch_rate_curve()
    else:
        _short_rate, _long_rate = float(short_rate), float(long_rate)

    earnings_contracts = earnings_contract_set(
        projected_earnings, all_exps if all_exps is not None else dates,
    )
    forecast_vol = resolve_forecast_vol(
        ewma_vol, garch_vol, use_garch_blend, hv_30=hv_30,
    )

    opt_type = (option_type or "all").strip().lower()
    today = datetime.now().date()
    ui_batch: List[Tuple[tuple, str]] = []
    under_rows: List[Tuple[float, OptionScanRow]] = []
    rows_out: List[OptionScanRow] = []
    scan_buf: List[dict] = []
    rules_logged = False

    def flush_ui() -> None:
        nonlocal ui_batch
        if not ui_batch or on_ui_batch is None:
            ui_batch = []
            return
        batch = ui_batch
        ui_batch = []
        on_ui_batch(batch)

    for date in dates:
        try:
            if not rules_logged:
                _log(SCAN_RULES_LOG)
                rules_logged = True

            exp_date = datetime.strptime(date, "%Y-%m-%d").date()
            trading_days = int(np.busday_count(today, exp_date))
            T = max(trading_days / 252.0, 1 / 252)
            RFR = interpolate_rfr(_short_rate, _long_rate, T)

            chain = data_provider.get_option_chain(stock, date)
            frames = []
            if opt_type in ("all", "call"):
                frames.append(chain.calls.assign(Type="CALL"))
            if opt_type in ("all", "put"):
                frames.append(chain.puts.assign(Type="PUT"))
            if not frames:
                continue
            all_options = pd.concat(frames, ignore_index=True)
            if all_options.empty:
                continue
            if "volume" in all_options.columns:
                all_options = all_options.sort_values(
                    "volume", ascending=False, kind="mergesort"
                )

            bid = pd.to_numeric(all_options.get("bid", 0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            ask = pd.to_numeric(all_options.get("ask", 0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            vol = pd.to_numeric(all_options.get("volume", 0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            oi_arr = pd.to_numeric(all_options.get("openInterest", 0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            iv_arr = pd.to_numeric(all_options.get("impliedVolatility", 0), errors="coerce").to_numpy(dtype=float)
            strikes = pd.to_numeric(all_options["strike"], errors="coerce").to_numpy(dtype=float)
            types = all_options["Type"].to_numpy()

            has_ba = (bid > 0) & (ask > 0) & (ask >= bid)
            mid = np.where(has_ba, 0.5 * (bid + ask), np.nan)
            spread_frac = np.where(
                has_ba & (mid > 0),
                (ask - bid) / mid,
                np.nan,
            )
            spread_pct = np.where(np.isfinite(spread_frac), spread_frac * 100.0, 999.0)

            parity_map = {}
            for i in range(len(strikes)):
                if not has_ba[i] or not np.isfinite(mid[i]) or mid[i] <= 0:
                    continue
                s = float(strikes[i])
                bucket = parity_map.setdefault(s, {})
                bucket[str(types[i]).lower()] = float(mid[i])

            sqrtT = math.sqrt(T)
            is_earnings = date in earnings_contracts
            parity_bounds = VegaChimpCore.american_put_call_parity_bounds
            Ncdf = VegaChimpCore.N

            kind_is_call = types == "CALL"
            oi_int = np.where(np.isfinite(oi_arr), oi_arr, 0.0)

            liquid = np.array([
                quote_passes_liquidity(bid[i], ask[i], oi_int[i], vol[i])
                for i in range(len(strikes))
            ], dtype=bool)
            atm = np.array([
                near_atm_strike(strikes[i], spot) for i in range(len(strikes))
            ], dtype=bool)
            valid = (
                liquid & atm
                & np.isfinite(mid) & (mid > 0)
                & np.isfinite(iv_arr) & (iv_arr >= 0.01)
                & np.isfinite(strikes) & (strikes > 0)
            )

            idx = np.flatnonzero(valid)
            if idx.size == 0:
                continue

            strikes_v = strikes[idx]
            mid_v = mid[idx]
            bid_v = bid[idx]
            ask_v = ask[idx]
            iv_v = iv_arr[idx]
            vol_v = vol[idx]
            oi_v = oi_int[idx]
            sp_v = spread_pct[idx]
            is_call_v = kind_is_call[idx]
            types_v = types[idx]

            iv_display = iv_v.copy()
            forward = spot * math.exp((RFR - DIV_YIELD) * T)
            if use_smile_vol:
                smile_coef = fit_quadratic_smile(strikes_v, iv_v, forward)
                if smile_coef is not None:
                    iv_display = smile_vol_arr(strikes_v, forward, smile_coef)

            vol_input = np.full(idx.size, float(forecast_vol), dtype=float)
            kinds = np.where(is_call_v, "call", "put")
            if idx.size >= _BATCH_N:
                fair_v = VegaChimpCore.bjerksund_stensland_batch(
                    spot, strikes_v, T, RFR, DIV_YIELD, vol_input, kinds,
                )
            else:
                fair_v = np.array([
                    VegaChimpCore.bjerksund_stensland(
                        spot, float(strikes_v[j]), T, RFR, DIV_YIELD,
                        float(vol_input[j]), kinds[j],
                    )
                    for j in range(idx.size)
                ], dtype=float)

            greeks_map = None
            if use_american_greeks and idx.size >= _BATCH_N:
                greeks_map = VegaChimpCore.american_greeks_batch(
                    spot, strikes_v, RFR, DIV_YIELD, iv_v, T, kinds,
                )

            for j, i in enumerate(idx):
                fair = float(fair_v[j])
                if fair <= 0 or not math.isfinite(fair):
                    continue
                market_price = float(mid_v[j])
                strike = float(strikes_v[j])
                iv = float(iv_display[j])
                iv_mkt = float(iv_v[j])
                kind_str = kinds[j]
                b = float(bid_v[j])
                a = float(ask_v[j])
                oi = int(oi_v[j])
                sp = float(sp_v[j])

                if greeks_map is not None:
                    greeks = {
                        "delta": float(greeks_map["delta"][j]),
                        "gamma": float(greeks_map["gamma"][j]),
                        "theta": float(greeks_map["theta"][j]),
                        "vega": float(greeks_map["vega"][j]),
                    }
                elif use_american_greeks:
                    greeks = VegaChimpCore.american_greeks(
                        spot, strike, RFR, DIV_YIELD, iv_mkt, T, kind_str,
                    )
                else:
                    greeks = VegaChimpCore.bs_greeks(
                        spot, strike, RFR, DIV_YIELD, iv_mkt, T, kind_str,
                    )

                if not delta_in_band(greeks.get("delta", float("nan"))):
                    continue

                verdict, ev_at_ask, edge_pct = scan_verdict(
                    fair, b, a, is_earnings=is_earnings,
                )

                try:
                    if kind_str == "call":
                        breakeven_price = strike + market_price
                    else:
                        breakeven_price = strike - market_price
                    if breakeven_price <= 0:
                        pop = 0.0
                    else:
                        d2_pop = (
                            math.log(spot / breakeven_price)
                            + (RFR - DIV_YIELD - 0.5 * iv_mkt * iv_mkt) * T
                        ) / (iv_mkt * sqrtT)
                        pop = Ncdf(d2_pop) * 100 if kind_str == "call" else Ncdf(-d2_pop) * 100
                        pop = max(0.0, min(100.0, pop))
                except Exception:
                    pop = 0.0

                parity_warn = ""
                parity_data = parity_map.get(strike, {})
                if "call" in parity_data and "put" in parity_data:
                    lower, upper = parity_bounds(spot, strike, RFR, DIV_YIELD, T)
                    observed = parity_data["call"] - parity_data["put"]
                    if observed < lower - 0.10 or observed > upper + 0.10:
                        parity_warn = "!"

                is_undervalued = "Under" in verdict
                scan_buf.append({
                    "date": date,
                    "type": types_v[j],
                    "strike": strike,
                    "ev": ev_at_ask,
                    "vol": float(vol_v[j]),
                    "is_earnings": is_earnings,
                    "is_good": is_undervalued,
                })

                breakeven = strike + market_price if kind_str == "call" else strike - market_price
                tag = ""
                if is_undervalued:
                    tag = "green"
                elif "Over" in verdict:
                    tag = "red"
                display_verdict = f"{verdict} {parity_warn}".strip() if parity_warn else verdict
                if under_only and "Under" not in display_verdict:
                    continue

                row = OptionScanRow(
                    date=date,
                    type=str(types_v[j]),
                    strike=strike,
                    volume=float(vol_v[j]),
                    oi=oi,
                    mid=market_price,
                    spread_pct=sp,
                    breakeven=breakeven,
                    iv=iv,
                    fair=fair,
                    ev_at_ask=ev_at_ask,
                    edge_pct=float(edge_pct),
                    delta=float(greeks["delta"]),
                    gamma=float(greeks["gamma"]),
                    theta=float(greeks["theta"]),
                    vega=float(greeks["vega"]),
                    pop=float(pop),
                    verdict=display_verdict,
                    is_earnings=is_earnings,
                    tag=tag,
                )
                rows_out.append(row)
                vals = row.tree_vals()
                if under_only:
                    under_rows.append((edge_pct, row))
                else:
                    ui_batch.append((vals, tag))
                    if len(ui_batch) >= UI_BATCH_SIZE:
                        flush_ui()

        except Exception as e:
            _log(f"Options fetch error for {date}: {e}")

    if under_only and under_rows:
        under_rows.sort(key=lambda r: r[0], reverse=True)
        # Re-order rows_out to match edge ranking for under-only
        rows_out = [r for _pct, r in under_rows]
        for _pct, row in under_rows:
            ui_batch.append((row.tree_vals(), row.tag))
            if len(ui_batch) >= UI_BATCH_SIZE:
                flush_ui()
    elif under_only:
        rows_out = []

    flush_ui()

    return ScanResult(
        rows=rows_out,
        scan_buf=scan_buf,
        forecast_vol=forecast_vol,
        spot=spot,
        dividend_yield=DIV_YIELD,
        rules_log=SCAN_RULES_LOG,
    )


def run_ticker_scan(
    data_provider,
    symbol: str,
    *,
    under_only: bool = False,
    max_expiries: Optional[int] = None,
    option_type: str = "all",
    use_garch_blend: bool = False,
    use_smile_vol: bool = False,
    use_american_greeks: bool = True,
    log: Optional[LogFn] = None,
) -> Tuple[TickerAnalysis, ScanResult]:
    """End-to-end: analyze ticker context, then scan option chains."""
    analysis = analyze_ticker(data_provider, symbol, log=log)
    stock = data_provider.create_ticker(analysis.ticker)
    expiries = list(data_provider.get_option_expirations(stock) or ())
    if max_expiries is not None and max_expiries > 0:
        expiries = expiries[: int(max_expiries)]
    projected = load_projected_earnings(data_provider, stock)
    result = scan_option_chains(
        data_provider=data_provider,
        stock=stock,
        spot=analysis.spot,
        dates=expiries,
        all_exps=list(data_provider.get_option_expirations(stock) or ()),
        projected_earnings=projected,
        ewma_vol=analysis.ewma_vol,
        garch_vol=analysis.garch_vol,
        hv_30=analysis.hv_30,
        use_garch_blend=use_garch_blend,
        use_smile_vol=use_smile_vol,
        use_american_greeks=use_american_greeks,
        option_type=option_type,
        under_only=under_only,
        log=log,
    )
    return analysis, result
