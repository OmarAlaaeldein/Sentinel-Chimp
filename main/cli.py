"""Headless CLI for Sentinel-Chimp (no tkinter).

Usage examples:
  python -m main.cli analyze LULU
  python -m main.cli scan LULU --under-only --max-expiries 6
  python -m main.cli scan SPY --under-only --div 0.98% --limit 10 --csv out.csv
  python -m main.cli verify
  python sentinel.py scan LULU --json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from typing import List, Optional, Sequence

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core.ollama import LocalModelError


def build_parser(machine=False) -> argparse.ArgumentParser:
    class Parser(argparse.ArgumentParser):
        def error(self, message):
            if machine:
                raise LocalModelError("INVALID_ARGUMENT", message)
            super().error(message)

    parser = Parser(
        prog="sentinel",
        description="Sentinel-Chimp CLI — analyze tickers and scan options headlessly.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="Spot, EWMA/HV, short technicals summary")
    p_an.add_argument("ticker", help="Symbol, e.g. LULU")
    p_an.add_argument("--json", action="store_true", help="Emit JSON instead of text")

    p_scan = sub.add_parser(
        "scan",
        help="Options Finder scan (same rules as GUI)",
    )
    p_scan.add_argument("ticker", help="Symbol, e.g. LULU")
    p_scan.add_argument(
        "--under-only",
        action="store_true",
        help="Only print Under / Earnings Under, ranked by edge %%",
    )
    p_scan.add_argument(
        "--max-expiries",
        type=int,
        default=None,
        metavar="N",
        help="Limit to the first N listed expirations",
    )
    p_scan.add_argument(
        "--expiry",
        action="append",
        default=None,
        metavar="DATE",
        help="Only scan this expiry (YYYY-MM-DD prefix match); repeatable",
    )
    p_scan.add_argument(
        "--type",
        dest="option_type",
        choices=("call", "put", "all"),
        default="all",
        help="Option side filter (default: all)",
    )
    p_scan.add_argument(
        "--garch",
        action="store_true",
        help="Blend EWMA with fitted GARCH(1,1) for forecast vol",
    )
    p_scan.add_argument(
        "--smile",
        action="store_true",
        help="Smooth display IV with the per-expiry quadratic smile fit",
    )
    p_scan.add_argument(
        "--euro-greeks",
        action="store_true",
        help="Use analytic European Greeks instead of American FD Greeks",
    )
    p_scan.add_argument(
        "--div",
        default=None,
        metavar="YIELD",
        help="Dividend yield override: decimal (0.0098) or percent (0.98%%)",
    )
    p_scan.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Print only the first N rows",
    )
    p_scan.add_argument(
        "--csv",
        default=None,
        metavar="PATH",
        help="Also write full scan rows to a CSV file",
    )
    p_scan.add_argument("--json", action="store_true", help="Emit JSON instead of text")

    sub.add_parser("verify", help="Offline math self-test (no network needed)")

    from main.model_cli import add_commands
    add_commands(sub)
    from main.graph_cli import add_commands as add_graph_commands
    add_graph_commands(sub)
    return parser


def parse_div_yield(raw: Optional[str]) -> Optional[float]:
    """Parse ``--div``: ``0.98%`` → 0.0098, ``0.0098`` → 0.0098."""
    if raw is None:
        return None
    text = raw.strip()
    try:
        if text.endswith("%"):
            value = float(text[:-1]) / 100.0
        else:
            value = float(text)
            if value > 1.0:
                # Bare number above 100% can only be percent shorthand.
                value = value / 100.0
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid dividend yield: {raw!r}")
    if not math.isfinite(value) or value < 0 or value > 0.25:
        raise argparse.ArgumentTypeError(
            f"Dividend yield out of range [0, 25%]: {raw!r}"
        )
    return value


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _print_analysis(analysis, as_json: bool) -> None:
    if as_json:
        print(json.dumps(_json_safe(analysis.to_dict()), indent=2, allow_nan=False))
        return
    for line in analysis.summary_lines:
        print(line)


def _print_scan(analysis, result, *, under_only: bool, as_json: bool,
                limit: Optional[int] = None) -> None:
    rows = result.rows
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    if as_json:
        payload = {
            "analysis": analysis.to_dict(),
            "forecast_vol": result.forecast_vol,
            "dividend_yield": result.dividend_yield,
            "rules": result.rules_log,
            "count": len(rows),
            "total_count": len(result.rows),
            "requested_expiries": result.requested_expiries,
            "rows": [r.to_dict() for r in rows],
            "errors": result.errors,
            "status": "partial" if result.errors and len(result.errors) < result.requested_expiries else "error" if result.errors else "ok",
        }
        print(json.dumps(_json_safe(payload), indent=2, allow_nan=False))
        return

    for line in analysis.summary_lines:
        print(line)
    print(
        f"Forecast vol={result.forecast_vol:.1%}  "
        f"div={result.dividend_yield:.2%}  "
        f"contracts={len(rows)}"
        + ("  (Under only, ranked by edge %)" if under_only else "")
    )
    if not rows:
        print("No contracts matched scan filters.")
        return

    header = (
        f"{'Expiry':<12} {'Type':<5} {'Strike':>8} {'Mid':>7} {'Fair':>7} "
        f"{'EV@Ask':>8} {'Edge%':>7} {'Delta':>7} {'IV':>6} {'OI':>7} Verdict"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.date:<12} {r.type:<5} {r.strike:8.2f} {r.mid:7.2f} {r.fair:7.2f} "
            f"{r.ev_at_ask:+8.2f} {r.edge_pct * 100:6.1f}% {r.delta:7.3f} "
            f"{r.iv:5.1%} {r.oi:7d} {r.verdict}"
        )


def _write_csv(path: str, rows) -> None:
    from dataclasses import fields
    from core.scan_service import OptionScanRow
    fieldnames = [field.name for field in fields(OptionScanRow)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_dict())
    print(f"Wrote {len(rows)} rows to {path}", file=sys.stderr)


def cmd_analyze(args: argparse.Namespace) -> int:
    from core.data import YFinanceProvider
    from core.scan_service import analyze_ticker
    provider = YFinanceProvider()
    analysis = analyze_ticker(provider, args.ticker)
    _print_analysis(analysis, args.json)
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    from core.data import YFinanceProvider
    from core.scan_service import run_ticker_scan
    provider = YFinanceProvider()

    def _log(msg: str) -> None:
        if not args.json:
            print(msg, file=sys.stderr)

    analysis, result = run_ticker_scan(
        provider,
        args.ticker,
        under_only=args.under_only,
        max_expiries=args.max_expiries,
        expiries=args.expiry,
        option_type=args.option_type,
        use_garch_blend=args.garch,
        use_smile_vol=args.smile,
        use_american_greeks=not args.euro_greeks,
        dividend_yield=parse_div_yield(args.div),
        log=_log,
    )
    if args.csv:
        _write_csv(args.csv, result.rows)
    _print_scan(analysis, result, under_only=args.under_only,
                as_json=args.json, limit=args.limit)
    if result.errors:
        return 4 if len(result.errors) == result.requested_expiries else 5
    return 0


def cmd_verify(_args: argparse.Namespace) -> int:
    """Offline math self-test: BSM / BS2002 / IV / EWMA / cone / parity."""
    import numpy as np
    from core.pricing import VegaChimpCore
    from core.vol_models import probability_cone
    failures: List[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    V = VegaChimpCore

    # --- Black-Scholes vs Hull benchmarks ---
    c = V.bs_price(100, 100, 0.05, 0.0, 0.20, 1.0, "call")
    check("BSM ATM call = 10.4506 (Hull)", abs(c - 10.4506) < 1e-3, f"{c:.4f}")
    p = V.bs_price(100, 100, 0.05, 0.0, 0.20, 1.0, "put")
    check("BSM ATM put = 5.5735 (Hull)", abs(p - 5.5735) < 1e-3, f"{p:.4f}")
    c2 = V.bs_price(42, 40, 0.10, 0.0, 0.20, 0.5, "call")
    check("BSM 42/40 call ~ 4.76 (Hull)", abs(c2 - 4.76) < 0.02, f"{c2:.4f}")

    # --- Put-call parity ---
    S, K, r, q, sig, T = 110, 100, 0.05, 0.02, 0.25, 0.5
    C = V.bs_price(S, K, r, q, sig, T, "call")
    P = V.bs_price(S, K, r, q, sig, T, "put")
    parity = S * math.exp(-q * T) - K * math.exp(-r * T)
    check("European put-call parity", abs((C - P) - parity) < 1e-9)

    # --- Greeks: call - put delta = e^(-qT) ---
    gc = V.bs_greeks(100, 100, 0.05, 0.02, 0.25, 1.0, "call")
    gp = V.bs_greeks(100, 100, 0.05, 0.02, 0.25, 1.0, "put")
    check("delta spread = discount factor",
          abs((gc["delta"] - gp["delta"]) - math.exp(-0.02)) < 1e-9)

    # --- IV round-trip + no-arb rejection ---
    px = V.bs_price(100, 100, 0.05, 0.02, 0.30, 0.5, "call")
    iv = V.implied_vol(px, 100, 100, 0.05, 0.02, 0.5, "call")
    check("IV round-trip 30%", abs(iv - 0.30) < 1e-4, f"{iv:.5f}")
    check("IV rejects price below bound",
          math.isnan(V.implied_vol(0, 100, 100, 0.05, 0.0, 1.0, "call")))
    check("IV zero at lower bound",
          V.implied_vol(0, 50, 200, 0.05, 0.0, 1.0, "call") == 0.0)

    # --- BS2002 paper anchors (Tables 1-3, +-0.01) ---
    check("BS2002 T1 call 4.69",
          abs(V.bjerksund_stensland(100, 100, 0.5, 0.08, 0.12, 0.20, "call") - 4.69) < 0.01)
    check("BS2002 T1 put 6.37",
          abs(V.bjerksund_stensland(100, 100, 0.5, 0.08, 0.12, 0.20, "put") - 6.37) < 0.01)
    check("BS2002 T2 call 6.50",
          abs(V.bjerksund_stensland(100, 100, 0.5, 0.08, 0.04, 0.20, "call") - 6.50) < 0.01)
    check("BS2002 T3 put 4.15",
          abs(V.bjerksund_stensland(100, 100, 0.5, 0.08, 0.0, 0.20, "put") - 4.15) < 0.01)

    # --- American dominance + q=0 call identity ---
    am_c = V.bjerksund_stensland(100, 100, 1.0, 0.05, 0.02, 0.25, "call")
    eu_c = V.bs_price(100, 100, 0.05, 0.02, 0.25, 1.0, "call")
    check("American call >= European", am_c >= eu_c - 1e-9)
    am0 = V.bjerksund_stensland(100, 100, 1.0, 0.05, 0.0, 0.25, "call")
    eu0 = V.bs_price(100, 100, 0.05, 0.0, 0.25, 1.0, "call")
    check("q=0 American call = European", abs(am0 - eu0) < 0.01)
    lo, hi = V.american_put_call_parity_bounds(100, 100, 0.05, 0.02, 1.0)
    am_p = V.bjerksund_stensland(100, 100, 1.0, 0.05, 0.02, 0.25, "put")
    check("American C-P within bounds", lo <= am_c - am_p <= hi)

    # --- Binomial lattice agrees with BS2002 ---
    lat = V.binomial_american(100, 100, 0.5, 0.08, 0.12, 0.20, "put", n=400)
    bs2002 = V.bjerksund_stensland(100, 100, 0.5, 0.08, 0.12, 0.20, "put")
    check("CRR lattice ~ BS2002 put", abs(lat - bs2002) < 0.05,
          f"lattice={lat:.4f} bs2002={bs2002:.4f}")

    # --- Bivariate normal: rho=0 factorization, M(0,0,0.5)=1/3 ---
    check("M factorizes at rho=0",
          abs(V._M(0.4, -0.2, 0.0) - V.N(0.4) * V.N(-0.2)) < 1e-12)
    check("M(0,0,0.5)=1/3", abs(V._M(0, 0, 0.5) - 1.0 / 3.0) < 1e-7)

    # --- EWMA closed form vs sequential recursion ---
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0003, 0.015, 252)
    got = V.ewma_vol_forecast(rets)
    var = float(np.var(rets))
    for x in rets:
        var = 0.94 * var + 0.06 * x * x
    check("EWMA matches recursion", abs(got - math.sqrt(var * 252)) < 1e-9,
          f"{got:.4f}")

    # --- Probability cone formula at t=21 ---
    days, upper, lower = probability_cone(100.0, 0.30, horizon_days=30)
    exp_u = 100.0 * math.exp(0.30 * math.sqrt(21 / 252))
    check("cone day-21 upper", abs(upper[21] - exp_u) < 1e-9)
    check("cone log-symmetric",
          bool(np.allclose(np.sqrt(upper * lower), 100.0, atol=1e-9)))

    print(f"\n{len(failures)} failure(s).")
    return 1 if failures else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    tokens = list(argv) if argv is not None else sys.argv[1:]
    machine = "--json" in tokens
    args = argparse.Namespace(json=machine)
    parser = build_parser(machine=machine)
    try:
        args = parser.parse_args(tokens)
        if args.command == "analyze":
            return cmd_analyze(args)
        if args.command == "scan":
            if args.max_expiries is not None and args.max_expiries <= 0:
                raise LocalModelError("INVALID_ARGUMENT", "--max-expiries must be positive.")
            if args.limit is not None and args.limit < 0:
                raise LocalModelError("INVALID_ARGUMENT", "--limit must be nonnegative.")
            parse_div_yield(args.div)
            return cmd_scan(args)
        if args.command == "verify":
            return cmd_verify(args)
        if args.command == "graph":
            from main.graph_cli import run as run_graph, render as render_graph
            render_graph(run_graph(args), args.json)
            return 0
        from main.model_cli import run, render
        render(run(args), args.json)
        return 0
    except (LocalModelError, argparse.ArgumentTypeError, OSError, RuntimeError, ValueError) as exc:
        code = getattr(exc, "code", "COMMAND_FAILED")
        if getattr(args, "json", False):
            print(json.dumps({"schema_version": 1, "status": "error",
                              "error": {"code": code, "message": str(exc)}}, allow_nan=False))
        else:
            print(f"Error [{code}]: {exc}", file=sys.stderr)
        return 2 if code in {"INVALID_ARGUMENT", "INVALID_PROFILE", "INVALID_PROMPT", "INVALID_HOST"} or isinstance(exc, argparse.ArgumentTypeError) else 4


if __name__ == "__main__":
    raise SystemExit(main())
