"""Headless CLI for Sentinel-Chimp (no tkinter).

Usage examples:
  python -m main.cli analyze LULU
  python -m main.cli scan LULU --under-only --max-expiries 6
  python sentinel.py scan LULU --json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from core.data import YFinanceProvider
from core.scan_service import analyze_ticker, run_ticker_scan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
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
    p_scan.add_argument("--json", action="store_true", help="Emit JSON instead of text")

    return parser


def _print_analysis(analysis, as_json: bool) -> None:
    if as_json:
        print(json.dumps(analysis.to_dict(), indent=2, default=str))
        return
    for line in analysis.summary_lines:
        print(line)


def _print_scan(analysis, result, *, under_only: bool, as_json: bool) -> None:
    if as_json:
        payload = {
            "analysis": analysis.to_dict(),
            "forecast_vol": result.forecast_vol,
            "dividend_yield": result.dividend_yield,
            "rules": result.rules_log,
            "count": len(result.rows),
            "rows": [r.to_dict() for r in result.rows],
        }
        print(json.dumps(payload, indent=2, default=str))
        return

    for line in analysis.summary_lines:
        print(line)
    print(
        f"Forecast vol={result.forecast_vol:.1%}  "
        f"div={result.dividend_yield:.2%}  "
        f"contracts={len(result.rows)}"
        + ("  (Under only, ranked by edge %)" if under_only else "")
    )
    print(result.rules_log)
    if not result.rows:
        print("No contracts matched scan filters.")
        return

    header = (
        f"{'Expiry':<12} {'Type':<5} {'Strike':>8} {'Mid':>7} {'Fair':>7} "
        f"{'EV@Ask':>8} {'Edge%':>7} {'Delta':>7} {'Verdict'}"
    )
    print(header)
    print("-" * len(header))
    for r in result.rows:
        print(
            f"{r.date:<12} {r.type:<5} {r.strike:8.2f} {r.mid:7.2f} {r.fair:7.2f} "
            f"{r.ev_at_ask:+8.2f} {r.edge_pct * 100:6.1f}% {r.delta:7.3f} {r.verdict}"
        )


def cmd_analyze(args: argparse.Namespace) -> int:
    provider = YFinanceProvider()
    analysis = analyze_ticker(provider, args.ticker)
    _print_analysis(analysis, args.json)
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    provider = YFinanceProvider()

    def _log(msg: str) -> None:
        if not args.json:
            print(msg, file=sys.stderr)

    analysis, result = run_ticker_scan(
        provider,
        args.ticker,
        under_only=args.under_only,
        max_expiries=args.max_expiries,
        option_type=args.option_type,
        use_garch_blend=args.garch,
        log=_log,
    )
    _print_scan(analysis, result, under_only=args.under_only, as_json=args.json)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "scan":
        return cmd_scan(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
