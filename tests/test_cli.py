"""CLI argparse / help coverage (no network)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main.cli import build_parser, main, parse_div_yield, cmd_verify
from types import SimpleNamespace


class TestCliArgparse:
    def test_help_exits_zero(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_analyze_requires_ticker(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["analyze"])

    def test_scan_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "scan", "LULU",
            "--under-only",
            "--max-expiries", "4",
            "--type", "call",
            "--garch",
            "--json",
        ])
        assert args.command == "scan"
        assert args.ticker == "LULU"
        assert args.under_only is True
        assert args.max_expiries == 4
        assert args.option_type == "call"
        assert args.garch is True
        assert args.json is True

    def test_analyze_json_flag(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "AMD", "--json"])
        assert args.command == "analyze"
        assert args.ticker == "AMD"
        assert args.json is True

    def test_main_rejects_unknown_command(self, monkeypatch):
        with pytest.raises(SystemExit):
            main(["nope"])

    def test_scan_extended_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "scan", "SPY",
            "--expiry", "2026-09-10",
            "--expiry", "2026-09-11",
            "--smile",
            "--euro-greeks",
            "--div", "0.98%",
            "--limit", "10",
            "--csv", "out.csv",
        ])
        assert args.expiry == ["2026-09-10", "2026-09-11"]
        assert args.smile is True
        assert args.euro_greeks is True
        assert args.limit == 10
        assert args.csv == "out.csv"

    def test_verify_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["verify"])
        assert args.command == "verify"


class TestParseDivYield:
    def test_percent_suffix(self):
        assert parse_div_yield("0.98%") == pytest.approx(0.0098)

    def test_decimal(self):
        assert parse_div_yield("0.0098") == pytest.approx(0.0098)

    def test_bare_number_is_decimal(self):
        # Bare values are decimal fractions; 0.98 = 98% is out of range,
        # nudging users toward the explicit "0.98%" form.
        import argparse
        assert parse_div_yield("0.0098") == pytest.approx(0.0098)
        with pytest.raises(argparse.ArgumentTypeError):
            parse_div_yield("0.98")

    def test_none(self):
        assert parse_div_yield(None) is None

    def test_rejects_garbage(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            parse_div_yield("abc")
        with pytest.raises(argparse.ArgumentTypeError):
            parse_div_yield("50%")


class TestVerifyCommand:
    def test_verify_passes_offline(self):
        rc = cmd_verify(SimpleNamespace())
        assert rc == 0
