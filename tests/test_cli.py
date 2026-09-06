"""CLI argparse / help coverage (no network)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main.cli import build_parser, main


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
