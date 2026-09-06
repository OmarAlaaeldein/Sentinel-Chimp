"""Alternate CLI entry: ``python -m sentinel_cli`` / ``python sentinel_cli.py``."""
from main.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
