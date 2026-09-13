"""US equity-option remaining session time (NYSE calendar proxy).

Volatility and rates use a common 252 x 6.5-hour trading-year clock, preserving
Sentinel's trading-day convention. This is not an ACT/365 rate conversion. Early
closes contribute only their actual session duration. AM-settled/index products
need a product-specific calendar/settlement adapter and are not supported here.
"""
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=64)
def _schedule(start, end, calendar_name):
    import pandas_market_calendars as calendars
    return calendars.get_calendar(calendar_name).schedule(start_date=start, end_date=end)


def remaining_years(expiry, valuation_time=None, *, calendar_name="NYSE"):
    """Remaining open-session seconds through expiry close; zero after expiry.

    Explicit valuation times must be timezone-aware. A non-session expiry uses
    the preceding session's close. All comparisons use UTC, independent of host TZ.
    """
    now = pd.Timestamp(valuation_time if valuation_time is not None else datetime.now(timezone.utc))
    if now.tzinfo is None:
        raise ValueError("valuation_time must be timezone-aware")
    now = now.tz_convert("UTC")
    start = now.tz_convert("America/New_York").date()
    end = pd.Timestamp(expiry).date()
    if end < start:
        return 0.0
    schedule = _schedule(start.isoformat(), end.isoformat(), calendar_name)
    seconds = sum(max(0.0, (row.market_close - max(now, row.market_open)).total_seconds())
                  for row in schedule.itertuples())
    return seconds / (252.0 * 6.5 * 3600.0)
