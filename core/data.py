"""Market data provider abstraction and yfinance implementation."""
from __future__ import annotations

import time
import threading
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import pandas as pd
import yfinance as yf


class DataProvider(ABC):
    """Abstract interface for market data fetches used by MarketApp."""

    @abstractmethod
    def create_ticker(self, symbol: str) -> Any:
        """Return a provider-specific ticker handle for ``symbol``."""

    @abstractmethod
    def fetch_history(
        self,
        ticker: Any,
        period: str,
        interval: str,
        retries: int = 2,
        delay: float = 1.0,
        log=None,
    ) -> pd.DataFrame:
        """Fetch OHLCV history with retries; raise on total failure."""

    @abstractmethod
    def get_info(self, ticker: Any) -> dict:
        """Return fundamental info dict for a ticker."""

    @abstractmethod
    def get_fast_last_price(self, ticker: Any) -> Optional[float]:
        """Return last price from a lightweight quote source, or None."""

    @abstractmethod
    def get_option_expirations(self, ticker: Any) -> tuple:
        """Return available option expiration date strings."""

    @abstractmethod
    def get_option_chain(self, ticker: Any, expiration: str) -> Any:
        """Return option chain for a single expiration."""

    @abstractmethod
    def fetch_rate_curve(self) -> Tuple[float, float]:
        """Return (short_rate, long_rate) as decimals."""

    @abstractmethod
    def get_calendar(self, ticker: Any) -> Any:
        """Return earnings calendar payload for a ticker."""

    @abstractmethod
    def get_fast_info(self, ticker: Any) -> Any:
        """Return the provider's fast_info object/mapping."""


class YFinanceProvider(DataProvider):
    """yfinance-backed DataProvider with optional OHLCV cache."""

    def __init__(self, cache_duration: float = 60.0):
        self.cache_duration = cache_duration
        self._data_cache = {}
        self._data_cache_lock = threading.Lock()

    def create_ticker(self, symbol: str) -> Any:
        return yf.Ticker(symbol)

    def fetch_history(
        self,
        ticker: Any,
        period: str,
        interval: str,
        retries: int = 2,
        delay: float = 1.0,
        log=None,
    ) -> pd.DataFrame:
        last_exc = None
        for attempt in range(retries + 1):
            try:
                df = ticker.history(period=period, interval=interval)
                if not df.empty:
                    return df
                last_exc = RuntimeError("Empty data returned")
            except Exception as e:
                last_exc = e
                if log:
                    log(f"History fetch error try {attempt+1}/{retries+1}: {e}")
            if attempt < retries:
                time.sleep(delay)
        if last_exc:
            raise last_exc
        raise RuntimeError("Unknown history fetch failure")

    def get_cached_df(self, symbol: str, period: str, interval: str):
        key = (symbol, period, interval)
        with self._data_cache_lock:
            if key in self._data_cache:
                data, ts = self._data_cache[key]
                if time.time() - ts < self.cache_duration:
                    return data, True
        return None, False

    def save_df_cache(self, symbol: str, period: str, interval: str, df: pd.DataFrame):
        key = (symbol, period, interval)
        with self._data_cache_lock:
            self._data_cache[key] = (df, time.time())

    def clear_cache(self):
        with self._data_cache_lock:
            self._data_cache.clear()

    def get_info(self, ticker: Any) -> dict:
        return ticker.info or {}

    def get_fast_last_price(self, ticker: Any) -> Optional[float]:
        try:
            return ticker.fast_info["last_price"]
        except Exception:
            return None

    def get_option_expirations(self, ticker: Any) -> tuple:
        return ticker.options

    def get_option_chain(self, ticker: Any, expiration: str) -> Any:
        return ticker.option_chain(expiration)

    def fetch_rate_curve(self) -> Tuple[float, float]:
        """Fetches ^IRX (short) and ^TNX (long) rates once."""
        try:
            short_rate = yf.Ticker("^IRX").fast_info["last_price"] / 100
            if short_rate <= 0:
                short_rate = 0.045
        except Exception:
            short_rate = 0.045
        try:
            long_rate = yf.Ticker("^TNX").fast_info["last_price"] / 100
            if long_rate <= 0:
                long_rate = short_rate
        except Exception:
            long_rate = short_rate
        return short_rate, long_rate

    def get_calendar(self, ticker: Any) -> Any:
        return ticker.calendar

    def get_fast_info(self, ticker: Any) -> Any:
        return ticker.fast_info
