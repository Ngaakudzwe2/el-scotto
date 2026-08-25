# -*- coding: utf-8 -*-
"""
data_binance.py

Fetches real historical candles from Binance's public klines REST endpoint.
No API key is required -- this is a public market-data endpoint. Every
replay run pulls a live candle set; no generated or fixture data is used
anywhere in this module.

Endpoint: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
"""

from __future__ import annotations

from typing import List, Optional

import requests

from src.candle import Candle

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500,
                  start: Optional[str] = None, end: Optional[str] = None) -> List[Candle]:
    """
    Pull candles for `symbol`/`interval` from Binance. If `start`/`end`
    (ISO date strings, e.g. "2018-01-01") are given, paginates through the
    full date range (Binance caps each request at 1000 candles); otherwise
    returns the `limit` most recent candles.
    """
    if start is None:
        return _fetch_recent(symbol, interval, limit)
    return _fetch_range(symbol, interval, start, end)


def _fetch_recent(symbol: str, interval: str, limit: int) -> List[Candle]:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
    resp.raise_for_status()
    return _rows_to_candles(resp.json())


def _fetch_range(symbol: str, interval: str, start: str, end: Optional[str]) -> List[Candle]:
    import datetime as _dt

    start_ms = int(_dt.datetime.fromisoformat(start).replace(tzinfo=_dt.timezone.utc).timestamp() * 1000)
    end_ms = (
        int(_dt.datetime.fromisoformat(end).replace(tzinfo=_dt.timezone.utc).timestamp() * 1000)
        if end else int(_dt.datetime.now(_dt.timezone.utc).timestamp() * 1000)
    )

    all_candles: List[Candle] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": 1000}
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_candles.extend(_rows_to_candles(rows))
        cursor = rows[-1][0] + 1  # next candle after the last one returned
        if len(rows) < 1000:
            break
    return all_candles


def _rows_to_candles(rows: list) -> List[Candle]:
    return [
        Candle(
            open_time=row[0],
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )
        for row in rows
    ]
