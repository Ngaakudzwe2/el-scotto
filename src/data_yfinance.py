# -*- coding: utf-8 -*-
"""
data_yfinance.py

Fetches real historical candles for non-Binance Forex/Commodity symbols
from Yahoo Finance (via the yfinance package). No generated or fixture
data is used -- every call hits Yahoo's live historical-data service.

Ticker mapping notes:
  GBPUSD -> "GBPUSD=X"  (Yahoo FX pair, full 2018-present daily history)
  USDJPY -> "USDJPY=X"  (Yahoo FX pair, full 2018-present daily history)
  XAUUSD -> "GC=F"      (COMEX Gold futures continuous contract -- Yahoo
                          has no direct XAUUSD=X spot ticker; GC=F is the
                          standard free-data proxy for spot gold. Prices
                          will differ from spot by the futures basis, but
                          the two track closely and this is the common
                          substitution used when spot gold data isn't
                          freely available.)

Interval note: Yahoo restricts intraday (sub-daily) history to short
lookback windows (~730 days for 1h, 7 days for 1m) regardless of how far
back `start` is. For a 2018-present request, only daily ("1d") data is
actually retrievable for the full range -- this module does not silently
truncate or fabricate data to fill the gap; it fetches whatever Yahoo
actually returns for the requested interval/range.
"""

from __future__ import annotations

from typing import List, Optional

import yfinance as yf

from src.candle import Candle

YFINANCE_TICKERS = {
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "XAUUSD": "GC=F",
}


def fetch_yfinance_candles(symbol: str, interval: str = "1d",
                            start: str = "2018-01-01", end: Optional[str] = None) -> List[Candle]:
    """Pull real historical candles for `symbol` (one of YFINANCE_TICKERS) from Yahoo Finance."""
    if symbol not in YFINANCE_TICKERS:
        raise ValueError(f"Unsupported yfinance symbol: {symbol}. Supported: {list(YFINANCE_TICKERS)}")

    ticker = YFINANCE_TICKERS[symbol]
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)

    if df.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {ticker} ({symbol}) interval={interval}")

    # yfinance returns MultiIndex columns (Field, Ticker) for single-ticker
    # downloads on recent versions -- flatten to plain column names.
    if isinstance(df.columns, type(df.columns)) and getattr(df.columns, "nlevels", 1) > 1:
        df.columns = df.columns.get_level_values(0)

    candles: List[Candle] = []
    for ts, row in df.iterrows():
        open_time_ms = int(ts.timestamp() * 1000)
        candles.append(Candle(
            open_time=open_time_ms,
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]) if "Volume" in row and row["Volume"] == row["Volume"] else 0.0,
        ))
    return candles
