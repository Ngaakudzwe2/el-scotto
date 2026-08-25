# -*- coding: utf-8 -*-
"""
market_data.py

Single entry point the replay/trading_robot layer calls to get candles,
regardless of which underlying source a symbol actually comes from. Keeps
strategy/memory code fully data-source-agnostic.

Also owns the per-symbol transaction cost model (spread, slippage,
commission) applied at execution in trading_robot.py. These are
ILLUSTRATIVE ASSUMPTIONS, not live-fetched real broker fee schedules --
documented per-field below. Override COST_PROFILES with your actual
broker's numbers before trusting any backtest for real money.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.candle import Candle
from src.data_binance import fetch_klines
from src.data_yfinance import fetch_yfinance_candles, YFINANCE_TICKERS
from src.data_deriv import fetch_deriv_candles, DERIV_SYMBOLS, DEFAULT_SPREAD

BINANCE_SYMBOLS = {"BTCUSDT"}
YFINANCE_SYMBOLS = set(YFINANCE_TICKERS)
DERIV_SYMBOL_NAMES = set(DERIV_SYMBOLS)
SUPPORTED_SYMBOLS = BINANCE_SYMBOLS | YFINANCE_SYMBOLS | DERIV_SYMBOL_NAMES


@dataclass
class CostProfile:
    spread: float          # round-trip spread, price units (half applied on each side, worsening the fill)
    slippage_pct: float    # extra adverse fill vs quote, as a fraction of price (models imperfect execution)
    commission_pct: float  # broker commission, as a fraction of notional, charged on EACH side (entry + exit)


# Illustrative retail-ish assumptions, not live broker data. Sources of the
# ballpark figures: typical retail forex spreads (~1-2 pips majors), typical
# gold spreads (~$0.30-0.50), Binance spot taker fee (0.10%). Replace with
# your actual broker's published spread/commission schedule before trusting
# any backtest for real capital.
COST_PROFILES = {
    "BTCUSDT":      CostProfile(spread=5.00,             slippage_pct=0.0005, commission_pct=0.0010),
    "GBPUSD":       CostProfile(spread=0.00015,          slippage_pct=0.0001, commission_pct=0.00005),
    "USDJPY":       CostProfile(spread=0.015,            slippage_pct=0.0001, commission_pct=0.00005),
    "XAUUSD":       CostProfile(spread=0.35,             slippage_pct=0.0001, commission_pct=0.00010),
    "XAUUSD_DERIV": CostProfile(spread=DEFAULT_SPREAD,   slippage_pct=0.0001, commission_pct=0.00010),
}
_NO_COST = CostProfile(spread=0.0, slippage_pct=0.0, commission_pct=0.0)


def cost_profile_for(symbol: str) -> CostProfile:
    return COST_PROFILES.get(symbol, _NO_COST)


def spread_for(symbol: str) -> float:
    """Kept for backward compatibility -- prefer cost_profile_for() for new code."""
    return cost_profile_for(symbol).spread


def source_for(symbol: str) -> str:
    if symbol in BINANCE_SYMBOLS:
        return "binance"
    if symbol in YFINANCE_SYMBOLS:
        return "yfinance"
    if symbol in DERIV_SYMBOL_NAMES:
        return "deriv"
    raise ValueError(f"Unsupported symbol: {symbol}. Supported: {sorted(SUPPORTED_SYMBOLS)}")


def default_interval_for(symbol: str) -> str:
    """Binance/Deriv default to 1h/1d respectively for recent windows;
    yfinance defaults to 1d since Yahoo doesn't retain multi-year
    intraday history."""
    if source_for(symbol) == "binance":
        return "1h"
    return "1d"


def fetch_candles(symbol: str, interval: Optional[str] = None, limit: int = 500,
                   start: Optional[str] = None, end: Optional[str] = None) -> List[Candle]:
    """
    Fetches real candles for `symbol`, dispatching to Binance, Yahoo
    Finance, or Deriv as appropriate. If `start` is given, pulls the full
    date range; otherwise pulls the `limit` most recent candles (Binance
    only -- yfinance/Deriv always fetch by date range and default to
    2018-01-01 if no start is given, though Deriv's actual history for
    XAUUSD_DERIV is capped to roughly the trailing year regardless -- see
    src/data_deriv.py).
    """
    source = source_for(symbol)
    interval = interval or default_interval_for(symbol)

    if source == "binance":
        return fetch_klines(symbol=symbol, interval=interval, limit=limit, start=start, end=end)
    if source == "deriv":
        return fetch_deriv_candles(symbol=symbol, interval=interval, start=start, end=end, limit=max(limit, 5000))

    return fetch_yfinance_candles(symbol=symbol, interval=interval, start=start or "2018-01-01", end=end)
