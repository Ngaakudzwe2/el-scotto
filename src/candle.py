# -*- coding: utf-8 -*-
"""
candle.py

Shared candle type used by every data source (Binance klines, yfinance)
and every strategy/replay module downstream, so they never need to know
which source a given symbol came from.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candle:
    open_time: int  # epoch millis, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
