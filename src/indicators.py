# -*- coding: utf-8 -*-
"""
indicators.py

Minimal, dependency-free technical indicators used by the medfreq
strategy. Kept separate from the root-level structures.py (the
unconnected SMC engine) so src/ stays self-contained.
"""

from __future__ import annotations

from typing import List, Optional

from src.candle import Candle


def ema(values: List[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if len(values) < period:
        return out
    k = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def rsi(values: List[float], period: int = 14) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if len(values) < period + 1:
        return out
    gains = [max(values[i] - values[i - 1], 0.0) for i in range(1, len(values))]
    losses = [max(values[i - 1] - values[i], 0.0) for i in range(1, len(values))]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out[period] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs_val = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
        out[i + 1] = rs_val
    return out


def atr(candles: List[Candle], period: int = 14) -> List[Optional[float]]:
    n = len(candles)
    out: List[Optional[float]] = [None] * n
    if n < period + 1:
        return out
    trs = []
    for i in range(1, n):
        h, l, prev_c = candles[i].high, candles[i].low, candles[i - 1].close
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    avg = sum(trs[:period]) / period
    out[period] = avg
    for i in range(period, len(trs)):
        avg = (avg * (period - 1) + trs[i]) / period
        out[i + 1] = avg
    return out
