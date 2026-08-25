# -*- coding: utf-8 -*-
"""
backtest_structures.py

Minimal, mechanical crossover strategy used by the replay backtester: a
fast/slow simple-moving-average crossover on closing price. Deliberately
simple -- this module exists to generate a clean, repeatable stream of
BUY/SELL signals for the memory system to be tested against, not to be a
sophisticated strategy in its own right.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from src.candle import Candle


class CrossDirection(Enum):
    GOLDEN = "golden_cross"  # fast SMA crosses above slow SMA -> bullish
    DEATH = "death_cross"    # fast SMA crosses below slow SMA -> bearish


@dataclass
class CrossSignal:
    index: int
    open_time: int
    price: float
    direction: CrossDirection

    @property
    def reason(self) -> str:
        if self.direction == CrossDirection.GOLDEN:
            return "Golden cross: fast SMA crossed above slow SMA"
        return "Death cross: fast SMA crossed below slow SMA"


def sma(values: List[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def detect_crossovers(candles: List[Candle], fast_period: int = 9, slow_period: int = 21) -> List[CrossSignal]:
    """Walk closes once, flag every point where the fast/slow SMA relationship flips sign."""
    closes = [c.close for c in candles]
    fast = sma(closes, fast_period)
    slow = sma(closes, slow_period)

    signals: List[CrossSignal] = []
    for i in range(1, len(candles)):
        if fast[i] is None or slow[i] is None or fast[i - 1] is None or slow[i - 1] is None:
            continue
        prev_diff = fast[i - 1] - slow[i - 1]
        curr_diff = fast[i] - slow[i]
        if prev_diff <= 0 and curr_diff > 0:
            signals.append(CrossSignal(index=i, open_time=candles[i].open_time, price=closes[i], direction=CrossDirection.GOLDEN))
        elif prev_diff >= 0 and curr_diff < 0:
            signals.append(CrossSignal(index=i, open_time=candles[i].open_time, price=closes[i], direction=CrossDirection.DEATH))
    return signals
