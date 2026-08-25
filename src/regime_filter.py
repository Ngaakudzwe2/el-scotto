# -*- coding: utf-8 -*-
"""
regime_filter.py

Binary volatility-regime gate: Can_Trade = True / False / None (warmup),
nothing else. No score, no weighting, no position-sizing input -- a
strategy either checks this before allowing an entry, or it doesn't.

This is the specific, concrete lever named as untested after three
separate strategy failures in this project (MEDFREQ's whipsaw diagnosis,
entries_v2's flat-to-negative walk-forward result, and breakout_v1's
55.66% training drawdown all trace to the same root cause: nothing in
any of those designs distinguishes a trending market from a ranging one
before entering). This module tests that lever directly, in isolation,
against the one strategy (entries_v2) with an already-established,
already-walk-forward-validated baseline to compare against.

Mechanism: ATR(14) expansion relative to its own trailing SMA(50) --
i.e. "is current volatility above its own recent average?" Chosen over
the ADX(14) alternative for two concrete reasons, not a coin flip:
  1. Reuses src.indicators.atr(), already tested and already the SL/TP
     basis for entries_v2/v3 -- no new indicator implementation surface
     (ADX requires a full Wilder DI+/DI-/ADX calculation from scratch,
     more code, more places for a subtle bug to hide).
  2. This is the exact mechanism already named in this project's own
     prior recommendations (data/performance_report.md's "what to test
     instead" sections, both after entries_v3 and after breakout_v1) --
     using it now is following through on stated reasoning, not
     cherry-picking after the fact.
Deliberately NOT implementing both and picking whichever backtests
better -- that would itself be a form of overfitting via selection
among design choices, exactly what this module exists to avoid.

Both parameters (ATR period 14, SMA-of-ATR period 50) are fixed,
standard conventions -- 14 is the ATR default used everywhere else in
this project; 50 matches the "50-bar" convention already used elsewhere
(e.g. entries_v3's daily trend SMA). Neither is searched or tuned here.
"""

from __future__ import annotations

from typing import List, Optional

from src.candle import Candle
from src.indicators import atr

ATR_PERIOD = 14
ATR_SMA_PERIOD = 50


def _rolling_sma_skip_none(values: List[Optional[float]], period: int) -> List[Optional[float]]:
    """Rolling SMA over a series that starts with None values (e.g. an indicator's own
    warmup) -- only starts producing output once `period` real values have accumulated,
    never treats a None as zero or skips it silently in a way that shifts the window."""
    out: List[Optional[float]] = [None] * len(values)
    window: List[float] = []
    for i, v in enumerate(values):
        if v is None:
            continue
        window.append(v)
        if len(window) > period:
            window.pop(0)
        if len(window) == period:
            out[i] = sum(window) / period
    return out


def atr_expansion_gate(candles: List[Candle], atr_period: int = ATR_PERIOD,
                        sma_period: int = ATR_SMA_PERIOD) -> List[Optional[bool]]:
    """
    Per-bar Can_Trade series, aligned 1:1 with `candles`:
      True  -- ATR(atr_period) is above its own trailing SMA(sma_period): volatility
               expanding, treated as a trending/tradeable regime.
      False -- ATR at or below its own trailing average: treated as chop, stand aside.
      None  -- insufficient history yet (ATR warmup + SMA-of-ATR warmup not complete).
    Purely a function of past and current bars -- no lookahead by construction, since
    both atr() and the rolling SMA above only ever look backward from index i.
    """
    atr_vals = atr(candles, atr_period)
    atr_sma = _rolling_sma_skip_none(atr_vals, sma_period)
    gate: List[Optional[bool]] = []
    for a, s in zip(atr_vals, atr_sma):
        if a is None or s is None:
            gate.append(None)
        else:
            gate.append(a > s)
    return gate
