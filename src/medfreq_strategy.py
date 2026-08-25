# -*- coding: utf-8 -*-
"""
medfreq_strategy.py

XAUUSD_MEDFREQ bot -- Top-Down Multi-Timeframe (MTF) model.

  H4 chart  -> macro trend filter: 200 EMA (price above = bullish regime,
               below = bearish regime).
  H1 chart  -> momentum filter: RSI(14), same long/short bands as before.
  M5 chart  -> execution: EMA(8/21) crossover triggers the entry, but
               ONLY when the crossover direction agrees with both the H4
               trend and the H1 RSI band. ATR-based stop-loss/take-profit
               use the M5 ATR (tighter stops, matching the execution
               timeframe's own volatility) rather than H1 ATR.

H4/H1 indicators are computed on their own resampled candles (see
src/data_dukascopy.py's resample()), then forward-filled onto the M5
timeline using ONLY the most recently CLOSED higher-timeframe bar as of
each M5 bar's open time -- align_htf_to_m5() ensures an M5 bar never sees
a still-forming H1/H4 bar's value. This is the standard, correct way to
combine timeframes without lookahead bias; getting this wrong (e.g. using
the still-forming H4 bar's live EMA) would silently leak future
information into every entry decision.

Real transaction costs applied at the M5 fill (spread/slippage/commission
from market_data.cost_profile_for), consistent with every other module in
this project. This module does not yet integrate the two-file memory
system (built around the lowfreq bot's long-only, opposite-signal-exit
model) -- raw simulation only, a deliberate v1 scope.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from src.candle import Candle
from src.indicators import ema, rsi, atr
from src import market_data
from src.data_dukascopy import resample

TRADING_DAYS_PER_YEAR = 252


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class MedFreqConfig:
    # Execution timeframe (M5)
    ema_fast: int = 8
    ema_slow: int = 21
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.75  # midpoint of the requested 2.5x-3.0x range
    # Higher-timeframe filters
    h4_trend_ema: int = 200
    h1_rsi_period: int = 14
    rsi_long: tuple = (40.0, 65.0)
    rsi_short: tuple = (35.0, 60.0)
    # Anti-whipsaw filters -- added after the first full-history run showed
    # ~1,120 trades/year (vs. a 50-75 target) with catastrophic drawdown,
    # traced to the M5 EMA(8,21) firing on pure noise during choppy periods
    # and immediately re-entering the same direction seconds after being
    # stopped out (see data/performance_report.md for the diagnosed
    # example: 5 same-direction re-entries in 6 hours on 2024-02-21).
    # Neither value was tuned to hit the trade-count target -- they're
    # standard, independently-justified noise filters:
    confirm_bars: int = 3   # crossover must hold for 3 consecutive M5 bars (15min) before it's treated as real
    cooldown_bars: int = 24  # 2 hours must pass after ANY exit before a new entry is considered


@dataclass
class TradeRecord:
    direction: Direction
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str  # "SL" | "TP"
    qty: float
    pnl: float


def align_htf_to_m5(m5_candles: List[Candle], htf_candles: List[Candle],
                     htf_values: List[Optional[float]], htf_minutes: int) -> List[Optional[float]]:
    """
    For each M5 candle, returns the higher-timeframe indicator value from
    the most recently CLOSED htf bar (close time <= the M5 bar's open
    time) -- never the still-forming htf bar covering that same moment.
    None until the first htf bar has actually closed.
    """
    htf_duration_ms = htf_minutes * 60_000
    aligned: List[Optional[float]] = [None] * len(m5_candles)
    j = 0
    last_closed_value: Optional[float] = None
    for i, c in enumerate(m5_candles):
        while j < len(htf_candles) and htf_candles[j].open_time + htf_duration_ms <= c.open_time:
            last_closed_value = htf_values[j]
            j += 1
        aligned[i] = last_closed_value
    return aligned


def simulate(m5_candles: List[Candle], config: MedFreqConfig, symbol: str,
             notional: float = 10_000.0) -> List[TradeRecord]:
    """Pure in-memory bar-by-bar MTF simulation, M5 execution. No ledger I/O."""
    h4_candles = resample(m5_candles, 240)
    h1_candles = resample(m5_candles, 60)
    h4_ema200 = ema([c.close for c in h4_candles], config.h4_trend_ema)
    h1_rsi = rsi([c.close for c in h1_candles], config.h1_rsi_period)

    trend_on_m5 = align_htf_to_m5(m5_candles, h4_candles, h4_ema200, 240)
    rsi_on_m5 = align_htf_to_m5(m5_candles, h1_candles, h1_rsi, 60)

    closes = [c.close for c in m5_candles]
    fast = ema(closes, config.ema_fast)
    slow = ema(closes, config.ema_slow)
    atr_vals = atr(m5_candles, config.atr_period)
    costs = market_data.cost_profile_for(symbol)

    trades: List[TradeRecord] = []
    position: Optional[dict] = None
    same_side_streak = 0     # consecutive bars fast has been on its current side of slow
    prev_side = 0            # +1 fast>slow, -1 fast<slow, 0 unknown
    bars_since_exit = config.cooldown_bars  # no cooldown restriction before the first exit

    for i in range(1, len(m5_candles)):
        if None in (fast[i], slow[i], fast[i - 1], slow[i - 1], trend_on_m5[i], rsi_on_m5[i], atr_vals[i]):
            continue

        c = m5_candles[i]
        t = datetime.fromtimestamp(c.open_time / 1000, tz=timezone.utc)

        curr_side = 1 if fast[i] > slow[i] else (-1 if fast[i] < slow[i] else 0)
        same_side_streak = same_side_streak + 1 if curr_side == prev_side else 1
        prev_side = curr_side
        if bars_since_exit < config.cooldown_bars:
            bars_since_exit += 1

        if position is not None:
            exit_price = exit_reason = None
            if position["direction"] == Direction.LONG:
                if c.low <= position["sl"]:
                    exit_price, exit_reason = position["sl"], "SL"
                elif c.high >= position["tp"]:
                    exit_price, exit_reason = position["tp"], "TP"
            else:
                if c.high >= position["sl"]:
                    exit_price, exit_reason = position["sl"], "SL"
                elif c.low <= position["tp"]:
                    exit_price, exit_reason = position["tp"], "TP"

            if exit_price is not None:
                spread_adj = costs.spread / 2 + exit_price * costs.slippage_pct
                fill = (exit_price - spread_adj) if position["direction"] == Direction.LONG else (exit_price + spread_adj)
                qty = position["qty"]
                gross = ((fill - position["entry_price"]) if position["direction"] == Direction.LONG
                         else (position["entry_price"] - fill)) * qty
                exit_commission = fill * qty * costs.commission_pct
                pnl = gross - position["entry_commission"] - exit_commission
                trades.append(TradeRecord(
                    direction=position["direction"], entry_time=position["entry_time"],
                    entry_price=position["entry_price"], exit_time=t, exit_price=fill,
                    exit_reason=exit_reason, qty=qty, pnl=pnl,
                ))
                position = None
                bars_since_exit = 0

        if position is not None:
            continue
        if bars_since_exit < config.cooldown_bars:
            continue

        # Confirmed crossover: fast has been on its current side for exactly
        # confirm_bars bars (fires once per crossover, not every bar of a
        # persistent trend) -- filters the single-bar noise flips that were
        # driving the whipsaw losses (see MedFreqConfig docstring).
        golden = curr_side == 1 and same_side_streak == config.confirm_bars
        death = curr_side == -1 and same_side_streak == config.confirm_bars
        price = c.close
        a = atr_vals[i]
        r = rsi_on_m5[i]
        h4_trend = trend_on_m5[i]

        direction = None
        if golden and price > h4_trend and config.rsi_long[0] <= r <= config.rsi_long[1]:
            direction = Direction.LONG
        elif death and price < h4_trend and config.rsi_short[0] <= r <= config.rsi_short[1]:
            direction = Direction.SHORT

        if direction is None or not a or a <= 0:
            continue

        spread_adj = costs.spread / 2 + price * costs.slippage_pct
        fill = (price + spread_adj) if direction == Direction.LONG else (price - spread_adj)
        qty = notional / fill
        entry_commission = notional * costs.commission_pct
        sl = (fill - config.atr_sl_mult * a) if direction == Direction.LONG else (fill + config.atr_sl_mult * a)
        tp = (fill + config.atr_tp_mult * a) if direction == Direction.LONG else (fill - config.atr_tp_mult * a)

        position = {"direction": direction, "entry_price": fill, "sl": sl, "tp": tp,
                    "entry_time": t, "qty": qty, "entry_commission": entry_commission}

    return trades


def compute_metrics(trades: List[TradeRecord], notional: float) -> dict:
    pnls = [t.pnl for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    win_rate = wins / n * 100 if n else 0.0
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    equity = notional
    peak = notional
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100)

    sharpe = sortino = None
    trades_per_year = None
    if n >= 5:
        times = sorted(t.exit_time for t in trades)
        years = (times[-1] - times[0]).total_seconds() / (365.25 * 24 * 3600)
        if years > 0:
            trades_per_year = n / years
            returns = [p / notional for p in pnls]
            mean_r = statistics.mean(returns)
            stdev_r = statistics.pstdev(returns)
            if stdev_r > 0:
                sharpe = mean_r / stdev_r * math.sqrt(trades_per_year)
            downside_dev = (statistics.mean(min(r, 0.0) ** 2 for r in returns)) ** 0.5
            if downside_dev > 0:
                sortino = mean_r / downside_dev * math.sqrt(trades_per_year)

    return {
        "n_trades": n, "wins": wins, "losses": losses, "win_rate_pct": win_rate,
        "profit_factor": profit_factor, "max_drawdown_pct": max_dd, "sharpe": sharpe, "sortino": sortino,
        "net_pnl": sum(pnls), "net_pnl_pct": sum(pnls) / notional * 100 if notional else 0.0,
        "trades_per_year": trades_per_year,
    }
