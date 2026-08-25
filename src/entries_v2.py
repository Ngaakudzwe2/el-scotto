# -*- coding: utf-8 -*-
"""
entries_v2.py

XAUUSD_LOWFREQ v2 -- adds an intraday entry layer to reach the 3-4
trades/week target the pure daily SMA(7,50) crossover structurally
cannot hit (~0.13 trades/week measured directly from data/ledger.csv --
see data/pre_retrain_snapshot.md). Does NOT touch or import
src/backtest_structures.py / src/backtest_entries.py / src/trading_robot.py
-- those remain exactly as they were; this is a new, separate module.

Design (one setup, not a committee, per the retraining brief):

  Daily chart  -> REGIME FILTER ONLY: price above its own daily SMA
                  (trend_sma_period) = long-only regime; below = short-
                  only regime. This is the ONLY use of the daily
                  timeframe in v2 -- it never times entries.
  H1 chart     -> ENTRY: a single pullback-to-EMA-and-bounce setup.
                  In a long regime, a bar whose LOW touches within
                  `pullback_tolerance_pct` of the H1 EMA
                  (pullback_ema_period) but CLOSES back above it
                  triggers a LONG. Mirror image for a short regime.
                  This is a real, standard trend-continuation entry
                  (buy the dip in an uptrend / sell the rip in a
                  downtrend), not a curve-fit invention.
  Exit         -> ATR-based stop-loss and take-profit (H1 ATR), fixed
                  at entry. This REPLACES the old "exit on opposite
                  crossover" model entirely -- the old LOWFREQ has no
                  stop-loss at all (see snapshot), which is not
                  something to carry forward into a higher-frequency,
                  short-permitting version.

Daily -> H1 alignment reuses medfreq_strategy.align_htf_to_m5's exact
no-lookahead logic (a daily SMA value is only visible to an H1 bar once
that daily bar has actually closed) rather than reimplementing it --
that logic is already correct and tested, and re-deriving it here would
just be a second place to introduce a lookahead bug.

Parameter budget: exactly 5 tunable parameters (trend_sma_period,
pullback_ema_period, pullback_tolerance_pct, atr_sl_mult, atr_tp_mult).
atr_period is fixed at 14 (the standard default used everywhere else in
this project, e.g. medfreq_strategy.py) and is NOT tuned -- deliberately
kept out of the search to hold the budget at 5, not 6.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from src.candle import Candle
from src.indicators import ema, atr
from src.backtest_structures import sma
from src.medfreq_strategy import align_htf_to_m5
from src.regime_filter import atr_expansion_gate

ATR_PERIOD = 14  # fixed, not tunable -- see module docstring


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class LowfreqV2Config:
    trend_sma_period: int = 100        # daily SMA regime filter
    pullback_ema_period: int = 21      # H1 EMA pullback reference
    pullback_tolerance_pct: float = 0.15  # how close (%) price must get to the EMA to count as a touch
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.5
    # Binary on/off (src.regime_filter.atr_expansion_gate) -- NOT a 6th tunable parameter in
    # the search-range sense; this is a toggle for a controlled A/B comparison against the
    # already-established baseline, not something optimized over a range. Defaults False so
    # every existing caller (the live paper-trading harness included) is completely unaffected
    # unless this is explicitly turned on.
    use_regime_filter: bool = False
    # How many CONSECUTIVE bars the gate must have already held True before an entry is
    # allowed -- targets the specific failure mode diagnosed in the last round (the noisy
    # edge of a chop/expansion transition firing the gate on a single bar that reverts
    # immediately after). Fixed at 3, not searched -- reuses the exact convention already
    # independently justified for medfreq_strategy.py's confirm_bars anti-whipsaw filter,
    # not a new number invented for this round. 1 (the default) means "no persistence
    # requirement," i.e. identical to the prior round's behavior.
    regime_confirm_bars: int = 1

    def as_dict(self) -> dict:
        return {
            "trend_sma_period": self.trend_sma_period,
            "pullback_ema_period": self.pullback_ema_period,
            "pullback_tolerance_pct": self.pullback_tolerance_pct,
            "atr_sl_mult": self.atr_sl_mult,
            "atr_tp_mult": self.atr_tp_mult,
            "use_regime_filter": self.use_regime_filter,
            "regime_confirm_bars": self.regime_confirm_bars,
        }


@dataclass
class TradeRecordV2:
    direction: Direction
    entry_time: datetime
    entry_price: float
    stop: float   # SL price level set at entry (fixed for the life of the trade)
    target: float  # TP price level set at entry (fixed for the life of the trade)
    exit_time: datetime
    exit_price: float
    exit_reason: str  # "SL" | "TP"
    qty: float
    pnl: float


@dataclass
class CostModel:
    """
    Deriv-realistic costs for this exercise, per the retraining brief --
    deliberately NOT market_data.COST_PROFILES["XAUUSD_DERIV"] (that one
    uses a percentage-based slippage model and a non-zero commission;
    this brief specifies a fixed-dollar spread/slippage and zero
    commission). Kept local to this module rather than overwriting the
    shared cost profile other bots (LOWFREQ v1, MEDFREQ) depend on.
    """
    spread: float = 0.40          # USD round-trip, midpoint of the stated 35-50 cent range
    slippage_per_side: float = 0.05  # USD, fixed, applied on both entry and exit
    commission: float = 0.0        # "no separate commission" per the brief


DEFAULT_COSTS = CostModel()


def simulate(h1_candles: List[Candle], daily_candles: List[Candle], config: LowfreqV2Config,
             notional: float = 10_000.0, costs: CostModel = DEFAULT_COSTS) -> List[TradeRecordV2]:
    """Pure in-memory bar-by-bar simulation, H1 execution, daily regime filter. No ledger I/O."""
    daily_closes = [c.close for c in daily_candles]
    daily_sma = sma(daily_closes, config.trend_sma_period)
    trend_on_h1 = align_htf_to_m5(h1_candles, daily_candles, daily_sma, 1440)

    h1_closes = [c.close for c in h1_candles]
    h1_ema = ema(h1_closes, config.pullback_ema_period)
    atr_vals = atr(h1_candles, ATR_PERIOD)
    tol = config.pullback_tolerance_pct / 100.0
    # Computed unconditionally (cheap, single pass) but only ever consulted below when
    # use_regime_filter is True -- entry/exit behavior is unchanged from before this
    # integration whenever the flag is left at its default False.
    regime_gate = atr_expansion_gate(h1_candles)
    # Precomputed as its own series (not a counter mutated inside the main loop below) --
    # the main loop has several `continue` statements before reaching the entry check, and
    # a counter that only updates on iterations that reach that point would silently miss
    # bars and desync from the true consecutive-True count. This has no such dependency on
    # loop control flow: it's purely a function of regime_gate itself, computed once.
    regime_streak: List[int] = []
    _streak = 0
    for _g in regime_gate:
        _streak = _streak + 1 if _g is True else 0
        regime_streak.append(_streak)

    trades: List[TradeRecordV2] = []
    position: Optional[dict] = None

    for i in range(1, len(h1_candles)):
        if None in (trend_on_h1[i], h1_ema[i], atr_vals[i]) or atr_vals[i] <= 0:
            continue
        c = h1_candles[i]
        t = datetime.fromtimestamp(c.open_time / 1000, tz=timezone.utc)
        trend_sma = trend_on_h1[i]
        e = h1_ema[i]
        a = atr_vals[i]

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
                fill = (exit_price - costs.slippage_per_side) if position["direction"] == Direction.LONG \
                    else (exit_price + costs.slippage_per_side)
                qty = position["qty"]
                gross = ((fill - position["entry_price"]) if position["direction"] == Direction.LONG
                         else (position["entry_price"] - fill)) * qty
                pnl = gross - costs.commission * qty * 2  # commission is 0.0 per the brief; kept for completeness
                trades.append(TradeRecordV2(
                    direction=position["direction"], entry_time=position["entry_time"],
                    entry_price=position["entry_price"], stop=position["sl"], target=position["tp"],
                    exit_time=t, exit_price=fill, exit_reason=exit_reason, qty=qty, pnl=pnl,
                ))
                position = None

        if position is not None:
            continue

        # Binary Can_Trade gate -- entries only, never blocks managing/closing a position
        # already open (that check happens above, before this point). No effect at all
        # unless explicitly enabled. regime_streak[i] is 0 whenever regime_gate[i] isn't
        # exactly True (including the None-during-warmup case), so requiring
        # regime_streak[i] >= regime_confirm_bars naturally subsumes the old direct
        # "is not True" check -- regime_confirm_bars=1 (the default) reproduces the prior
        # round's exact behavior with no persistence requirement.
        if config.use_regime_filter and regime_streak[i] < config.regime_confirm_bars:
            continue

        price = c.close
        long_regime = price > trend_sma
        short_regime = price < trend_sma

        # Single pullback-to-EMA-and-bounce setup, mirrored for both regimes.
        long_trigger = long_regime and c.low <= e * (1 + tol) and c.close > e
        short_trigger = short_regime and c.high >= e * (1 - tol) and c.close < e

        direction = Direction.LONG if long_trigger else (Direction.SHORT if short_trigger else None)
        if direction is None:
            continue

        spread_adj = costs.spread / 2 + costs.slippage_per_side
        fill = (price + spread_adj) if direction == Direction.LONG else (price - spread_adj)
        qty = notional / fill
        sl = (fill - config.atr_sl_mult * a) if direction == Direction.LONG else (fill + config.atr_sl_mult * a)
        tp = (fill + config.atr_tp_mult * a) if direction == Direction.LONG else (fill - config.atr_tp_mult * a)

        position = {"direction": direction, "entry_price": fill, "sl": sl, "tp": tp, "entry_time": t, "qty": qty}

    return trades


def compute_metrics(trades: List[TradeRecordV2], notional: float, risk_free_annual: float = 0.05) -> dict:
    pnls = [t.pnl for t in trades]
    n = len(pnls)
    wins_list = [p for p in pnls if p > 0]
    losses_list = [p for p in pnls if p < 0]
    wins, losses = len(wins_list), len(losses_list)
    win_rate = wins / n * 100 if n else 0.0
    avg_win = statistics.mean(wins_list) if wins_list else 0.0
    avg_loss = statistics.mean(losses_list) if losses_list else 0.0
    gross_profit = sum(wins_list)
    gross_loss = abs(sum(losses_list))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    expectancy = statistics.mean(pnls) if pnls else 0.0

    equity = notional
    peak = notional
    peak_time = None
    max_dd = 0.0
    max_dd_dollars = 0.0
    max_dd_duration_days = 0.0
    dd_start_time = None
    times = [t.exit_time for t in trades]
    for p, exit_time in zip(pnls, times):
        equity += p
        if equity >= peak:
            peak = equity
            peak_time = exit_time
        else:
            dd_pct = (peak - equity) / peak * 100 if peak > 0 else 0.0
            dd_dollars = peak - equity
            if dd_pct > max_dd:
                max_dd = dd_pct
                max_dd_dollars = dd_dollars
                if peak_time is not None:
                    max_dd_duration_days = (exit_time - peak_time).total_seconds() / 86400

    sharpe = sortino = None
    trades_per_week = None
    if n >= 2:
        span_days = (times[-1] - times[0]).total_seconds() / 86400
        years = span_days / 365.25
        if years > 0:
            trades_per_year = n / years
            trades_per_week = trades_per_year / 52.1775
            returns = [p / notional for p in pnls]
            mean_r = statistics.mean(returns)
            stdev_r = statistics.pstdev(returns)
            rf_per_trade = risk_free_annual / trades_per_year if trades_per_year > 0 else 0.0
            if stdev_r > 0:
                sharpe = (mean_r - rf_per_trade) / stdev_r * math.sqrt(trades_per_year)
            downside_dev = (statistics.mean(min(r - rf_per_trade, 0.0) ** 2 for r in returns)) ** 0.5
            if downside_dev > 0:
                sortino = (mean_r - rf_per_trade) / downside_dev * math.sqrt(trades_per_year)

    return {
        "n_trades": n, "wins": wins, "losses": losses, "win_rate_pct": win_rate,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "avg_win_loss_ratio": (avg_win / abs(avg_loss)) if avg_loss != 0 else None,
        "profit_factor": profit_factor, "expectancy": expectancy,
        "max_drawdown_pct": max_dd, "max_drawdown_dollars": max_dd_dollars,
        "max_drawdown_duration_days": max_dd_duration_days,
        "sharpe": sharpe, "sortino": sortino,
        "net_pnl": sum(pnls), "net_pnl_pct": sum(pnls) / notional * 100 if notional else 0.0,
        "trades_per_week": trades_per_week,
    }
