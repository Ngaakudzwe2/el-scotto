# -*- coding: utf-8 -*-
"""
entries_v2_paper.py

Live paper-trading harness for XAUUSD_LOWFREQ v2 (src/entries_v2.py).
Read-only with respect to trading -- simulates fills against real,
live-fetched prices and logs them; places no real orders, needs no
broker authentication.

Data source is Deriv (XAUUSD_DERIV), NOT Dukascopy -- Dukascopy has no
live/current-price capability (it's an hourly historical archive fetched
by explicit date/hour, used for backtesting only; see
src/data_dukascopy.py and MT5_SETUP.md for why). This means paper-trade
fills come from a genuinely different feed (Deriv CFD pricing) than the
Dukascopy-sourced backtest this strategy was trained/tested against --
a real, honest difference worth remembering when comparing forward-test
results to the backtest, not an identical replay.

Mechanism: every poll, fetches the most recent H1 and daily candles from
Deriv and re-runs entries_v2.simulate() over that window (same function
the backtest uses -- no separate hand-written "live" entry logic to
drift out of sync with the tested one). Any trade in the result that
wasn't in the previous poll's result is newly closed and gets appended
to the log. Because simulate() only returns CLOSED trades, a row is
written once a trade's stop or target is actually hit -- there's no
"still open" row.

Logs to data/entries_v2_paper_trades.csv -- deliberately NOT
data/paper_trades.csv (that file uses a different schema with no
stop/target columns and is actively written by src/live_monitor.py's
v1 crossover paper-trading process; writing here too would corrupt
both and race with that already-running process).

Default parameters are the walk-forward-selected combo from the
retraining round (trend_sma_period=50, pullback_ema_period=21,
pullback_tolerance_pct=0.20, atr_sl_mult=2.0, atr_tp_mult=2.5), PLUS the
binary ATR-expansion regime filter with a 3-bar persistence requirement
(use_regime_filter=True, regime_confirm_bars=3) -- the strongest, most
rigorously-surviving candidate this project has produced (holdout
Sharpe 1.843, p=0.033 significant, Monte Carlo 1.42% of paths losing --
see data/performance_report.md's "N-Bar Persistence Filter" section for
the full result and the honest caveat: training-period Sharpe was
still -0.519, so this is promising, not proven). Not the tp=5.0
"breakeven noise" point found while extending the grid in an earlier
round -- that value was never validated beyond the grid search itself.

Usage:
    python -m src.entries_v2_paper --max-iterations 1          # smoke test, one poll then exit
    python -m src.entries_v2_paper                              # run until Ctrl+C, polls every 30 min
    python -m src.entries_v2_paper_stats                        # print current forward-test stats vs backtest
"""

from __future__ import annotations

import asyncio
import csv
import os
import statistics
import time
from datetime import datetime, timezone
from typing import List, Optional

from src import memory
from src.data_deriv import fetch_deriv_candles_async
from src.entries_v2 import LowfreqV2Config, TradeRecordV2, simulate, DEFAULT_COSTS

PAPER_LOG_PATH = os.path.join(memory.DATA_DIR, "entries_v2_paper_trades.csv")
PAPER_LOG_HEADER = ["logged_at", "pair", "direction", "entry_time", "entry_price", "stop", "target",
                     "exit_time", "exit_price", "exit_reason", "pnl",
                     "running_trades", "running_win_rate_pct", "running_pf", "running_net_pnl_pct"]

DEFAULT_CONFIG = LowfreqV2Config(trend_sma_period=50, pullback_ema_period=21,
                                  pullback_tolerance_pct=0.20, atr_sl_mult=2.0, atr_tp_mult=2.5,
                                  use_regime_filter=True, regime_confirm_bars=3)
H1_WINDOW = 500   # ~3 weeks of H1 bars -- comfortably covers EMA(21)/ATR(14) warmup plus real setup history
DAILY_WINDOW = 100  # comfortably covers trend_sma_period=50 warmup


def _ensure_paper_log() -> None:
    os.makedirs(memory.DATA_DIR, exist_ok=True)
    if not os.path.exists(PAPER_LOG_PATH):
        with open(PAPER_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(PAPER_LOG_HEADER)


def _read_logged_entry_times() -> set:
    _ensure_paper_log()
    with open(PAPER_LOG_PATH, newline="", encoding="utf-8") as f:
        return {row["entry_time"] for row in csv.DictReader(f)}


def _running_stats(all_pnls: List[float], notional: float) -> dict:
    n = len(all_pnls)
    if n == 0:
        return {"n": 0, "win_rate_pct": 0.0, "pf": None, "net_pnl_pct": 0.0}
    wins = [p for p in all_pnls if p > 0]
    losses = [p for p in all_pnls if p < 0]
    gross_profit, gross_loss = sum(wins), abs(sum(losses))
    return {
        "n": n,
        "win_rate_pct": len(wins) / n * 100,
        "pf": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "net_pnl_pct": sum(all_pnls) / notional * 100,
    }


def _log_trade(t: TradeRecordV2, symbol: str, running: dict) -> None:
    _ensure_paper_log()
    with open(PAPER_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now(timezone.utc).isoformat(), symbol, t.direction.value,
            t.entry_time.isoformat(), f"{t.entry_price:.5f}", f"{t.stop:.5f}", f"{t.target:.5f}",
            t.exit_time.isoformat(), f"{t.exit_price:.5f}", t.exit_reason, f"{t.pnl:.2f}",
            running["n"], f"{running['win_rate_pct']:.2f}",
            f"{running['pf']:.3f}" if running["pf"] is not None else "undef",
            f"{running['net_pnl_pct']:.2f}",
        ])


async def run_paper_mode(symbol: str = "XAUUSD_DERIV", notional: float = 10_000.0,
                          config: LowfreqV2Config = DEFAULT_CONFIG, poll_seconds: float = 1800.0,
                          max_iterations: Optional[int] = None, log=print) -> None:
    log(f"[ENTRIES_V2 PAPER] Config: {config.as_dict()}")
    log(f"[ENTRIES_V2 PAPER] Data source: Deriv ({symbol}), H1 execution / daily trend filter. "
        f"Logging to {PAPER_LOG_PATH}\n")

    logged_entry_times = _read_logged_entry_times()
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        iteration += 1
        try:
            h1 = await fetch_deriv_candles_async(symbol=symbol, interval="1h", limit=H1_WINDOW)
            daily = await fetch_deriv_candles_async(symbol=symbol, interval="1d", limit=DAILY_WINDOW)
            trades = simulate(h1, daily, config, notional, DEFAULT_COSTS)
        except Exception as e:
            log(f"[ENTRIES_V2 PAPER] Poll failed ({type(e).__name__}: {e}), will retry next poll.")
            trades = []

        new_trades = [t for t in trades if t.entry_time.isoformat() not in logged_entry_times]
        if new_trades:
            # Running stats accumulate over ALL logged trades, oldest to newest.
            existing_pnls = []
            with open(PAPER_LOG_PATH, newline="", encoding="utf-8") as f:
                existing_pnls = [float(row["pnl"]) for row in csv.DictReader(f)]
            for t in sorted(new_trades, key=lambda t: t.exit_time):
                existing_pnls.append(t.pnl)
                running = _running_stats(existing_pnls, notional)
                _log_trade(t, symbol, running)
                logged_entry_times.add(t.entry_time.isoformat())
                pf_str = f"{running['pf']:.2f}" if running["pf"] is not None else "undef"
                log(f"[ENTRIES_V2 PAPER] NEW closed trade: {t.direction.value} {t.entry_time.date()} @ "
                    f"{t.entry_price:.2f} -> {t.exit_time.date()} @ {t.exit_price:.2f} ({t.exit_reason}), "
                    f"pnl {t.pnl:+.2f}. Running: n={running['n']} win%={running['win_rate_pct']:.1f} "
                    f"pf={pf_str} net%={running['net_pnl_pct']:+.2f}\n")
        else:
            log(f"[ENTRIES_V2 PAPER] Poll {iteration}: no newly-closed trades.")

        if max_iterations is None or iteration < max_iterations:
            await asyncio.sleep(poll_seconds)

    log("[ENTRIES_V2 PAPER] Stopped.")


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Live paper-trading harness for XAUUSD_LOWFREQ v2 (Deriv-sourced, no real orders).")
    parser.add_argument("--symbol", default="XAUUSD_DERIV")
    parser.add_argument("--notional", type=float, default=10_000.0)
    parser.add_argument("--poll-seconds", type=float, default=1800.0, help="How often to check for new H1 bars.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Stop after N polls (omit to run until Ctrl+C).")
    args = parser.parse_args()

    asyncio.run(run_paper_mode(symbol=args.symbol, notional=args.notional,
                                poll_seconds=args.poll_seconds, max_iterations=args.max_iterations))


if __name__ == "__main__":
    _cli()
