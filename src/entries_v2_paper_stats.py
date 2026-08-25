# -*- coding: utf-8 -*-
"""
entries_v2_paper_stats.py

Small script: reads data/entries_v2_paper_trades.csv (written by
src/entries_v2_paper.py) and prints current forward-test stats next to
the two backtest reference points -- the training-period expectation
(the more representative one) and the single test-window result (the
one that didn't hold up statistically, kept here as a comparison point
precisely so it's obvious if live results are only echoing that one
lucky-looking year rather than the honest training baseline).

Usage:
    python -m src.entries_v2_paper_stats
"""

from __future__ import annotations

import csv
import math
import os
import statistics

from src import memory

PAPER_LOG_PATH = os.path.join(memory.DATA_DIR, "entries_v2_paper_trades.csv")

# Reference points from data/performance_report.md's retraining rounds --
# updated by hand if those numbers are ever revised, not read live from
# the report (keeps this script dependency-free and fast).
TRAIN_REFERENCE = {"label": "Training (2018-2025-07, full period, selected params)",
                    "sharpe": -0.567, "pf": 0.99, "win_rate_pct": 44.4, "net_pnl_pct": -6.10}
TEST_REFERENCE = {"label": "Test window (2025-08 to 2026-07, one-time run, NOT statistically significant, p=0.186)",
                   "sharpe": 1.056, "pf": 1.212, "win_rate_pct": 50.2, "net_pnl_pct": 23.59}


def load_paper_trades() -> list:
    if not os.path.exists(PAPER_LOG_PATH):
        return []
    with open(PAPER_LOG_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_live_stats(rows: list, notional: float = 10_000.0, risk_free_annual: float = 0.05) -> dict:
    pnls = [float(r["pnl"]) for r in rows]
    n = len(pnls)
    if n == 0:
        return {"n": 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / n * 100
    gross_profit, gross_loss = sum(wins), abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else None
    net_pnl_pct = sum(pnls) / notional * 100

    sharpe = None
    if n >= 5:
        from datetime import datetime
        times = sorted(datetime.fromisoformat(r["exit_time"]) for r in rows)
        years = (times[-1] - times[0]).total_seconds() / (365.25 * 86400)
        if years > 0:
            trades_per_year = n / years
            returns = [p / notional for p in pnls]
            mean_r = statistics.mean(returns)
            std_r = statistics.pstdev(returns)
            rf_per_trade = risk_free_annual / trades_per_year
            if std_r > 0:
                sharpe = (mean_r - rf_per_trade) / std_r * math.sqrt(trades_per_year)

    return {"n": n, "win_rate_pct": win_rate, "pf": pf, "net_pnl_pct": net_pnl_pct, "sharpe": sharpe}


def _fmt(v, fmt="{:.2f}"):
    return fmt.format(v) if v is not None else "--"


def main() -> None:
    rows = load_paper_trades()
    live = compute_live_stats(rows)

    print("=== XAUUSD_LOWFREQ v2 -- forward-test vs. backtest ===\n")
    if live["n"] == 0:
        print("No closed paper trades logged yet. Run `python -m src.entries_v2_paper` to start collecting them.")
        return

    print(f"{'':38} {'Trades':>7} {'Win%':>7} {'PF':>7} {'Sharpe':>8} {'NetPnL%':>9}")
    print(f"{'LIVE (paper, so far)':38} {live['n']:7d} {_fmt(live['win_rate_pct']):>7} "
          f"{_fmt(live['pf']):>7} {_fmt(live['sharpe']):>8} {_fmt(live['net_pnl_pct']):>9}")
    print(f"{TRAIN_REFERENCE['label']:38.38} {'2153':>7} {_fmt(TRAIN_REFERENCE['win_rate_pct']):>7} "
          f"{_fmt(TRAIN_REFERENCE['pf']):>7} {_fmt(TRAIN_REFERENCE['sharpe']):>8} {_fmt(TRAIN_REFERENCE['net_pnl_pct']):>9}")
    print(f"{TEST_REFERENCE['label']:38.38} {'247':>7} {_fmt(TEST_REFERENCE['win_rate_pct']):>7} "
          f"{_fmt(TEST_REFERENCE['pf']):>7} {_fmt(TEST_REFERENCE['sharpe']):>8} {_fmt(TEST_REFERENCE['net_pnl_pct']):>9}")

    print("\nRead this as: is live tracking closer to the training baseline (the honest expectation) "
          "or the test-window result (the one that didn't hold up to a significance test)? "
          "Needs a meaningful trade count (dozens+) before this comparison means much -- "
          f"currently at {live['n']} trades.")


if __name__ == "__main__":
    main()
