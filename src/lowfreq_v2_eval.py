# -*- coding: utf-8 -*-
"""
lowfreq_v2_eval.py

Retraining/validation driver for XAUUSD_LOWFREQ v2 (src/entries_v2.py),
per the retraining brief. Three phases, run in this order and never out
of order:

  1. Walk-forward parameter selection -- grid search over entries_v2's
     5-parameter budget, scored ONLY on training data (2018-01-01 to
     2025-07-31) via rolling 12-month-context/3-month-validate folds,
     sliding 3 months. A combo's score is the MEDIAN of its per-fold
     Sharpe (not mean, not max) -- guards against a combo that's
     spectacular on one fold and carried by it alone.
  2. Overfitting guards on the SELECTED combo only, still training data
     only: parameter sensitivity (+/-20%/+/-40% perturbation) and a
     pre-COVID/COVID/post-COVID regime breakdown.
  3. ONE evaluation run on the untouched test window (2025-08-01 to
     2026-07-31) -- equity curve, full metrics table, Monte Carlo,
     significance test. This function is only ever called once per
     report and its output is never fed back into phases 1-2.

All candle data comes from src/data_dukascopy.py's real, cross-validated
M5 cache (2018-01-01 through the present, verified complete for this
range in data/pre_retrain_snapshot.md) -- H1 and daily are pure
resamples of it, same as medfreq_strategy.py's approach.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from itertools import product
from typing import List, Optional

from src import data_dukascopy
from src.candle import Candle
from src.entries_v2 import LowfreqV2Config, simulate, compute_metrics, CostModel, DEFAULT_COSTS

TRAIN_START = datetime(2018, 1, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2025, 7, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START = datetime(2025, 8, 1, tzinfo=timezone.utc)
TEST_END = datetime(2026, 7, 31, 23, 59, 59, tzinfo=timezone.utc)
NOTIONAL = 10_000.0

# Search ranges -- deliberately small (48 combinations), per this project's
# established anti-overfitting convention (see src/optimize.py's docstring:
# "searching a huge space against one split is itself a form of overfitting").
PARAM_GRID = {
    "trend_sma_period": [50, 100],
    "pullback_ema_period": [21, 50],
    "pullback_tolerance_pct": [0.10, 0.20],
    "atr_sl_mult": [1.0, 1.5, 2.0],
    "atr_tp_mult": [1.5, 2.5],
}


def _slice(candles: List[Candle], start: datetime, end: datetime) -> List[Candle]:
    start_ms, end_ms = start.timestamp() * 1000, end.timestamp() * 1000
    return [c for c in candles if start_ms <= c.open_time <= end_ms]


def load_all_candles(from_date: str = "2018-01-01"):
    m5 = data_dukascopy.load_m5_candles(from_date)
    h1 = data_dukascopy.resample(m5, 60)
    daily = data_dukascopy.resample(m5, 1440)
    return h1, daily


def generate_folds(train_start: datetime, train_end: datetime,
                    train_months: int = 12, validate_months: int = 3, slide_months: int = 3) -> List[dict]:
    folds = []
    fold_train_start = train_start
    while True:
        fold_train_end = _add_months(fold_train_start, train_months)
        fold_validate_end = _add_months(fold_train_end, validate_months)
        if fold_validate_end > train_end:
            break
        folds.append({"train_start": fold_train_start, "train_end": fold_train_end, "validate_end": fold_validate_end})
        fold_train_start = _add_months(fold_train_start, slide_months)
    return folds


def _add_months(dt: datetime, months: int) -> datetime:
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, 28)
    return dt.replace(year=year, month=month, day=day)


def run_window(h1_all: List[Candle], daily_all: List[Candle], config: LowfreqV2Config,
                window_start: datetime, window_end: datetime, lookback_days: int = 220,
                notional: float = NOTIONAL, costs: CostModel = DEFAULT_COSTS) -> dict:
    """
    Simulates from (window_start - lookback_days) through window_end (so
    indicators are warmed up before the window starts), then scores only
    trades whose ENTRY falls inside [window_start, window_end) -- a trade
    opened just before the window closes may still exit slightly after
    window_end, which is realistic and kept.
    """
    pad_start = window_start - timedelta(days=lookback_days)
    h1_slice = _slice(h1_all, pad_start, window_end)
    daily_slice = _slice(daily_all, pad_start, window_end)
    all_trades = simulate(h1_slice, daily_slice, config, notional, costs)
    window_trades = [t for t in all_trades if window_start <= t.entry_time < window_end]
    metrics = compute_metrics(window_trades, notional)
    return {"trades": window_trades, "metrics": metrics}


def score_combo_median_sharpe(h1_all, daily_all, config: LowfreqV2Config, folds: List[dict]) -> dict:
    """Runs one param combo across all folds' validate windows. Folds with <3 trades score Sharpe=0.0
    (neutral -- not favorable, not a penalty invented to hit a target) rather than being dropped,
    so a combo can't win by simply avoiding trading in most folds."""
    fold_sharpes = []
    fold_results = []
    for fold in folds:
        result = run_window(h1_all, daily_all, config, fold["train_end"], fold["validate_end"])
        m = result["metrics"]
        sharpe = m["sharpe"] if (m["sharpe"] is not None and m["n_trades"] >= 3) else 0.0
        fold_sharpes.append(sharpe)
        fold_results.append({"validate_start": fold["train_end"], "validate_end": fold["validate_end"], "metrics": m})
    return {"median_sharpe": statistics.median(fold_sharpes), "fold_sharpes": fold_sharpes, "fold_results": fold_results}


if __name__ == "__main__":
    import sys
    import time

    print("Loading full Dukascopy M5 cache and resampling to H1/daily...")
    t0 = time.time()
    h1_all, daily_all = load_all_candles("2018-01-01")
    print(f"Loaded {len(h1_all)} H1 bars, {len(daily_all)} daily bars in {time.time()-t0:.1f}s.")

    folds = generate_folds(TRAIN_START, TRAIN_END)
    print(f"\nGenerated {len(folds)} walk-forward folds (12mo context / 3mo validate / slide 3mo):")
    for f in folds:
        print(f"  context {f['train_start'].date()} -> {f['train_end'].date()} | validate {f['train_end'].date()} -> {f['validate_end'].date()}")

    if "--benchmark" in sys.argv:
        print("\n--- Benchmark: 1 combo x first 2 folds ---")
        cfg = LowfreqV2Config()
        t0 = time.time()
        result = score_combo_median_sharpe(h1_all, daily_all, cfg, folds[:2])
        elapsed = time.time() - t0
        print(f"2-fold score in {elapsed:.2f}s -> median_sharpe={result['median_sharpe']:.3f}, per-fold={result['fold_sharpes']}")
        n_combos = 1
        for v in PARAM_GRID.values():
            n_combos *= len(v)
        est_total = elapsed / 2 * len(folds) * n_combos
        print(f"Grid has {n_combos} combos x {len(folds)} folds = {n_combos*len(folds)} runs.")
        print(f"Estimated full grid-search runtime: {est_total:.0f}s (~{est_total/60:.1f} min)")
