# el-scotto

XAUUSD_LOWFREQ v2 + N-bar persistence regime filter — a focused
extraction of one specific trading model from a larger research
project ([full history and every other strategy tried is in
`trading-robot`](https://github.com/Ngaakudzwe2/trading-robot)). This
repo contains exactly what's needed to test and run this one model:
nothing else.

## What this is

A gold (XAUUSD) trading strategy: an H1 pullback-to-EMA entry inside a
daily-trend regime, gated by an ATR-expansion "Can_Trade" filter that
must hold for 3 consecutive bars before an entry is allowed. All
parameters are fixed, not curve-fit — see `src/entries_v2.py` and
`src/regime_filter.py` for the exact, documented reasoning behind each
one.

**Read this before trusting any of it.** Training and holdout disagree,
and that tension is not resolved:

| Metric | Training (2018 – 2025-07, 7.5 yrs) | Holdout (2025-08 – 2026-07, locked, run once) |
|---|---:|---:|
| Trades | 1,073 | 132 |
| Win rate | 45.1% | 56.8% |
| Profit Factor | 1.018 | **1.544** |
| Sharpe | -0.519 | **1.843** |
| Sortino | -0.742 | 2.994 |
| Max Drawdown | 14.30% | 4.32% |
| Net P&L | +6.00% | **+31.17%** |

The holdout result is the strongest this project has produced — it's
the first one to pass a significance test (t=2.16, p=0.033), it holds
up reasonably well under outlier removal (Sharpe stays at 1.18 even
with the 3 best trades removed), and a 5,000-resample Monte Carlo on
those 132 trades shows only 1.42% of paths losing money. But **training
never validated this design on its own terms** (Sharpe was negative
across 7.5 years) — one good year, however rigorously it holds up,
cannot by itself overturn that. Treat this as promising and worth
continued forward-testing, not as a proven edge ready for capital.

**No real money has ever been risked with this model.** Every trade in
`data/entries_v2_paper_trades.csv` is a simulated fill against real,
live-polled Deriv prices — there is no order-execution or broker-auth
code in this repo at all.

## What's in here

```
src/entries_v2.py            the strategy itself (LowfreqV2Config, simulate(), compute_metrics())
src/regime_filter.py         the binary ATR-expansion Can_Trade gate
src/lowfreq_v2_eval.py       walk-forward training-data evaluation harness
src/entries_v2_paper.py      live paper-trading harness (Deriv-sourced, no real orders)
src/entries_v2_paper_stats.py   compares live results against the table above
src/data_dukascopy.py        real historical M5 data fetch/cache (Dukascopy)
src/data_deriv.py            real live price feed for paper trading (Deriv)
src/medfreq_strategy.py      only used for one shared utility (align_htf_to_m5); not a dependency otherwise
src/market_data.py, data_binance.py, data_yfinance.py   transitive imports of medfreq_strategy.py, unused by entries_v2 directly
src/candle.py, indicators.py, backtest_structures.py, memory.py   shared low-level utilities
data/dukascopy_m5_cache.csv, data/dukascopy_hours_done.csv   real XAUUSD M5 tick-derived history, 2018-01-01 to present (~40MB) -- bundled so testing doesn't require re-fetching for hours
data/entries_v2_paper_trades.csv   the real forward-test log for this exact configuration, as of the point this repo was created
```

No synthetic or generated market data anywhere in this repo -- the
Dukascopy cache is real tick-derived history, cross-validated against
Yahoo Finance in the parent project.

## Setup

```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Testing (training-data walk-forward, no live connection needed)

```
python -m src.lowfreq_v2_eval
```

Uses the bundled `data/dukascopy_m5_cache.csv` -- reproduces the
training-period numbers in the table above.

## Running (live paper-trading -- no real orders, no credentials needed)

```
python -m src.entries_v2_paper                # runs until Ctrl+C, polls Deriv every 30 min
python -m src.entries_v2_paper --max-iterations 1   # single bounded poll, for a smoke test
python -m src.entries_v2_paper_stats           # compare accumulated live results against training/holdout
```

Logs every decision to `data/entries_v2_paper_trades.csv`. Needs no
API key or account -- Deriv's public historical/streaming endpoints
require no authentication.

## Eventual use (not built yet)

This repo is deliberately paper-trading-only. Connecting it to real
order execution (MT5 or otherwise) is a separate, not-yet-done step --
see the parent `trading-robot` repo's MT5 ZeroMQ bridge work if that's
the direction this goes, but nothing here places a real order, and
that shouldn't change without treating it as the real decision it is.
