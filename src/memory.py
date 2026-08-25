# -*- coding: utf-8 -*-
"""
memory.py

Two-file memory system for the replay backtester:

  data/ledger.csv    -- append-only trade log: every BUY/SELL/SKIP decision
                         and, once known, its realized outcome.
  data/learnings.md  -- plain-English warnings distilled from realized
                         losses, written in a lightly structured format
                         this module can also parse back out.

check_memory() is called before every BUY/SELL decision is finalized. It
reads BOTH files, looks for prior crossover losses (from the ledger) and
learnings warnings (from learnings.md) that match the current symbol,
crossover direction, and price zone. If either matches, the action is
downgraded to SKIP. It always returns the full reasoning chain as a list
of plain-English lines so the caller can print exactly why a decision was
made.

Decay: matches older than MEMORY_DECAY_DAYS are ignored. Without this, a
single old loss can permanently veto a price zone forever -- this is a
real failure mode we observed (GBPUSD memory mode skipped every exit
signal for 4+ years off one 2021 loss, never closing its position). Decay
keeps memory responsive to recent evidence instead of freezing on stale
data ("overtraining" on old losses). This module also surfaces a loud
warning when a symbol has racked up a long consecutive-skip streak, since
that's the visible symptom of the memory filter getting stuck.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

LEDGER_HEADER = ["timestamp", "symbol", "action", "price", "quantity", "reason", "mode", "outcome", "pnl"]
PRICE_TOLERANCE_PCT = 1.0  # +/- % band used to decide whether a price "matches" a past zone
MEMORY_DECAY_DAYS = 365  # matches older than this no longer block a trade -- see module docstring.
# Chosen empirically: tested 90/180/365/730 days against the 2018-present Yahoo backtest. XAUUSD
# benefits from memory at every window tested; USDJPY's memory mode underperforms raw mode at
# EVERY window tested (this asset's "loss zones" apparently get revisited profitably later --
# skipping them costs more than it saves). 365 days was chosen as a defensible middle ground, not
# because it's proven optimal -- if you retune this, rerun both --raw and --memory afterward, since
# raw is decay-independent but memory is not, and stale ledger data can silently understate risk.
CONSECUTIVE_SKIP_ALERT_THRESHOLD = 10  # loud warning once a symbol has skipped this many decisions in a row

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LEDGER_PATH = os.path.join(DATA_DIR, "ledger.csv")
LEARNINGS_PATH = os.path.join(DATA_DIR, "learnings.md")

LEARNINGS_HEADER = "# Trading Learnings\n\nPlain-English warnings distilled from realized losses.\n\n"

# Parses lines of the form:
# - WARNING: BTCUSDT golden-cross near 64500.00-65500.00 produced a loss of -42.30 on 2026-07-15T03:00:00+00:00.
WARNING_RE = re.compile(
    r"-\s*WARNING:\s*(?P<symbol>\S+)\s+(?P<direction>golden-cross|death-cross)\s+.*?"
    r"near\s+(?P<low>[\d.]+)-(?P<high>[\d.]+).*?"
    r"on\s+(?P<timestamp>\S+)\.",
    re.IGNORECASE,
)


def _within_decay_window(timestamp_str: str, decay_days: Optional[int] = None) -> bool:
    """True if `timestamp_str` (ISO format) is recent enough to still count as a match."""
    if decay_days is None:
        decay_days = MEMORY_DECAY_DAYS  # read at call time, not bind time -- module global must stay overridable
    try:
        ts = datetime.fromisoformat(timestamp_str)
    except ValueError:
        return True  # unparseable timestamp -- fail open rather than silently dropping a real warning
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
    return age_days <= decay_days


def ensure_memory_files() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LEDGER_HEADER)
    if not os.path.exists(LEARNINGS_PATH):
        with open(LEARNINGS_PATH, "w", encoding="utf-8") as f:
            f.write(LEARNINGS_HEADER)


def reset_memory_files() -> None:
    """The `--reset` / memory:reset equivalent: wipes both files back to their empty starting state."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LEDGER_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(LEDGER_HEADER)
    with open(LEARNINGS_PATH, "w", encoding="utf-8") as f:
        f.write(LEARNINGS_HEADER)


def _read_ledger_rows() -> List[dict]:
    ensure_memory_files()
    with open(LEDGER_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_learnings_warnings() -> List[dict]:
    ensure_memory_files()
    with open(LEARNINGS_PATH, encoding="utf-8") as f:
        text = f.read()
    return [
        {
            "symbol": m.group("symbol"),
            "direction": m.group("direction").lower(),
            "low": float(m.group("low")),
            "high": float(m.group("high")),
            "timestamp": m.group("timestamp"),
        }
        for m in WARNING_RE.finditer(text)
    ]


def _consecutive_skip_streak(symbol: str) -> int:
    """Counts trailing consecutive SKIP rows for `symbol` (any mode) at the end of the ledger."""
    rows = [r for r in _read_ledger_rows() if r.get("symbol") == symbol and r.get("action") in ("BUY", "SELL", "SKIP")]
    streak = 0
    for row in reversed(rows):
        if row["action"] != "SKIP":
            break
        streak += 1
    return streak


def direction_keyword(reason: str) -> str:
    """Normalizes a strategy reason string ("Golden cross: ...") to "golden-cross" / "death-cross"."""
    return "golden-cross" if "golden" in reason.lower() else "death-cross"


def _price_matches(price: float, ref_price: float, pct: float = PRICE_TOLERANCE_PCT) -> bool:
    band = ref_price * (pct / 100.0)
    return abs(price - ref_price) <= band


@dataclass
class MemoryVerdict:
    final_action: str    # "BUY" | "SELL" | "SKIP"
    reasoning: List[str]  # full plain-English reasoning chain, ready to print


def check_memory(symbol: str, action: str, price: float, reason: str) -> MemoryVerdict:
    """Reads both memory files and decides whether `action` (BUY/SELL) should be downgraded to SKIP."""
    lines: List[str] = [
        f"[MEMORY CHECK] Proposed {action} {symbol} @ {price:.2f} -- {reason}",
        f"  -> Reading data/ledger.csv for prior {symbol} crossover losses near this price...",
    ]

    direction = direction_keyword(reason)
    stale_ledger_count = 0
    ledger_matches = []
    for row in _read_ledger_rows():
        if row.get("symbol") != symbol or row.get("outcome") != "LOSS":
            continue
        if direction.replace("-", " ") not in row.get("reason", "").lower().replace("-", " "):
            continue
        try:
            row_price = float(row["price"])
        except (KeyError, ValueError, TypeError):
            continue
        if not _price_matches(price, row_price):
            continue
        if not _within_decay_window(row.get("timestamp", "")):
            stale_ledger_count += 1
            continue
        ledger_matches.append(row)

    if ledger_matches:
        for row in ledger_matches:
            lines.append(
                f"     found a prior {row['action']} on {row['timestamp']} at {float(row['price']):.2f} "
                f"that closed at a LOSS of {float(row['pnl']):.2f} ({row['reason']})"
            )
    else:
        lines.append(f"     no prior {symbol} crossover losses found near {price:.2f} within the last {MEMORY_DECAY_DAYS} days")
    if stale_ledger_count:
        lines.append(f"     ({stale_ledger_count} older matching loss(es) ignored -- past the {MEMORY_DECAY_DAYS}-day decay window)")

    lines.append("  -> Reading data/learnings.md for matching warnings...")
    all_learnings = [
        w for w in _read_learnings_warnings()
        if w["symbol"] == symbol and w["direction"] == direction and w["low"] <= price <= w["high"]
    ]
    learnings_matches = [w for w in all_learnings if _within_decay_window(w["timestamp"])]
    stale_learnings_count = len(all_learnings) - len(learnings_matches)

    if learnings_matches:
        for w in learnings_matches:
            lines.append(f"     found a learnings.md warning: {symbol} {w['direction']} near {w['low']:.2f}-{w['high']:.2f}")
    else:
        lines.append("     no matching warnings found in learnings.md within the decay window")
    if stale_learnings_count:
        lines.append(f"     ({stale_learnings_count} older matching warning(s) ignored -- past the {MEMORY_DECAY_DAYS}-day decay window)")

    if ledger_matches or learnings_matches:
        lines.append(
            f"  -> DOWNGRADING {action} -> SKIP: {len(ledger_matches)} ledger loss(es) + "
            f"{len(learnings_matches)} learnings warning(s) matched this setup (within {MEMORY_DECAY_DAYS} days)."
        )
        final_action = "SKIP"
    else:
        lines.append(f"  -> No recent precedent found. Proceeding with {action}.")
        final_action = action

    if final_action == "SKIP":
        streak = _consecutive_skip_streak(symbol) + 1  # +1 for the SKIP about to be recorded
        if streak >= CONSECUTIVE_SKIP_ALERT_THRESHOLD:
            lines.append(
                f"  !! ALERT: {symbol} has now skipped {streak} decisions in a row. The memory filter may be "
                f"stuck on this symbol -- consider reviewing data/learnings.md for stale warnings, or that this "
                f"decay window ({MEMORY_DECAY_DAYS}d) may need shortening."
            )

    return MemoryVerdict(final_action=final_action, reasoning=lines)


def record_trade(symbol: str, action: str, price: float, quantity: float, reason: str,
                  mode: str, outcome: str, pnl: float, timestamp: Optional[str] = None) -> None:
    ensure_memory_files()
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    with open(LEDGER_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, symbol, action, f"{price:.2f}", quantity, reason, mode, outcome, f"{pnl:.2f}"])


def record_learning(symbol: str, direction: str, price: float, pnl: float, timestamp: str) -> None:
    """Appends a plain-English, machine-parseable warning after a realized loss."""
    ensure_memory_files()
    band = price * (PRICE_TOLERANCE_PCT / 100.0)
    low, high = price - band, price + band
    line = (
        f"- WARNING: {symbol} {direction} near {low:.2f}-{high:.2f} produced a loss of "
        f"{pnl:.2f} on {timestamp}. Treat similar crossover setups in this zone with caution.\n"
    )
    with open(LEARNINGS_PATH, "a", encoding="utf-8") as f:
        f.write(line)
