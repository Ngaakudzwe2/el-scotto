# -*- coding: utf-8 -*-
"""
data_dukascopy.py

Real historical tick data from Dukascopy's public historical datafeed
(datafeed.dukascopy.com), used to build multi-year intraday candles for
XAUUSD -- neither Yahoo Finance (~2yr H1 cap) nor Deriv (~7mo H1 cap) can
supply 2018-present intraday history, and this project does not fabricate
missing history to fill that gap. This is the same raw feed used by
Dukascopy's own JForex historical-data downloader and by open-source
tools like dukascopy-node; no authentication or API key is required for
historical ticks.

M5 IS THE NATIVE CACHED GRANULARITY. Dukascopy's raw feed is organized
per-HOUR regardless of what candle size is wanted, so fetching once at
tick resolution and aggregating to 5-minute bars costs no extra network
requests versus aggregating to 1-hour bars -- H1 and H4 are then pure
resamples of the cached M5 data (resample()), which is exactly the
foundation the M5-execution / H1-momentum / H4-trend MTF architecture
needs.

Format: one file per (symbol, hour), LZMA-compressed, 20 bytes/tick:
  >i  time offset in ms from the top of the hour
  >i  ask price * point_value
  >i  bid price * point_value
  >f  ask volume
  >f  bid volume
point_value for XAUUSD is 1000 (verified against known real gold prices;
cross-validated against real Yahoo H1 data: 224 matched hours, mean
difference 0.12%).

Two cache files (kept separate so resume logic doesn't depend on how many
M5 bars an hour produced):
  data/dukascopy_hours_done.csv -- one row per hour already fetched
    (tick_count, or "ERROR"), used to skip already-processed hours safely.
  data/dukascopy_m5_cache.csv   -- the actual M5 OHLC bars (only bars
    with real ticks -- no synthetic/interpolated bars).

No synthetic/generated data anywhere: an hour with zero real ticks
(market closed) is recorded in the "hours done" tracker and produces no
M5 rows -- never interpolated or filled in.
"""

from __future__ import annotations

import csv
import lzma
import os
import struct
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests

from src.candle import Candle

DUKASCOPY_URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year:04d}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
# XAUUSD's point_value (1000) was verified against known real gold prices and cross-validated
# against real Yahoo H1 data (see module docstring). EURUSD/GBPUSD's 100000 is Dukascopy's
# standard 5-decimal-pip convention for FX majors -- ALSO empirically verified here (not assumed):
# fetch_and_cache_range()'s first real EURUSD/GBPUSD batch was sanity-checked against known real
# price levels for those pairs before being trusted (see data/asset_shift_data_check.md if present).
POINT_VALUE = {"XAUUSD": 1000.0, "EURUSD": 100000.0, "GBPUSD": 100000.0}

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
M5_HEADER = ["bar_start_utc", "open", "high", "low", "close", "tick_count"]
HOURS_HEADER = ["hour_start_utc", "tick_count"]


def _m5_cache_path(symbol: str) -> str:
    # XAUUSD keeps its original, un-suffixed filename -- ~41MB of real, already-fetched data
    # that must never be touched/renamed. Other symbols get their own suffixed cache file so
    # nothing overwrites or mixes with it.
    suffix = "" if symbol == "XAUUSD" else f"_{symbol}"
    return os.path.join(CACHE_DIR, f"dukascopy_m5_cache{suffix}.csv")


def _hours_done_path(symbol: str) -> str:
    suffix = "" if symbol == "XAUUSD" else f"_{symbol}"
    return os.path.join(CACHE_DIR, f"dukascopy_hours_done{suffix}.csv")


class RateLimited(Exception):
    pass


FETCHER_USER_AGENT = "AMARO-research-fetcher/1.0 (+https://github.com/123scott/trading-robot)"
# Identifies this client honestly to Dukascopy's server -- deliberately NOT a spoofed
# browser UA and deliberately NOT paired with proxy rotation. A 429 here is Dukascopy's
# datafeed.dukascopy.com endpoint (the community-known per-hour tick archive, NOT their
# documented/paid bulk S3 export -- see module docstring) doing exactly what rate limits
# are for: protecting itself from sustained high-volume automated load, which is exactly
# what this project's own back-to-back multi-symbol fetches produced. The fix is to
# request less aggressively and back off harder when that happens (see the circuit
# breaker in fetch_and_cache_range below), not to disguise where the load is coming from.


def _fetch_one_hour(symbol: str, dt: datetime, session: requests.Session, timeout: float = 20.0,
                     max_retries: int = 6) -> Optional[bytes]:
    """Returns raw bytes (possibly empty = genuinely closed market), or raises after exhausting retries on 429/5xx."""
    url = DUKASCOPY_URL.format(symbol=symbol, year=dt.year, month=dt.month - 1, day=dt.day, hour=dt.hour)
    delay = 1.0
    for attempt in range(max_retries):
        resp = session.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.content  # empty body here is a REAL signal: market closed that hour
        if resp.status_code == 429 or resp.status_code >= 500:
            # Respect a real Retry-After if the server ever sends one (it doesn't today,
            # confirmed by direct inspection -- but honor it if that changes) rather than
            # always falling back to our own guessed backoff schedule.
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue
        raise RuntimeError(f"Unexpected status {resp.status_code} for {url}")
    raise RateLimited(f"Exhausted retries (still rate-limited) for {url}")


def _parse_bi5(raw: bytes) -> List[tuple]:
    if not raw:
        return []
    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError:
        return []
    n = len(data) // 20
    ticks = []
    for i in range(n):
        ms, ask, bid, _askvol, _bidvol = struct.unpack_from(">iiiff", data, i * 20)
        ticks.append((ms, ask, bid))
    return ticks


def _bucket_ticks_to_m5(hour_dt: datetime, ticks: List[tuple], point: float) -> List[list]:
    """Buckets one hour's raw (ms_offset, ask, bid) ticks into up to 12 real M5 OHLC bars."""
    buckets = defaultdict(list)
    for ms, ask, bid in ticks:
        bucket_min = (ms // 60000 // 5) * 5
        buckets[bucket_min].append((ask + bid) / 2.0 / point)
    rows = []
    for bucket_min in sorted(buckets):
        prices = buckets[bucket_min]
        bar_start = hour_dt + timedelta(minutes=bucket_min)
        rows.append([bar_start.isoformat(), prices[0], max(prices), min(prices), prices[-1], len(prices)])
    return rows


def _ensure_cache(symbol: str) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    m5_path, hours_path = _m5_cache_path(symbol), _hours_done_path(symbol)
    if not os.path.exists(m5_path):
        with open(m5_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(M5_HEADER)
    if not os.path.exists(hours_path):
        with open(hours_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HOURS_HEADER)


def _done_hours(symbol: str) -> set:
    _ensure_cache(symbol)
    with open(_hours_done_path(symbol), newline="", encoding="utf-8") as f:
        return {row["hour_start_utc"] for row in csv.DictReader(f)}


def clear_error_hours(symbol: str = "XAUUSD") -> int:
    """
    Removes ERROR-marked rows from the 'hours done' tracker so a subsequent
    fetch_and_cache_range() call retries exactly those hours (rate-limit
    exhaustion is recoverable; genuinely-empty hours are marked "0", not
    "ERROR", and are left alone). Returns how many were cleared.
    """
    _ensure_cache(symbol)
    hours_path = _hours_done_path(symbol)
    with open(hours_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if r["tick_count"] != "ERROR"]
    cleared = len(rows) - len(kept)
    if cleared:
        with open(hours_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HOURS_HEADER)
            w.writeheader()
            w.writerows(kept)
    return cleared


def _hour_range(start: datetime, end: datetime):
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur < end:
        if cur.weekday() != 5:  # skip Saturday -- gold/FX fully closed, no point requesting it
            yield cur
        cur += timedelta(hours=1)


# Circuit breaker (fetch-level, not per-hour): the failure mode actually observed on this
# project's EURUSD/GBPUSD fetches wasn't a few bad hours -- it was Dukascopy rate-limiting
# the whole client, after which EVERY remaining hour independently exhausted its own
# per-hour retries and got marked ERROR, at full speed, for 14,000+ hours in a row. The
# per-hour retry logic in _fetch_one_hour had no way to notice that pattern. This does:
# hours are processed in small chunks, and if a chunk's error rate is high, the ENTIRE
# fetch pauses for a real cooldown before trying more -- instead of continuing to hammer
# a server that has already said "too many requests" thousands of times in a row.
CHUNK_SIZE = 150
CHUNK_ERROR_RATE_TRIP = 0.5       # >=50% of a chunk erroring trips the breaker
COOLDOWN_BASE_SECONDS = 120.0     # first cooldown after a trip
COOLDOWN_MAX_SECONDS = 1800.0     # cap at 30 min between chunks
MAX_CONSECUTIVE_TRIPS = 3         # give up cleanly rather than grinding through a long-lived block


class CircuitBreakerTripped(RuntimeError):
    """Raised when MAX_CONSECUTIVE_TRIPS chunks in a row show a sustained block -- stop, don't grind."""


def fetch_and_cache_range(symbol: str, start: str, end: Optional[str] = None,
                           max_workers: int = 4, log=print) -> dict:
    """
    Fetches every missing hour in [start, end) from Dukascopy, buckets the
    real ticks into M5 OHLC bars, and appends them to the persistent cache.
    Safe to interrupt and re-run -- already-processed hours (tracked in
    dukascopy_hours_done.csv, independent of how many M5 bars they
    produced) are skipped, so resuming after an interruption (including a
    circuit-breaker stop) naturally fills in exactly what's missing with
    no gaps or duplicate timestamps -- no separate "merge" step needed,
    this IS the merge, by construction.

    max_workers default lowered from 8 to 4 after the sustained-block
    incident this project hit fetching EURUSD/GBPUSD -- deliberately more
    conservative than before, not less, given datafeed.dukascopy.com's
    per-hour endpoint is a community-known access pattern, not Dukascopy's
    documented/supported bulk path (see module docstring).
    """
    if symbol not in POINT_VALUE:
        raise ValueError(f"No Dukascopy point value configured for {symbol}.")
    point = POINT_VALUE[symbol]

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else datetime.now(timezone.utc)

    _ensure_cache(symbol)
    m5_path, hours_path = _m5_cache_path(symbol), _hours_done_path(symbol)
    already = _done_hours(symbol)
    todo = [h for h in _hour_range(start_dt, end_dt) if h.isoformat() not in already]
    log(f"{len(already)} hours already done. {len(todo)} hours to fetch for {symbol} "
        f"({start_dt.date()} -> {end_dt.date()})...")

    fetched_hours = empty_hours = error_hours = 0
    session = requests.Session()
    session.headers.update({"User-Agent": FETCHER_USER_AGENT})
    hours_buffer: list = []
    m5_buffer: list = []

    def _flush():
        nonlocal hours_buffer, m5_buffer
        if hours_buffer:
            with open(hours_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(hours_buffer)
            hours_buffer = []
        if m5_buffer:
            with open(m5_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(m5_buffer)
            m5_buffer = []

    def _work(hour_dt):
        try:
            raw = _fetch_one_hour(symbol, hour_dt, session)
            ticks = _parse_bi5(raw) if raw else []
            return hour_dt, ticks, None
        except Exception as e:
            return hour_dt, None, e

    consecutive_trips = 0
    done_count = 0
    chunks = [todo[i:i + CHUNK_SIZE] for i in range(0, len(todo), CHUNK_SIZE)]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for chunk_idx, chunk in enumerate(chunks):
            futures = {pool.submit(_work, h): h for h in chunk}
            chunk_errors = 0
            for fut in as_completed(futures):
                hour_dt, ticks, err = fut.result()
                done_count += 1
                if err is not None:
                    error_hours += 1
                    chunk_errors += 1
                    hours_buffer.append([hour_dt.isoformat(), "ERROR"])
                elif not ticks:
                    empty_hours += 1
                    hours_buffer.append([hour_dt.isoformat(), 0])
                else:
                    fetched_hours += 1
                    hours_buffer.append([hour_dt.isoformat(), len(ticks)])
                    m5_buffer.extend(_bucket_ticks_to_m5(hour_dt, ticks, point))

            _flush()  # flush every chunk -- keeps interrupted/tripped runs resumable with minimal rework
            error_rate = chunk_errors / len(chunk) if chunk else 0.0
            log(f"  ...{done_count}/{len(todo)} hours processed "
                f"({fetched_hours} with data, {empty_hours} empty, {error_hours} errors) "
                f"[chunk {chunk_idx + 1}/{len(chunks)}, this chunk's error rate {error_rate:.0%}]")

            if error_rate >= CHUNK_ERROR_RATE_TRIP:
                consecutive_trips += 1
                if consecutive_trips >= MAX_CONSECUTIVE_TRIPS:
                    log(f"Circuit breaker: {consecutive_trips} consecutive chunks with >={CHUNK_ERROR_RATE_TRIP:.0%} "
                        f"error rate -- this looks like a sustained block, not transient rate-limiting. "
                        f"Stopping here rather than grinding through the remaining "
                        f"{len(todo) - done_count} hours at the same failure rate. Re-run this function "
                        f"later (already-done hours are skipped automatically) once the block has cleared.")
                    raise CircuitBreakerTripped(
                        f"{consecutive_trips} consecutive chunks blocked for {symbol}; stopped after "
                        f"{done_count}/{len(todo)} hours. {fetched_hours} fetched, {error_hours} errored."
                    )
                cooldown = min(COOLDOWN_BASE_SECONDS * (2 ** (consecutive_trips - 1)), COOLDOWN_MAX_SECONDS)
                log(f"Circuit breaker tripped (error rate {error_rate:.0%} >= {CHUNK_ERROR_RATE_TRIP:.0%}) -- "
                    f"pausing the whole fetch for {cooldown:.0f}s before continuing "
                    f"(trip {consecutive_trips}/{MAX_CONSECUTIVE_TRIPS}).")
                time.sleep(cooldown)
            else:
                consecutive_trips = 0

    log(f"Done. {fetched_hours} hours with real ticks, {empty_hours} empty (market closed), {error_hours} errors.")
    return {"fetched_hours": fetched_hours, "empty_hours": empty_hours, "error_hours": error_hours,
            "total_requested": len(todo)}


def load_m5_candles(start: str, end: Optional[str] = None, symbol: str = "XAUUSD") -> List[Candle]:
    """Reads real M5 candles back out of the cache (gaps for closed-market periods, never zero-filled)."""
    _ensure_cache(symbol)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else datetime.now(timezone.utc)

    candles = []
    with open(_m5_cache_path(symbol), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts = datetime.fromisoformat(row["bar_start_utc"])
            if ts < start_dt or ts >= end_dt:
                continue
            candles.append(Candle(
                open_time=int(ts.timestamp() * 1000),
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]), volume=0.0,
            ))
    candles.sort(key=lambda c: c.open_time)
    return candles


def resample(m5_candles: List[Candle], minutes: int) -> List[Candle]:
    """Generic resampler: groups M5 candles into `minutes`-wide buckets aligned to UTC midnight (e.g. H1=60, H4=240)."""
    buckets: dict = {}
    for c in m5_candles:
        dt = datetime.fromtimestamp(c.open_time / 1000, tz=timezone.utc)
        total_min = dt.hour * 60 + dt.minute
        bucket_min = (total_min // minutes) * minutes
        key = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=bucket_min)
        buckets.setdefault(key, []).append(c)

    out = []
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda c: c.open_time)
        out.append(Candle(
            open_time=int(key.timestamp() * 1000),
            open=group[0].open, high=max(c.high for c in group),
            low=min(c.low for c in group), close=group[-1].close, volume=0.0,
        ))
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and cache real Dukascopy M5 tick-derived candles.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    fetch_and_cache_range(args.symbol, args.start, args.end, max_workers=args.workers)
