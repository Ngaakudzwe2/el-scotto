# -*- coding: utf-8 -*-
"""
data_deriv.py

Real historical and live market data for Deriv's XAUUSD (Gold) instrument,
via Deriv's public WebSocket API (no account/API token required for market
data). No generated or fixture data is used anywhere in this module.

Deriv symbol: "frxXAUUSD" (their forex-style Gold-vs-USD quote).

IMPORTANT, VERIFIED DATA LIMITATION: Deriv's `ticks_history` endpoint for
frxXAUUSD daily candles only serves roughly the trailing ~1 year of
history -- requesting an earlier `start` (tested 2018, 2020, 2023, 2024)
silently returns the same ~258 most-recent daily candles regardless. This
is a real constraint of the public feed, not a bug in this module and not
something this module works around with synthetic data. Callers wanting
the full 2018-present range should use src/data_yfinance.py (the XAUUSD
symbol backed by GC=F) instead; this module is for genuine Deriv-sourced
comparison over the window Deriv actually provides.

Spread handling: the history/ticks endpoints return a single quote price
(no per-candle bid/ask), so this module can't read Deriv's *actual*
historical spread. `DEFAULT_SPREAD` is an explicit, documented
approximation (not a live-fetched figure) applied at the execution layer
(trading_robot.py) so round-trip trades pay a realistic spread cost:
buys fill at quote + spread/2, sells fill at quote - spread/2.
"""

from __future__ import annotations

import asyncio
import json
import ssl
from typing import AsyncIterator, Callable, List, Optional

import certifi
import websockets

from src.candle import Candle

DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"  # public demo app_id, no auth needed for market data
DERIV_SYMBOLS = {"XAUUSD_DERIV": "frxXAUUSD"}
DEFAULT_SPREAD = 0.30  # approximate USD spread for gold; see module docstring

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def deriv_ticker(symbol: str) -> str:
    if symbol not in DERIV_SYMBOLS:
        raise ValueError(f"Unsupported Deriv symbol: {symbol}. Supported: {list(DERIV_SYMBOLS)}")
    return DERIV_SYMBOLS[symbol]


async def fetch_candles_raw_async(ticker: str, granularity: int, start: Optional[int],
                                   end: str, count: int) -> List[dict]:
    async with websockets.connect(DERIV_WS_URL, ssl=_SSL_CTX, open_timeout=15) as ws:
        req = {
            "ticks_history": ticker,
            "adjust_start_time": 1,
            "count": count,
            "end": end,
            "start": start or 1,
            "style": "candles",
            "granularity": granularity,
        }
        await ws.send(json.dumps(req))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
        if "error" in resp:
            raise RuntimeError(f"Deriv API error for {ticker}: {resp['error']}")
        return resp.get("candles", [])


def fetch_deriv_candles(symbol: str = "XAUUSD_DERIV", interval: str = "1d",
                         start: Optional[str] = None, end: Optional[str] = None,
                         limit: int = 5000) -> List[Candle]:
    """
    Pull real historical candles for `symbol` from Deriv. `interval` accepts
    "1d"/"1h"/"1m" (mapped to Deriv's granularity in seconds). `start`/`end`
    are accepted for API-shape compatibility with the other data sources,
    but per the module-level note, Deriv currently only honors the trailing
    ~1 year window for this symbol regardless of what's requested here.
    """
    import datetime as _dt

    granularity = {"1m": 60, "1h": 3600, "1d": 86400}.get(interval, 86400)
    ticker = deriv_ticker(symbol)

    start_epoch = None
    if start:
        start_epoch = int(_dt.datetime.fromisoformat(start).replace(tzinfo=_dt.timezone.utc).timestamp())
    end_param = "latest"
    if end:
        end_param = str(int(_dt.datetime.fromisoformat(end).replace(tzinfo=_dt.timezone.utc).timestamp()))

    rows = asyncio.run(fetch_candles_raw_async(ticker, granularity, start_epoch, end_param, limit))
    return _rows_to_candles(rows)


def _rows_to_candles(rows: List[dict]) -> List[Candle]:
    return [
        Candle(
            open_time=int(row["epoch"]) * 1000,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=0.0,  # Deriv candles don't carry volume
        )
        for row in rows
    ]


async def fetch_deriv_candles_async(symbol: str = "XAUUSD_DERIV", interval: str = "1d",
                                     limit: int = 60) -> List[Candle]:
    """Async variant of fetch_deriv_candles, safe to call from within a running event loop (e.g. live_monitor.py)."""
    granularity = {"1m": 60, "1h": 3600, "1d": 86400}.get(interval, 86400)
    ticker = deriv_ticker(symbol)
    rows = await fetch_candles_raw_async(ticker, granularity, None, "latest", limit)
    return _rows_to_candles(rows)


class DerivApiError(RuntimeError):
    """Raised when Deriv's API responds to a request with an {"error": ...} payload."""


async def stream_deriv_ticks(ticker: str, on_tick: Callable[[dict], None],
                              stop_event: Optional[asyncio.Event] = None) -> None:
    """
    Subscribes to Deriv's live tick stream for the raw Deriv `ticker`
    (e.g. "frxXAUUSD" -- use deriv_ticker(symbol) to resolve from the
    short symbol name first) and calls `on_tick` with each raw tick dict
    ({"quote": ..., "bid": ..., "ask": ..., "epoch": ...}) as it arrives.
    Runs until `stop_event` is set (or the connection drops). Used by the
    --paper forward-testing mode; this function only reads the live feed,
    it never places orders.

    Raises DerivApiError immediately if Deriv rejects the subscription
    (e.g. InvalidSymbol, or the app_id losing streaming authorization --
    a real failure mode observed in production: the previous version of
    this function only checked for msg_type=="tick", so an {"error":...}
    response with msg_type=="tick" and no "tick" key was silently
    ignored -- the loop just sat there forever waiting for a tick that
    would never come, and the caller's only symptom was the connection
    eventually timing out and reconnecting into the exact same silent
    failure, forever. Surface it instead so the caller (and its
    reconnect-with-backoff logic) can actually see what's wrong.
    """
    async with websockets.connect(DERIV_WS_URL, ssl=_SSL_CTX, open_timeout=15) as ws:
        await ws.send(json.dumps({"ticks": ticker, "subscribe": 1}))
        first = json.loads(await ws.recv())
        if "error" in first:
            raise DerivApiError(f"Deriv rejected tick subscription for {ticker!r}: "
                                 f"{first['error'].get('code')}: {first['error'].get('message')}")
        if first.get("msg_type") == "tick" and "tick" in first:
            on_tick(first["tick"])
        while stop_event is None or not stop_event.is_set():
            raw = await ws.recv()
            msg = json.loads(raw)
            if "error" in msg:
                raise DerivApiError(f"Deriv sent an error mid-stream for {ticker!r}: "
                                     f"{msg['error'].get('code')}: {msg['error'].get('message')}")
            if msg.get("msg_type") == "tick" and "tick" in msg:
                on_tick(msg["tick"])
