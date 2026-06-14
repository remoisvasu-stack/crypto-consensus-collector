"""main.py — Collector service (Render Frankfurt, or any non-US region).

Part 1 of 3 (see ../README.md). Responsibilities:
  * Poll the Binance Options chain every POLL_INTERVAL seconds (uninterrupted).
  * Record data by APPENDING new rows into the HF dataset (single source of
    truth) — batched every FLUSH_MINUTES. No local persistence; the host disk is
    treated as ephemeral.
  * Forward each {row, chain} to the Brain over HTTPS for live signals + paper.

Independent of the brain/executor: if they're down, recording continues.
Exposes GET /health for an uptime monitor (Render free spins down on idle).
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI

from core.binance_feed import BinanceOptionsFeed
import store
import webhook

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("collector")

# --- config ---------------------------------------------------------------
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "60"))
FLUSH_MINUTES = float(os.getenv("FLUSH_MINUTES", "10"))     # HF append cadence
BRAIN_URL = os.getenv("BRAIN_URL", "").rstrip("/")          # e.g. https://xxx.hf.space
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "")
SKIP_SAME_DAY = os.getenv("SKIP_SAME_DAY_EXPIRY", "true").lower() == "true"
MIN_EXPIRY_HOURS = float(os.getenv("MIN_EXPIRY_HOURS", "3"))

# --- state ----------------------------------------------------------------
_pending: list[dict] = []          # rows collected since the last HF flush
_pending_lock = threading.Lock()
_stats = {"ticks": 0, "forwarded": 0, "appended": 0, "last_tick_ts": None,
          "last_flush": None, "last_error": None}
feed: Optional[BinanceOptionsFeed] = None
_feed_task: Optional[asyncio.Task] = None
scheduler = AsyncIOScheduler()


def _on_row(row: dict, chain: list) -> None:
    """Called every poll cycle by the feed loop."""
    _stats["ticks"] += 1
    _stats["last_tick_ts"] = row.get("ts")

    with _pending_lock:
        _pending.append(row)

    # Live path: forward to the brain (fire-and-forget).
    if BRAIN_URL:
        ok = webhook.post_json(f"{BRAIN_URL}/tick", {"row": row, "chain": chain},
                               token=WEBHOOK_TOKEN)
        if ok:
            _stats["forwarded"] += 1


def _flush_to_hub() -> None:
    """Drain the pending buffer into the HF dataset (append new ts only)."""
    with _pending_lock:
        batch, _pending[:] = list(_pending), []
    if not batch:
        return
    try:
        res = store.append_rows(batch)
        if res.get("ok"):
            _stats["appended"] += res.get("added", 0)
            _stats["last_flush"] = datetime.now(timezone.utc).isoformat()
        else:
            # Re-queue on failure so nothing is lost before the next attempt.
            with _pending_lock:
                _pending[:0] = batch
            _stats["last_error"] = res.get("reason")
    except Exception as e:
        with _pending_lock:
            _pending[:0] = batch
        _stats["last_error"] = str(e)
        log.exception("Flush to hub failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global feed, _feed_task
    store.ensure_repo()
    feed = BinanceOptionsFeed(skip_same_day=SKIP_SAME_DAY,
                              min_expiry_hours=MIN_EXPIRY_HOURS)
    _feed_task = asyncio.create_task(feed.run_loop(_on_row, interval=POLL_INTERVAL))
    scheduler.add_job(_flush_to_hub, IntervalTrigger(minutes=FLUSH_MINUTES),
                      id="flush", replace_existing=True)
    scheduler.start()
    log.info("Collector started — poll %.0fs | flush %.0fmin | brain=%s",
             POLL_INTERVAL, FLUSH_MINUTES, BRAIN_URL or "NONE")
    yield
    if _feed_task:
        _feed_task.cancel()
    _flush_to_hub()           # best-effort final flush on shutdown
    scheduler.shutdown(wait=False)


app = FastAPI(title="BTC Collector", lifespan=lifespan)


@app.get("/health")
def health():
    with _pending_lock:
        pending = len(_pending)
    return {
        "ok": True,
        "service": "collector",
        "hub": store.hub_enabled(),
        "brain": bool(BRAIN_URL),
        "pending_rows": pending,
        **_stats,
    }
