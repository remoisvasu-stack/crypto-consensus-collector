"""app.py — Binance BTC options data recorder.

Self-contained 24/7 collector. Its only job is to record data and store it:
  * An internal loop polls the Binance Options chain every minute, builds one
    feature row (core/features), and appends it to training_history.csv.
  * That CSV is periodically uploaded to a Hugging Face dataset repo so it
    survives restarts (core/hf_store).

There is no model here — no training and no loading of trained models. Signals /
predictions are intentionally out of scope; this service collects and stores the
raw feature history for offline use.

- POST /tick     : manually inject one feature row (recorded like a polled row)
- GET  /health   : status (used as the UptimeRobot keep-alive target)
- GET  /         : Gradio dashboard (or JSON status when ENABLE_DASHBOARD=false)
"""
from __future__ import annotations
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI, HTTPException, Request, Header

from core.binance_feed import BinanceOptionsFeed
from core.data_store import TickBuffer
from core.hf_store import (
    restore_all_from_hub, schedule_persist,
    hub_enabled, ensure_repo_exists, DATA_DIR,
)

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s | %(message)s',
)
log = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HISTORY_PATH = DATA_DIR / 'training_history.csv'

WEBHOOK_TOKEN = os.getenv('WEBHOOK_TOKEN', '')

POLL_INTERVAL = float(os.getenv('POLL_INTERVAL', '60'))     # collector cadence, seconds
PERSIST_MIN   = float(os.getenv('PERSIST_MINUTES', '30'))   # HF Hub upload interval
# The Gradio dashboard is the heaviest dependency at runtime. Disable it on small
# hosts (e.g. 512MB free tiers) — /health and /tick still work without it.
ENABLE_DASHBOARD = os.getenv('ENABLE_DASHBOARD', 'true').lower() == 'true'
# Expiry selection. Default: skip the same-day (0DTE) expiry and use the next one,
# held until it settles. Set SKIP_SAME_DAY_EXPIRY=false to fall back to the
# hours-based rule (nearest expiry ≥ MIN_EXPIRY_HOURS out).
SKIP_SAME_DAY_EXPIRY = os.getenv('SKIP_SAME_DAY_EXPIRY', 'true').lower() == 'true'
MIN_EXPIRY_HOURS = float(os.getenv('MIN_EXPIRY_HOURS', '3'))

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
buffer: TickBuffer = TickBuffer()
feed: Optional[BinanceOptionsFeed] = None
_feed_task: Optional[asyncio.Task] = None
last_tick_ts: Optional[int] = None
rows_recorded: int = 0   # rows appended this session


# ---------------------------------------------------------------------------
# Feature-history data log (accumulates forward, persisted to HF Hub)
# ---------------------------------------------------------------------------
def append_to_history(row: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cols = list(row.keys())
    # If the feature schema changed (e.g. new features added), archive the old
    # file instead of corrupting it with mismatched columns.
    if HISTORY_PATH.exists():
        try:
            existing = pd.read_csv(HISTORY_PATH, nrows=0).columns.tolist()
        except Exception:
            existing = cols
        if existing != cols:
            bak = HISTORY_PATH.with_suffix('.csv.bak')
            HISTORY_PATH.replace(bak)
            log.warning("History schema changed (%d->%d cols); archived to %s",
                        len(existing), len(cols), bak.name)
    write_header = not HISTORY_PATH.exists()
    pd.DataFrame([row], columns=cols).to_csv(HISTORY_PATH, mode='a',
                                             header=write_header, index=False)


def _history_rows() -> Optional[int]:
    """Total rows persisted to training_history.csv (excludes header)."""
    if not HISTORY_PATH.exists():
        return 0
    try:
        with HISTORY_PATH.open() as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tick processing — collector loop and manual /tick both land here
# ---------------------------------------------------------------------------
def process_tick(row: dict) -> dict:
    global last_tick_ts, rows_recorded
    if not buffer.append(row):
        return {'status': 'duplicate', 'buffer_size': buffer.size()}

    try:
        append_to_history(row)
        rows_recorded += 1
    except Exception:
        log.exception("History append failed")
        return {'status': 'write_failed', 'buffer_size': buffer.size()}

    try:
        last_tick_ts = int(row['ts'])
    except (KeyError, ValueError, TypeError):
        pass

    return {
        'status': 'recorded',
        'ts': row.get('ts'),
        'spot': row.get('spot'),
        'atm_iv': row.get('atm_iv'),
        'buffer_size': buffer.size(),
    }


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global feed, _feed_task
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if hub_enabled():
        try:
            ensure_repo_exists()
            restore_all_from_hub()
        except Exception:
            log.exception("Hub restore failed")

    feed = BinanceOptionsFeed(skip_same_day=SKIP_SAME_DAY_EXPIRY,
                              min_expiry_hours=MIN_EXPIRY_HOURS)

    def _on_row(row: dict, chain: list) -> None:
        try:
            process_tick(row)
        except Exception:
            log.exception("process_tick failed")

    _feed_task = asyncio.create_task(feed.run_loop(_on_row, interval=POLL_INTERVAL))

    scheduler.add_job(schedule_persist, IntervalTrigger(minutes=PERSIST_MIN),
                      id='persist', replace_existing=True)
    scheduler.start()
    log.info("Started data recorder — poll %.0fs | persist %.0fmin | hub %s",
             POLL_INTERVAL, PERSIST_MIN, hub_enabled())
    yield
    if _feed_task:
        _feed_task.cancel()
    scheduler.shutdown(wait=False)


app = FastAPI(title="BTC Options Data Recorder", lifespan=lifespan)


@app.get('/health')
def health():
    return {
        'ok': True,
        'buffer': buffer.size(),
        'rows_recorded': rows_recorded,
        'history_rows': _history_rows(),
        'last_tick_ts': last_tick_ts,
        'hub': hub_enabled(),
    }


@app.post('/tick')
async def tick(request: Request,
               x_webhook_token: Optional[str] = Header(default=None)):
    if WEBHOOK_TOKEN and x_webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(401, 'bad token')
    payload = await request.json()
    if isinstance(payload, list):
        return {'results': [process_tick(r) for r in payload]}
    return process_tick(payload)


# ---------------------------------------------------------------------------
# Gradio dashboard
# ---------------------------------------------------------------------------
def _status() -> str:
    last = buffer.latest()
    last_str = "—"
    if last:
        when = (datetime.fromtimestamp(int(last['ts']) / 1000, tz=timezone.utc)
                .strftime('%Y-%m-%d %H:%M UTC')) if last.get('ts') else '?'
        last_str = (f"{when}  spot ${last.get('spot', '?'):,}  "
                    f"ATM IV {last.get('atm_iv', '?')}")
    return (
        f"### Status\n\n"
        f"- **Last tick**: {last_str}\n"
        f"- **Buffer**: {buffer.size()} rows in memory\n"
        f"- **Recorded**: {_history_rows()} rows in training_history.csv "
        f"({rows_recorded} this session)\n"
        f"- **Poll**: every {POLL_INTERVAL:.0f}s\n"
        f"- **Expiry**: {'skip 0DTE, use next' if SKIP_SAME_DAY_EXPIRY else f'≥ {MIN_EXPIRY_HOURS}h out'}\n"
        f"- **HF Hub**: {'✅ persisting' if hub_enabled() else '❌ local only'}\n"
    )


def _recent(n: int = 50) -> pd.DataFrame:
    cols = ['time', 'spot', 'atm_iv', 'oi_pcr', 'net_dir_score', 'n_contracts']
    df = buffer.to_frame()
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.tail(n).copy()
    df['time'] = (pd.to_datetime(df['ts'], unit='ms', utc=True)
                  .dt.strftime('%m-%d %H:%M'))
    keep = [c for c in cols if c in df.columns]
    return df[keep].iloc[::-1].reset_index(drop=True)


if ENABLE_DASHBOARD:
    import gradio as gr

    with gr.Blocks(title="BTC Options Data Recorder") as ui:
        gr.Markdown(
            "# BTC Options Data Recorder\n"
            "_24/7 collector for Binance BTC options — records feature rows to "
            "training_history.csv and stores them on the HF Hub._"
        )
        status_md = gr.Markdown(_status())
        refresh = gr.Button("🔄 Refresh", size='sm')
        recent_df = gr.Dataframe(_recent(), label="Recent recorded rows (newest first)")

        def _refresh():
            return _status(), _recent()

        refresh.click(_refresh, outputs=[status_md, recent_df])

    app = gr.mount_gradio_app(app, ui, path="/")
else:
    @app.get('/')
    def _root():
        return {
            'service': 'BTC Options Data Recorder',
            'buffer': buffer.size(),
            'rows_recorded': rows_recorded,
            'history_rows': _history_rows(),
            'last_tick_ts': last_tick_ts,
            'hub': hub_enabled(),
            'dashboard': 'disabled — set ENABLE_DASHBOARD=true to enable',
        }
