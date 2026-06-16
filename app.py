"""app.py — BTC Consensus Signal Server on Hugging Face Spaces.

Self-contained 24/7 inference service for Binance BTC options:
  * An internal collector loop polls the Binance Options chain every minute,
    builds a feature row, and runs the (pre-trained) consensus model.
  * Feature rows accumulate forward into training_history.csv as a data log —
    models are trained offline and loaded from the model dir / HF Hub.

- POST /tick     : manually inject one feature row (signal-only)
- GET  /health   : status
- GET  /         : Gradio dashboard
"""
from __future__ import annotations
import asyncio
import json
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
from models.consensus import (
    ConsensusModel, ConsensusPredictor,
    DEFAULT_MODEL_DIR, latest_model_dir,
    add_engineered_features, STRATEGIES, THRESHOLDS, HORIZON_MIN,
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
SIGNALS_PATH = DATA_DIR / 'signals.jsonl'
HISTORY_PATH = DATA_DIR / 'training_history.csv'

IV_FLOOR       = float(os.getenv('IV_FLOOR', '0.0'))         # ATM-IV vol gate (0 = off)
PROB_THRESHOLD = float(os.getenv('PROB_THRESHOLD', '0.50'))  # model votes YES if prob > 50%
WEBHOOK_TOKEN  = os.getenv('WEBHOOK_TOKEN', '')

POLL_INTERVAL  = float(os.getenv('POLL_INTERVAL', '60'))     # collector cadence, seconds
PERSIST_MIN    = float(os.getenv('PERSIST_MINUTES', '30'))
# The Gradio dashboard is the heaviest dependency at runtime. Disable it on small
# hosts (e.g. Koyeb free 512MB) — /health and /tick still work without it.
ENABLE_DASHBOARD = os.getenv('ENABLE_DASHBOARD', 'true').lower() == 'true'
# Expiry selection. Default: skip the same-day (0DTE) expiry and use the next one,
# held until it settles. Set SKIP_SAME_DAY_EXPIRY=false to fall back to the
# hours-based rule (nearest expiry ≥ MIN_EXPIRY_HOURS out).
SKIP_SAME_DAY_EXPIRY = os.getenv('SKIP_SAME_DAY_EXPIRY', 'true').lower() == 'true'
MIN_EXPIRY_HOURS = float(os.getenv('MIN_EXPIRY_HOURS', str(max(2.0, HORIZON_MIN / 60 + 1))))
WARMUP = 31  # rows needed before engineered (30-min) features are valid

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
buffer: TickBuffer = TickBuffer()
predictor: Optional[ConsensusPredictor] = None
model_meta: dict = {}
feed: Optional[BinanceOptionsFeed] = None
_feed_task: Optional[asyncio.Task] = None
last_tick_ts: Optional[int] = None


def _load_predictor() -> bool:
    global predictor, model_meta
    d = latest_model_dir(DEFAULT_MODEL_DIR)
    if d is None:
        log.warning("No models found at %s", DEFAULT_MODEL_DIR)
        predictor, model_meta = None, {}
        return False
    model = ConsensusModel.load(d)
    predictor = ConsensusPredictor(model=model)
    model_meta = {
        'version': d.name,
        'trained_at': model.trained_at,
        'train_rows': model.train_rows,
        'n_dates': len(model.train_dates),
        'first_date': model.train_dates[0] if model.train_dates else None,
        'last_date': model.train_dates[-1] if model.train_dates else None,
    }
    log.info("Loaded model %s (%d rows)", d.name, model.train_rows)
    return True


# ---------------------------------------------------------------------------
# Feature-history data log (accumulates forward; used for offline training)
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


# ---------------------------------------------------------------------------
# Tick processing — collector loop and manual /tick both land here
# ---------------------------------------------------------------------------
def process_tick(row: dict) -> dict:
    global last_tick_ts
    if not buffer.append(row):
        return {'status': 'duplicate', 'buffer_size': buffer.size()}

    try:
        append_to_history(row)
    except Exception:
        log.exception("History append failed")

    try:
        spot = float(row['spot'])
        atm_iv = float(row['atm_iv'])
        ts = int(row['ts'])
    except (KeyError, ValueError, TypeError) as e:
        return {'status': 'bad_payload', 'fired': False, 'error': str(e),
                'buffer_size': buffer.size()}
    last_tick_ts = ts
    ts_iso = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()

    if predictor is None:
        return {'status': 'no_model', 'fired': False, 'buffer_size': buffer.size()}

    df_buf = buffer.to_frame()
    if len(df_buf) < WARMUP:
        return {'status': 'warmup', 'fired': False, 'buffer_size': len(df_buf)}

    df_buf = add_engineered_features(df_buf)
    last = df_buf.iloc[-1]

    feat_dict, missing = {}, []
    for f in predictor.model.feature_names:
        if f in last.index:
            feat_dict[f] = float(last[f])
        else:
            missing.append(f)
    if missing:
        return {'status': 'missing_features', 'fired': False, 'missing': missing[:5]}

    result = predictor.evaluate(
        features=feat_dict, spot=spot, atm_iv=atm_iv, timestamp=ts_iso,
        prob_threshold=PROB_THRESHOLD, iv_floor=IV_FLOOR,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with SIGNALS_PATH.open('a') as f:
        f.write(result.to_json() + '\n')

    confidence = (result.up_confidence
                  if result.direction != 'DOWN' else result.down_confidence)
    return {
        'status': 'ok', 'fired': result.fired, 'direction': result.direction,
        'confidence': confidence,
        'up_confidence': result.up_confidence, 'down_confidence': result.down_confidence,
        'conf_spread': result.conf_spread,
        'up_votes': result.up_votes, 'dn_votes': result.dn_votes,
        'strategies': result.strategies, 'reasons': result.reasons,
        'spot': spot, 'atm_iv': atm_iv,
        'rank_pct': result.rank_pct, 'raw_probabilities': result.raw_probabilities,
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

    _load_predictor()
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
    log.info("Started — model: %s | poll %.0fs",
             model_meta.get('version', 'NONE'), POLL_INTERVAL)
    yield
    if _feed_task:
        _feed_task.cancel()
    scheduler.shutdown(wait=False)


app = FastAPI(title="BTC Consensus Signals", lifespan=lifespan)


@app.get('/health')
def health():
    return {
        'ok': True,
        'model': model_meta.get('version'),
        'model_loaded': predictor is not None,
        'buffer': buffer.size(),
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
    m = model_meta
    model_str = (f"{m.get('version', 'NONE')} • {m.get('train_rows', '?')} rows • "
                 f"{m.get('n_dates', '?')} days ({m.get('first_date', '?')} → {m.get('last_date', '?')})")
    return (
        f"### Status\n\n"
        f"- **Last tick**: {last_str}\n"
        f"- **Buffer**: {buffer.size()} rows\n"
        f"- **Model**: {model_str}\n"
        f"- **Ladder**: {THRESHOLDS}% • horizon {HORIZON_MIN}min\n"
        f"- **IV gate**: ≥ {IV_FLOOR}  •  **Vote**: model prob > {PROB_THRESHOLD * 100:.0f}%\n"
        f"- **Strategies**: {', '.join(STRATEGIES)} (unanimous threshold subset)\n"
        f"- **HF Hub**: {'✅' if hub_enabled() else '❌'}\n"
    )


def _signals(n: int = 50) -> pd.DataFrame:
    if not SIGNALS_PATH.exists():
        return pd.DataFrame(columns=['timestamp', 'spot', 'atm_iv', 'fired'])
    rows = []
    with SIGNALS_PATH.open() as f:
        for line in f.readlines()[-n:]:
            try:
                r = json.loads(line)
                rows.append({
                    'timestamp': r['timestamp'],
                    'spot': r['spot'],
                    'atm_iv': r['atm_iv'],
                    'dir': r.get('direction') or '—',
                    'fired': '🟢' if r['fired'] else '–',
                    'reasons': ', '.join(r.get('reasons', [])) or '—',
                })
            except Exception:
                continue
    df = pd.DataFrame(rows)
    return df.iloc[::-1] if not df.empty else df


if ENABLE_DASHBOARD:
    import gradio as gr

    with gr.Blocks(title="BTC Consensus Signals") as ui:
        gr.Markdown(
            "# BTC Consensus Signals\n"
            "_24/7 ML signal server for Binance BTC options._"
        )
        status_md = gr.Markdown(_status())
        refresh = gr.Button("🔄 Refresh", size='sm')
        signals_df = gr.Dataframe(_signals(), label="Recent signals (newest first)")

        def _refresh():
            return _status(), _signals()

        refresh.click(_refresh, outputs=[status_md, signals_df])

    app = gr.mount_gradio_app(app, ui, path="/")
else:
    @app.get('/')
    def _root():
        return {
            'service': 'BTC Consensus Signals',
            'model': model_meta.get('version'),
            'model_loaded': predictor is not None,
            'buffer': buffer.size(),
            'last_tick_ts': last_tick_ts,
            'dashboard': 'disabled — set ENABLE_DASHBOARD=true to enable',
        }
