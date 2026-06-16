---
title: BTC Options Data Recorder
emoji: 📈
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# BTC Options Data Recorder

24/7 data recorder for **Binance BTC options**.
A self-contained Python service — no model, no training, no broker token. Its only
job is to **record feature rows and store them**: poll the option chain every
minute, build one feature row, append it to `training_history.csv`, and back that
file up to a Hugging Face dataset repo so it survives restarts.

## How It Works

```
Binance Options API (eapi.binance.com)
  │
  │ every 60s: index + mark + OI + ticker
  ▼
core/binance_feed  ──►  core/features  ──►  training_history.csv  ──►  HF Hub
  (collect + merge)     (per-minute row)    (append, /data)            (periodic upload)
                                                │
                                                ▼
                                        Gradio dashboard / /health
```

Everything runs in one process: the collector loop polls Binance, builds a feature
row, and appends it to `training_history.csv`. The file is uploaded to the HF Hub
dataset repo every `PERSIST_MINUTES` and restored on startup, so the history keeps
accumulating across restarts. There is **no prediction, no model loading, and no
training** — you train offline on the recorded CSV if/when you want.

## Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/tick` | `X-Webhook-Token` | Manually inject one feature row (recorded like a polled row) |
| `GET` | `/health` | — | Status JSON — use this as the UptimeRobot keep-alive target |
| `GET` | `/` | — | Gradio dashboard (or JSON status when `ENABLE_DASHBOARD=false`) |

## Environment Variables

| Var | Required | Default | Purpose |
|-----|----------|---------|---------|
| `HF_TOKEN` | for persistence | — | HF Hub write token. Without it, data stays local only. |
| `HF_DATASET_REPO` | no | `ndsideload/btc-consensus-data` | Dataset repo to store `training_history.csv`. Use your own (e.g. `youruser/btc-data`). |
| `WEBHOOK_TOKEN` | no | — | Guards manual `/tick` |
| `POLL_INTERVAL` | no | `60` | Collector cadence (s) |
| `PERSIST_MINUTES` | no | `30` | HF Hub upload interval |
| `SKIP_SAME_DAY_EXPIRY` | no | `true` | Skip 0DTE; use next expiry until it settles |
| `MIN_EXPIRY_HOURS` | no | `3` | Min hours to expiry (used only when skip-0DTE is off) |
| `ENABLE_DASHBOARD` | no | `true` | Set `false` on small hosts (512MB free tiers) to skip Gradio |
| `BINANCE_EAPI` / `BINANCE_FAPI` / `BINANCE_SPOT` | no | Binance direct | Proxy base URLs if the host is geo-blocked (HTTP 451) |
| `BINANCE_PROXY_SECRET` | no | — | Shared secret sent to the proxy |

## Files

```
app.py                       FastAPI server + collector loop + Gradio dashboard
Dockerfile                   Container build
requirements.txt             Python deps
core/
  binance_feed.py            Binance Options collector (async) + fetch_klines
  features.py                per-minute option-chain feature-row builder
  data_store.py              TickBuffer (continuous, minute-bucket dedup)
  hf_store.py                HF Hub persistence (training_history.csv)
scripts/
  backtest_thresholds.py     Offline analysis: BTC move base rates from historical klines
  binance_proxy.py           Optional proxy for geo-blocked hosts
```

## Deployment

1. Push to an HF Space (Docker SDK) or any Docker host (Render, Koyeb, Fly…).
2. Set `HF_TOKEN` and `HF_DATASET_REPO` (and optionally `WEBHOOK_TOKEN`) as secrets.
3. The collector starts automatically and begins appending to `training_history.csv`,
   uploading it to the dataset repo every `PERSIST_MINUTES`.
4. **Keep-alive:** the collector is an internal 24/7 loop, so the host must stay awake.
   Free tiers sleep without inbound HTTP — point a free uptime monitor (UptimeRobot)
   at `/health`, or use always-on hardware.

## Deploy on Koyeb (free, non-US region)

HF Spaces and many US datacenters get HTTP 451 from Binance. To run free in an
allowed region, deploy the *same Dockerfile* to Koyeb's **Frankfurt** region:

1. Push this repo to GitHub.
2. Koyeb → Create Service → GitHub → your repo → Builder: **Dockerfile** →
   Region: **Frankfurt** → **Free** instance → Port `7860`.
3. Env: `HF_TOKEN`, `HF_DATASET_REPO`, and `ENABLE_DASHBOARD=false`
   (fits the 512MB free RAM). **No `BINANCE_*` proxy needed** — Frankfurt reaches
   Binance directly.
4. Keep-alive: point a free UptimeRobot monitor at `/health` (the free tier
   scales to zero after ~1h idle).

View data with `ENABLE_DASHBOARD=false`: `GET /health`, or the HF dataset repo file
`training_history.csv`.

## Geo-blocked host (HTTP 451)

Binance answers some datacenter regions (incl. the US region where HF Spaces run)
with HTTP 451. If the collector logs `Binance geo-blocked (451)`, run
`scripts/binance_proxy.py` on a host in an allowed region (home PC / Raspberry Pi
/ a free VM such as Oracle Cloud Always-Free in Mumbai), then set `BINANCE_EAPI`,
`BINANCE_FAPI`, `BINANCE_SPOT` (all to the proxy URL) and `BINANCE_PROXY_SECRET`
on the host. See the header of `scripts/binance_proxy.py` for full steps.

## Offline analysis

`python scripts/backtest_thresholds.py --days 60` measures BTC move base rates per
horizon from historical klines — useful when you later decide how to label/train on
the recorded data. It's a standalone script and is not part of the running recorder.
