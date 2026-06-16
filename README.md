---
title: BTC Consensus Signals
emoji: 📈
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# BTC Consensus Signals

24/7 ML signal server for **Binance BTC options**.
A self-contained Python service — no Google Sheets, no Apps Script, no broker token.

> 10 XGBoost models predict ±0.15 / 0.30 / 0.50 / 0.75 / 1.10% moves within 2hr
> (calibrated for BTC; tunable). Three nested strategies fire when **every** model
> in their threshold subset predicts probability > 50%, optionally gated by ATM IV:
>
> - **3of5** → first 3 rungs
> - **4of5** → first 4 rungs
> - **5of5** → all 5 rungs

## How It Works

```
Binance Options API (eapi.binance.com)
  │
  │ every 60s: index + mark + OI + ticker
  ▼
core/binance_feed  ──►  core/features  ──►  models/consensus  ──►  signals.jsonl
  (collect + merge)     (15 features)      (10 models, vote)      + dashboard
                                                │
                                                ▼
                              /data (training_history.csv, models) → HF Hub
                              Gradio dashboard
```

Everything runs inside one Space: the collector loop polls Binance, builds a
feature row, and runs the (pre-trained) consensus model to emit a signal. Feature
rows accumulate forward into `training_history.csv` as a data log — option-chain
features have no Binance history, so this log is what you train on **offline**.
Drop the resulting model into the model dir (or the HF Hub dataset) and the server
loads it on startup; this service performs inference only, never training.

## Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/tick` | `X-Webhook-Token` | Manually inject a feature row (signal-only) |
| `GET` | `/health` | — | Status JSON (model, buffer) |
| `GET` | `/` | — | Gradio dashboard |

## Environment Variables

| Var | Required | Default | Purpose |
|-----|----------|---------|---------|
| `HF_TOKEN` | for persistence | — | HF Hub write token |
| `HF_DATASET_REPO` | no | `ndsideload/btc-consensus-data` | Hub dataset repo (fresh; BTC schema ≠ NIFTY) |
| `WEBHOOK_TOKEN` | no | — | Guards manual `/tick` |
| `CONSENSUS_THRESHOLDS` | no | `0.15,0.30,0.50,0.75,1.10` | %-move ladder |
| `CONSENSUS_HORIZON_MIN` | no | `120` | Forward horizon (min) |
| `IV_FLOOR` | no | `0.0` | ATM-IV vol gate (0 = off) |
| `PROB_THRESHOLD` | no | `0.50` | Model votes YES above this |
| `POLL_INTERVAL` | no | `60` | Collector cadence (s) |
| `PERSIST_MINUTES` | no | `30` | HF Hub upload interval |
| `SKIP_SAME_DAY_EXPIRY` | no | `true` | Skip 0DTE; use next expiry until it settles |
| `MIN_EXPIRY_HOURS` | no | `horizon/60 + 1` | Min hours to expiry (used only when skip-0DTE is off) |
| `ENABLE_DASHBOARD` | no | `true` | Set `false` on small hosts (Koyeb free 512MB) to skip Gradio |
| `BINANCE_EAPI` / `BINANCE_FAPI` / `BINANCE_SPOT` | no | Binance direct | Proxy base URLs if the host is geo-blocked (HTTP 451) |
| `BINANCE_PROXY_SECRET` | no | — | Shared secret sent to the proxy |

## Files

```
app.py                       FastAPI server + collector loop + Gradio dashboard
Dockerfile                   Container build
requirements.txt             Python deps
core/
  binance_feed.py            Binance Options collector (async) + fetch_klines
  features.py                15-feature option-chain row builder
  data_store.py              TickBuffer (continuous, minute-bucket dedup)
  hf_store.py                HF Hub persistence (models, history)
models/
  consensus.py               10 XGBoost classifiers + 3of5/4of5/5of5 consensus (inference)
scripts/
  backtest_thresholds.py     Threshold/horizon calibration from historical klines
```

## Deployment

1. Push to an HF Space (Docker SDK).
2. Set `HF_TOKEN` (and optionally `WEBHOOK_TOKEN`) in Space secrets.
3. Place a pre-trained model under the model dir (or in the HF Hub dataset repo so
   it's restored on startup). Without one, the server collects data and reports
   `model_loaded: false` until a model is added.
4. The collector starts automatically and accumulates `training_history.csv` as the
   data log to train future models on offline.
5. **Keep-alive:** the collector is an internal 24/7 loop, so the Space must stay awake.
   A free Space sleeps without inbound HTTP — point a free uptime monitor at `/health`,
   or use always-on (paid) hardware.

## Deploy on Koyeb (free, non-US region)

HF Spaces are US-hosted, so Binance answers them with 451. To run free in an
allowed region, deploy the *same Dockerfile* to Koyeb's **Frankfurt** region:

1. Push this repo to GitHub.
2. Koyeb → Create Service → GitHub → your repo → Builder: **Dockerfile** →
   Region: **Frankfurt** → **Free** instance → Port `7860`.
3. Env: `HF_TOKEN`, `HF_DATASET_REPO`, and `ENABLE_DASHBOARD=false`
   (fits the 512MB free RAM). **No `BINANCE_*` proxy needed** — Frankfurt reaches
   Binance directly.
4. Keep-alive: point a free UptimeRobot monitor at `/health` (the free tier
   scales to zero after ~1h idle).

View data with `ENABLE_DASHBOARD=false`: `GET /health` or the HF dataset repo files
(`signals.jsonl`, `training_history.csv`).

## Geo-blocked host (HTTP 451)

Binance answers some datacenter regions (incl. the US region where HF Spaces run)
with HTTP 451. If the collector logs `Binance geo-blocked (451)`, run
`scripts/binance_proxy.py` on a host in an allowed region (home PC / Raspberry Pi
/ a free VM such as Oracle Cloud Always-Free in Mumbai), then set `BINANCE_EAPI`,
`BINANCE_FAPI`, `BINANCE_SPOT` (all to the proxy URL) and `BINANCE_PROXY_SECRET`
on the Space. See the header of `scripts/binance_proxy.py` for full steps.

## Calibration

`python scripts/backtest_thresholds.py --days 60` measures BTC move base rates per
horizon to tune `CONSENSUS_THRESHOLDS`. Re-run periodically; re-tune on real model
precision once option-chain features have accumulated (base rates ≠ model skill).
